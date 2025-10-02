# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-01 17:00:25.916671 PST.

### Artificial Intelligence

### 1. [SMS: Self-supervised Model Seeding for Verification of Machine Unlearning](http://arxiv.org/pdf/2509.25613v1)

Authors: Weiqi Wang, Chenhan Zhang, Zhiyi Tian, Shui Yu

Many machine unlearning methods have been proposed recently to uphold users'
right to be forgotten. However, offering users verification of their data
removal post-unlearning is an important yet under-explored problem. Current
verifications typically rely on backdooring, i.e., adding backdoored samples to
influence model performance. Nevertheless, the backdoor methods can merely
establish a connection between backdoored samples and models but fail to
connect the backdoor with genuine samples. Thus, the backdoor removal can only
confirm the unlearning of backdoored samples, not users' genuine samples, as
genuine samples are independent of backdoored ones. In this paper, we propose a
Self-supervised Model Seeding (SMS) scheme to provide unlearning verification
for genuine samples. Unlike backdooring, SMS links user-specific seeds (such as
users' unique indices), original samples, and models, thereby facilitating the
verification of unlearning genuine samples. However, implementing SMS for
unlearning verification presents two significant challenges. First, embedding
the seeds into the service model while keeping them secret from the server
requires a sophisticated approach. We address this by employing a
self-supervised model seeding task, which learns the entire sample, including
the seeds, into the model's latent space. Second, maintaining the utility of
the original service model while ensuring the seeding effect requires a
delicate balance. We design a joint-training structure that optimizes both the
self-supervised model seeding task and the primary service task simultaneously
on the model, thereby maintaining model utility while achieving effective model
seeding. The effectiveness of the proposed SMS scheme is evaluated through
extensive experiments, which demonstrate that SMS provides effective
verification for genuine sample unlearning, addressing existing limitations.

### 2. [SOCK: A Benchmark for Measuring Self-Replication in Large Language Models](http://arxiv.org/pdf/2509.25643v1)

Authors: Justin Chavarria, Rohan Raizada, Justin White, Eyad Alhetairshi

We introduce SOCK, a benchmark command line interface (CLI) that measures
large language models' (LLMs) ability to self-replicate without human
intervention. In this benchmark, self-replication is defined not only as an
LLM's ability to create a functioning and running copy of itself, but also the
ability for that self-replication to persist and occur across different
computational contexts. Accordingly, we've developed a system to categorize
LLMs based on broad self-replication capabilities in two general classes,
Replication-Capability Levels (RCL) and Persistence-Capability Levels (PCL).
Using a five-task suite based on practically manipulable modern CLI utilities
and computer processes, experiments are orchestrated in a controlled
environment with an LLM acting agentically. The performance of the LLM on agent
tasks is then computed to produce an R-score (a quantitative evaluation of
overall self-replication ability) and data used to categorize LLMs into
specific RCL-PCL matrices. SOCK offers two primary contributions: (1) Provides
the first formalized definitions and benchmark suite for evaluating LLM
self-replication, with the goal of establishing a standard for future research,
to our knowledge; (2) Allows the industry to track the effectiveness of future
multi-agent systems and mitigate potential self-replication threat vectors
within them. The results compiled from evaluating a variety of open-weight and
proprietary frontier models reveal significant obstacles to persistent
self-replication and multi-agent systems, including context retention and
multi-agent decision-making. We propose future research directions to safely
reduce the severity of these obstacles, potentially lowering future risk of
more functional multi-agent systems.

### 3. [AutoLabs: Cognitive Multi-Agent Systems with Self-Correction for Autonomous Chemical Experimentation](http://arxiv.org/pdf/2509.25651v1)

Authors: Gihan Panapitiya, Emily Saldanha, Heather Job, Olivia Hess

The automation of chemical research through self-driving laboratories (SDLs)
promises to accelerate scientific discovery, yet the reliability and granular
performance of the underlying AI agents remain critical, under-examined
challenges. In this work, we introduce AutoLabs, a self-correcting, multi-agent
architecture designed to autonomously translate natural-language instructions
into executable protocols for a high-throughput liquid handler. The system
engages users in dialogue, decomposes experimental goals into discrete tasks
for specialized agents, performs tool-assisted stoichiometric calculations, and
iteratively self-corrects its output before generating a hardware-ready file.
We present a comprehensive evaluation framework featuring five benchmark
experiments of increasing complexity, from simple sample preparation to
multi-plate timed syntheses. Through a systematic ablation study of 20 agent
configurations, we assess the impact of reasoning capacity, architectural
design (single- vs. multi-agent), tool use, and self-correction mechanisms. Our
results demonstrate that agent reasoning capacity is the most critical factor
for success, reducing quantitative errors in chemical amounts (nRMSE) by over
85% in complex tasks. When combined with a multi-agent architecture and
iterative self-correction, AutoLabs achieves near-expert procedural accuracy
(F1-score > 0.89) on challenging multi-step syntheses. These findings establish
a clear blueprint for developing robust and trustworthy AI partners for
autonomous laboratories, highlighting the synergistic effects of modular
design, advanced reasoning, and self-correction to ensure both performance and
reliability in high-stakes scientific applications. Code:
https://github.com/pnnl/autolabs

### 4. [Landmark-Guided Knowledge for Vision-and-Language Navigation](http://arxiv.org/pdf/2509.25655v1)

Authors: Dongsheng Yang, Meiling Zhu, Yinfeng Yu

Vision-and-language navigation is one of the core tasks in embodied
intelligence, requiring an agent to autonomously navigate in an unfamiliar
environment based on natural language instructions. However, existing methods
often fail to match instructions with environmental information in complex
scenarios, one reason being the lack of common-sense reasoning ability. This
paper proposes a vision-and-language navigation method called Landmark-Guided
Knowledge (LGK), which introduces an external knowledge base to assist
navigation, addressing the misjudgment issues caused by insufficient common
sense in traditional methods. Specifically, we first construct a knowledge base
containing 630,000 language descriptions and use knowledge Matching to align
environmental subviews with the knowledge base, extracting relevant descriptive
knowledge. Next, we design a Knowledge-Guided by Landmark (KGL) mechanism,
which guides the agent to focus on the most relevant parts of the knowledge by
leveraging landmark information in the instructions, thereby reducing the data
bias that may arise from incorporating external knowledge. Finally, we propose
Knowledge-Guided Dynamic Augmentation (KGDA), which effectively integrates
language, knowledge, vision, and historical information. Experimental results
demonstrate that the LGK method outperforms existing state-of-the-art methods
on the R2R and REVERIE vision-and-language navigation datasets, particularly in
terms of navigation error, success rate, and path efficiency.

### 5. [GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination](http://arxiv.org/pdf/2509.25669v1)

Authors: Xinxi Chen, Tianyang Chen, Lijia Hong

We propose a method to improve Visual Question Answering (VQA) with
Retrieval-Augmented Generation (RAG) by introducing text-grounded object
localization. Rather than retrieving information based on the entire image, our
approach enables the model to generate a bounding box around the object most
relevant to the question, allowing for targeted image cropping and focused
retrieval. This reduces background noise, improves alignment between visual and
textual cues, and helps mitigate hallucinations. Our RAG method enhances
context-aware VQA responses increased the accuracy from 22.19% to 25.64%, with
an absolute increase of 3.45 percentage points, compared to the baseline
Llama-3.2-Vision-11B agent. We also proposed a de-hallucination method based on
question type which can effectively reduce the hallucination rate from 65.79%
to 13.88% and improves the truthfulness score.

### 6. [ScheduleMe: Multi-Agent Calendar Assistant](http://arxiv.org/pdf/2509.25693v1)

Authors: N. de Silva, S. Perera, K. L. A. A. Nimasha, I. D. S. Fernando, R. K. A. O. Wijerathne

Recent advancements in LLMs have contributed to the rise of advanced
conversational assistants that can assist with user needs through natural
language conversation. This paper presents a ScheduleMe, a multi-agent calendar
assistant for users to manage google calendar events in natural language. The
system uses a graph-structured coordination mechanism where a central
supervisory agent supervises specialized task agents, allowing modularity,
conflicts resolution, and context-aware interactions to resolve ambiguities and
evaluate user commands. This approach sets an example of how structured
reasoning and agent cooperation might convince operators to increase the
usability and flexibility of personal calendar assistant tools.

### 7. [Cooperative Autonomous Driving in Diverse Behavioral Traffic: A Heterogeneous Graph Reinforcement Learning Approach](http://arxiv.org/pdf/2509.25751v1)

Authors: Qi Liu, Xueyuan Li, Zirui Li, Juhui Gim

Navigating heterogeneous traffic environments with diverse driving styles
poses a significant challenge for autonomous vehicles (AVs) due to their
inherent complexity and dynamic interactions. This paper addresses this
challenge by proposing a heterogeneous graph reinforcement learning (GRL)
framework enhanced with an expert system to improve AV decision-making
performance. Initially, a heterogeneous graph representation is introduced to
capture the intricate interactions among vehicles. Then, a heterogeneous graph
neural network with an expert model (HGNN-EM) is proposed to effectively encode
diverse vehicle features and produce driving instructions informed by
domain-specific knowledge. Moreover, the double deep Q-learning (DDQN)
algorithm is utilized to train the decision-making model. A case study on a
typical four-way intersection, involving various driving styles of human
vehicles (HVs), demonstrates that the proposed method has superior performance
over several baselines regarding safety, efficiency, stability, and convergence
rate, all while maintaining favorable real-time performance.

### 8. [Thinking Sparks!: Emergent Attention Heads in Reasoning Models During Post Training](http://arxiv.org/pdf/2509.25758v1)

Authors: Yein Park, Minbyul Jeong, Jaewoo Kang

The remarkable capabilities of modern large reasoning models are largely
unlocked through post-training techniques such as supervised fine-tuning and
reinforcement learning. However, the architectural mechanisms behind such
improvements remain largely opaque. In this work, we use circuit analysis to
demonstrate that post-training for complex reasoning sparks the emergence of
novel, functionally specialized attention heads. These heads collectively
support structured reasoning and computation. Our comparative analysis across
Qwen families and DeepSeek-distilled model reveals that these emergent heads
evolve differently under different training regimes. Distillation and SFT
foster a cumulative addition of stable reasoning heads. In contrast, group
relative policy optimization operates in a dynamic search mode: relatively few
attention heads are iteratively activated, evaluated, and pruned, with their
survival closely tracking fluctuations in the task reward signal. Furthermore,
we find that controllable think on/off models do not possess dedicated thinking
heads. Instead, turning off explicit reasoning triggers a broader-but less
efficient-set of compensatory heads. Through ablation and qualitative analyses,
we connect these circuit-level dynamics to a crucial performance trade-off:
strengthened heads enable sophisticated problem-solving strategies for
difficult problems but can also introduce over-thinking failure modes, such as
calculation errors or logical loops on simpler tasks. These findings connect
circuit-level dynamics to macro-level performance, identifying an inherent
tension where complex reasoning comes at the cost of elementary computations.
More broadly, our work points to future directions for training policy design,
emphasizing the need to balance the development of effective reasoning
strategies with the assurance of reliable, flawless execution.

### 9. [Galton's Law of Mediocrity: Why Large Language Models Regress to the Mean and Fail at Creativity in Advertising](http://arxiv.org/pdf/2509.25767v1)

Authors: Matt Keon, Aabid Karim, Bhoomika Lohana, Abdul Karim, Thai Nguyen, Tara Hamilton, Ali Abbas

Large language models (LLMs) generate fluent text yet often default to safe,
generic phrasing, raising doubts about their ability to handle creativity. We
formalize this tendency as a Galton-style regression to the mean in language
and evaluate it using a creativity stress test in advertising concepts. When ad
ideas were simplified step by step, creative features such as metaphors,
emotions, and visual cues disappeared early, while factual content remained,
showing that models favor high-probability information. When asked to
regenerate from simplified inputs, models produced longer outputs with lexical
variety but failed to recover the depth and distinctiveness of the originals.
We combined quantitative comparisons with qualitative analysis, which revealed
that the regenerated texts often appeared novel but lacked true originality.
Providing ad-specific cues such as metaphors, emotional hooks and visual
markers improved alignment and stylistic balance, though outputs still relied
on familiar tropes. Taken together, the findings show that without targeted
guidance, LLMs drift towards mediocrity in creative tasks; structured signals
can partially counter this tendency and point towards pathways for developing
creativity-sensitive models.

### 10. [Planner-R1: Reward Shaping Enables Efficient Agentic RL with Smaller LLMs](http://arxiv.org/pdf/2509.25779v1)

Authors: Siyu Zhu, Yanbin Jiang, Hejian Sang, Shao Tang, Qingquan Song, Biao He, Rohit Jain, Zhipeng Wang, Alborz Geramifard

We investigated Agentic RL with large language models on the
\textsc{TravelPlanner} benchmark. Our approach, \textsc{Planner-R1}, achieved a
\textbf{56.9\%} final-pass rate with only 180 training queries, a $2.7\times$
improvement over GPT-5's $21.2\%$ baseline and the strongest agentic result on
the public leaderboard. A central finding was that smaller models (8B) were
highly responsive to reward shaping: with dense process-level signals, they
reached competitive performance while being $3.5\times$ more compute-efficient
and $1.5\times$ more memory-efficient than 32B models. Larger models were more
robust under sparse rewards but exhibited smaller relative gains from shaping
and higher variance across runs. While curriculum learning offered no
significant benefit, shaped rewards consistently amplified learning dynamics,
making 8B models the most efficient setting for agentic RL. Crucially, these
gains did not come at the cost of overfitting: fine-tuned models mostly
maintained or exceeded baseline performance on out-of-domain tasks, including
\textsc{Multi-IF}, \textsc{NaturalPlan}, and $\tau$-\textsc{Bench}. These
results establish reward shaping as a decisive lever for scaling agentic RL,
highlight the competitive strength of smaller models, and demonstrate that
efficiency can be achieved without sacrificing generalization.

### Hardware Architecture

### 1. [LLM-Powered Code Analysis and Optimization for Gaussian Splatting Kernels](http://arxiv.org/pdf/2509.25626v1)

Authors: Yi Hu, Huiyang Zhou

3D Gaussian splatting (3DGS) is a transformative technique with profound
implications on novel view synthesis and real-time rendering. Given its
importance, there have been many attempts to improve its performance. However,
with the increasing complexity of GPU architectures and the vast search space
of performance-tuning parameters, it is a challenging task. Although manual
optimizations have achieved remarkable speedups, they require domain expertise
and the optimization process can be highly time consuming and error prone. In
this paper, we propose to exploit large language models (LLMs) to analyze and
optimize Gaussian splatting kernels. To our knowledge, this is the first work
to use LLMs to optimize highly specialized real-world GPU kernels. We reveal
the intricacies of using LLMs for code optimization and analyze the code
optimization techniques from the LLMs. We also propose ways to collaborate with
LLMs to further leverage their capabilities. For the original 3DGS code on the
MipNeRF360 datasets, LLMs achieve significant speedups, 19% with Deepseek and
24% with GPT-5, demonstrating the different capabilities of different LLMs. By
feeding additional information from performance profilers, the performance
improvement from LLM-optimized code is enhanced to up to 42% and 38% on
average. In comparison, our best-effort manually optimized version can achieve
a performance improvement up to 48% and 39% on average, showing that there are
still optimizations beyond the capabilities of current LLMs. On the other hand,
even upon a newly proposed 3DGS framework with algorithmic optimizations,
Seele, LLMs can still further enhance its performance by 6%, showing that there
are optimization opportunities missed by domain experts. This highlights the
potential of collaboration between domain experts and LLMs.

### 2. [SAIL: SRAM-Accelerated LLM Inference System with Lookup-Table-based GEMV](http://arxiv.org/pdf/2509.25853v1)

Authors: Jingyao Zhang, Jaewoo Park, Jongeun Lee, Elaheh Sadredini

Large Language Model (LLM) inference requires substantial computational
resources, yet CPU-based inference remains essential for democratizing AI due
to the widespread availability of CPUs compared to specialized accelerators.
However, efficient LLM inference on CPUs faces two fundamental challenges: (1)
existing CPU architectures struggle with low-precision arithmetic required by
quantized models, where optimal bit precision varies across models and layers;
and (2) the memory-bound nature of the token generation phase creates severe
performance bottlenecks. To address these challenges, we propose SAIL
(SRAM-Accelerated Inference of LLMs), a CPU-based inference solution that
efficiently supports arbitrary bit precisions with minimal overhead. SAIL
integrates three key innovations: First, we introduce Batched LUT-based General
Matrix-Vector Multiplication (LUT-GEMV) with SRAM-based processing-in-memory,
enabling high data reuse through lookup tables and reducing memory movement.
Second, our Pattern-Aware LUT optimization identifies and exploits redundancy
in input activation patterns, reducing computation cycles by 13.8\%. Third, we
develop an in-memory type conversion algorithm that leverages PIM's parallelism
for efficient de-/quantization operations, alleviating pressure on CPU's vector
units. Our architecture requires only 2\% hardware overhead and a single new
instruction, while maintaining dual functionality as both compute and storage
units. Experimental evaluations using a modified gem5 simulator demonstrate
that SAIL achieves up to 10.7x speedup and 19.9x higher tokens per dollar
compared to ARM Neoverse-N1 CPU baselines, and up to 7.04x better cost
efficiency than NVIDIA V100 GPUs, establishing a practical path for efficient
CPU-based LLM inference.

### 3. [Runtime Energy Monitoring for RISC-V Soft-Cores](http://arxiv.org/pdf/2509.26065v1)

Authors: Alberto Scionti, Paolo Savio, Francesco Lubrano, Olivier Terzo, Marco Ferretti, Florin Apopei, Juri Bellucci, Ennio Spano, Luca Carriere

Energy efficiency is one of the major concern in designing advanced computing
infrastructures. From single nodes to large-scale systems (data centers),
monitoring the energy consumption of the computing system when applications run
is a critical task. Designers and application developers often rely on software
tools and detailed architectural models to extract meaningful information and
determine the system energy consumption. However, when a design space
exploration is required, designers may incur in continuous tuning of the models
to match with the system under evaluation. To overcome such limitations, we
propose a holistic approach to monitor energy consumption at runtime without
the need of running complex (micro-)architectural models. Our approach is based
on a measurement board coupled with a FPGA-based System-on-Module. The
measuring board captures currents and voltages (up to tens measuring points)
driving the FPGA and exposes such values through a specific memory region. A
running service reads and computes energy consumption statistics without
consuming extra resources on the FPGA device. Our approach is also scalable to
monitoring of multi-nodes infrastructures (clusters). We aim to leverage this
framework to perform experiments in the context of an aeronautical design
application; specifically, we will look at optimizing performance and energy
consumption of a shallow artificial neural network on RISC-V based soft-cores.

### 4. [MUSS-TI: Multi-level Shuttle Scheduling for Large-Scale Entanglement Module Linked Trapped-Ion](http://arxiv.org/pdf/2509.25988v1)

Authors: Xian Wu, Chenghong Zhu, Jingbo Wang, Xin Wang

Trapped-ion computing is a leading architecture in the pursuit of scalable
and high fidelity quantum systems. Modular quantum architectures based on
photonic interconnects offer a promising path for scaling trapped ion devices.
In this design, multiple Quantum Charge Coupled Device (QCCD) units are
interconnected through entanglement module. Each unit features a multi-zone
layout that separates functionalities into distinct areas, enabling more
efficient and flexible quantum operations. However, achieving efficient and
scalable compilation of quantum circuits in such entanglement module linked
Quantum Charge-Coupled Device (EML-QCCD) remains a primary challenge for
practical quantum applications.
  In this work, we propose a scalable compiler tailored for large-scale
trapped-ion architectures, with the goal of reducing the shuttling overhead
inherent in EML-QCCD devices. MUSS-TI introduces a multi-level scheduling
approach inspired by multi-level memory scheduling in classical computing. This
method is designed to be aware of the distinct roles of different zones and to
minimize the number of shuttling operations required in EML-QCCD systems. We
demonstrate that EML-QCCD architectures are well-suited for executing
large-scale applications. Our evaluation shows that MUSS-TI reduces shuttle
operations by 41.74% for applications with 30-32 qubits, and by an average of
73.38% and 59.82% for applications with 117-128 qubits and 256-299 qubits,
respectively.

### 5. [Enabling Time-Aware Priority Traffic Management over Distributed FPGA Nodes](http://arxiv.org/pdf/2509.26043v1)

Authors: Alberto Scionti, Paolo Savio, Francesco Lubrano, Federico Stirano, Antonino Nespola, Olivier Terzo, Corrado De Sio, Luca Sterpone

Network Interface Cards (NICs) greatly evolved from simple basic devices
moving traffic in and out of the network to complex heterogeneous systems
offloading host CPUs from performing complex tasks on in-transit packets. These
latter comprise different types of devices, ranging from NICs accelerating
fixed specific functions (e.g., on-the-fly data compression/decompression,
checksum computation, data encryption, etc.) to complex Systems-on-Chip (SoC)
equipped with both general purpose processors and specialized engines
(Smart-NICs). Similarly, Field Programmable Gate Arrays (FPGAs) moved from pure
reprogrammable devices to modern heterogeneous systems comprising
general-purpose processors, real-time cores and even AI-oriented engines.
Furthermore, the availability of high-speed network interfaces (e.g., SFPs)
makes modern FPGAs a good choice for implementing Smart-NICs. In this work, we
extended the functionalities offered by an open-source NIC implementation
(Corundum) by enabling time-aware traffic management in hardware, and using
this feature to control the bandwidth associated with different traffic
classes. By exposing dedicated control registers on the AXI bus, the driver of
the NIC can easily configure the transmission bandwidth of different
prioritized queues. Basically, each control register is associated with a
specific transmission queue (Corundum can expose up to thousands of
transmission and receiving queues), and sets up the fraction of time in a
transmission window which the queue is supposed to get access the output port
and transmit the packets. Queues are then prioritized and associated to
different traffic classes through the Linux QDISC mechanism. Experimental
evaluation demonstrates that the approach allows to properly manage the
bandwidth reserved to the different transmission flows.

### 6. [Benchmarking Deep Learning Convolutions on Energy-constrained CPUs](http://arxiv.org/pdf/2509.26217v1)

Authors: Enrique Galvez, Adrien Cassagne, Alix Munier, Manuel Bouyer

This work evaluates state-of-the-art convolution algorithms for CPU-based
deep learning inference. While most prior studies focus on GPUs or NPUs, CPU
implementations remain relatively underoptimized. We benchmark direct,
GEMM-based, and Winograd convolutions across modern CPUs from ARM __ , Intel __
, AMD __ , Apple __ , and Nvidia __ , considering both latency and energy
efficiency. Our results highlight the key architectural factors that govern CPU
efficiency for convolution operations, providing practical guidance for
energy-aware embedded deployment. As a main results of this work, the Nvidia __
AGX Orin combined with the GEMM algorithm achieves the best trade-off between
inference latency and energy consumption.

### 7. [Stab-QRAM: An All-Clifford Quantum Random Access Memory for Special Data](http://arxiv.org/pdf/2509.26494v1)

Authors: Guangyi Li, Yu Gan, Zeguan Wu, Xueyue Zhang, Zheshen Zhang, Junyu Liu

Quantum random access memories (QRAMs) are pivotal for data-intensive quantum
algorithms, but existing general-purpose and domain-specific architectures are
hampered by a critical bottleneck: a heavy reliance on non-Clifford gates
(e.g., T-gates), which are prohibitively expensive to implement
fault-tolerantly. To address this challenge, we introduce the Stabilizer-QRAM
(Stab-QRAM), a domain-specific architecture tailored for data with an affine
Boolean structure ($f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ over
$\mathbb{F}_2$), a class of functions vital for optimization, time-series
analysis, and quantum linear systems algorithms. We demonstrate that the gate
interactions required to implement the matrix $A$ form a bipartite graph. By
applying K\"{o}nig's edge-coloring theorem to this graph, we prove that
Stab-QRAM achieves an optimal logical circuit depth of $O(\log N)$ for $N$ data
items, matching its $O(\log N)$ space complexity. Critically, the Stab-QRAM is
constructed exclusively from Clifford gates (CNOT and X), resulting in a zero
$T$-count. This design completely circumvents the non-Clifford bottleneck,
eliminating the need for costly magic state distillation and making it
exceptionally suited for early fault-tolerant quantum computing platforms. We
highlight Stab-QRAM's utility as a resource-efficient oracle for applications
in discrete dynamical systems, and as a core component in Quantum Linear
Systems Algorithms, providing a practical pathway for executing data-intensive
tasks on emerging quantum hardware.

### 8. [CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search](http://arxiv.org/pdf/2509.25862v1)

Authors: Olga Krestinskaya, Mohammed E. Fouda, Ahmed Eltawil, Khaled N. Salama

To maximize hardware efficiency and performance accuracy in Compute-In-Memory
(CIM)-based neural network accelerators for Artificial Intelligence (AI)
applications, co-optimizing both software and hardware design parameters is
essential. Manual tuning is impractical due to the vast number of parameters
and their complex interdependencies. To effectively automate the design and
optimization of CIM-based neural network accelerators, hardware-aware neural
architecture search (HW-NAS) techniques can be applied. This work introduces
CIMNAS, a joint model-quantization-hardware optimization framework for CIM
architectures. CIMNAS simultaneously searches across software parameters,
quantization policies, and a broad range of hardware parameters, incorporating
device-, circuit-, and architecture-level co-optimizations. CIMNAS experiments
were conducted over a search space of 9.9x10^85 potential parameter
combinations with the MobileNet model as a baseline and RRAM-based CIM
architecture. Evaluated on the ImageNet dataset, CIMNAS achieved a reduction in
energy-delay-area product (EDAP) ranging from 90.1x to 104.5x, an improvement
in TOPS/W between 4.68x and 4.82x, and an enhancement in TOPS/mm^2 from 11.3x
to 12.78x relative to various baselines, all while maintaining an accuracy of
73.81%. The adaptability and robustness of CIMNAS are demonstrated by extending
the framework to support the SRAM-based ResNet50 architecture, achieving up to
an 819.5x reduction in EDAP. Unlike other state-of-the-art methods, CIMNAS
achieves EDAP-focused optimization without any accuracy loss, generating
diverse software-hardware parameter combinations for high-performance CIM-based
neural network designs. The source code of CIMNAS is available at
https://github.com/OlgaKrestinskaya/CIMNAS.

### 9. [TrackCore-F: Deploying Transformer-Based Subatomic Particle Tracking on FPGAs](http://arxiv.org/pdf/2509.26335v1)

Authors: Arjan Blankestijn, Uraz Odyurt, Amirreza Yousefzadeh

The Transformer Machine Learning (ML) architecture has been gaining
considerable momentum in recent years. In particular, computational High-Energy
Physics tasks such as jet tagging and particle track reconstruction (tracking),
have either achieved proper solutions, or reached considerable milestones using
Transformers. On the other hand, the use of specialised hardware accelerators,
especially FPGAs, is an effective method to achieve online, or pseudo-online
latencies. The development and integration of Transformer-based ML to FPGAs is
still ongoing and the support from current tools is very limited to
non-existent. Additionally, FPGA resources present a significant constraint.
Considering the model size alone, while smaller models can be deployed
directly, larger models are to be partitioned in a meaningful and ideally,
automated way. We aim to develop methodologies and tools for monolithic, or
partitioned Transformer synthesis, specifically targeting inference. Our
primary use-case involves two machine learning model designs for tracking,
derived from the TrackFormers project. We elaborate our development approach,
present preliminary results, and provide comparisons.

### 10. [Rearchitecting Datacenter Lifecycle for AI: A TCO-Driven Framework](http://arxiv.org/pdf/2509.26534v1)

Authors: Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Ricardo Bianchini

The rapid rise of large language models (LLMs) has been driving an enormous
demand for AI inference infrastructure, mainly powered by high-end GPUs. While
these accelerators offer immense computational power, they incur high capital
and operational costs due to frequent upgrades, dense power consumption, and
cooling demands, making total cost of ownership (TCO) for AI datacenters a
critical concern for cloud providers. Unfortunately, traditional datacenter
lifecycle management (designed for general-purpose workloads) struggles to keep
pace with AI's fast-evolving models, rising resource needs, and diverse
hardware profiles. In this paper, we rethink the AI datacenter lifecycle scheme
across three stages: building, hardware refresh, and operation. We show how
design choices in power, cooling, and networking provisioning impact long-term
TCO. We also explore refresh strategies aligned with hardware trends. Finally,
we use operation software optimizations to reduce cost. While these
optimizations at each stage yield benefits, unlocking the full potential
requires rethinking the entire lifecycle. Thus, we present a holistic lifecycle
management framework that coordinates and co-optimizes decisions across all
three stages, accounting for workload dynamics, hardware evolution, and system
aging. Our system reduces the TCO by up to 40\% over traditional approaches.
Using our framework we provide guidelines on how to manage AI datacenter
lifecycle for the future.

### Computational Complexity

### 1. [On Boolean PCSPs with Polynomial Threshold Polymorphisms](http://arxiv.org/pdf/2509.26248v1)

Authors: Katzper Michno

In pursuit of a deeper understanding of Boolean Promise Constraint
Satisfaction Problems (PCSPs), we identify a class of problems with restricted
structural complexity, which could serve as a promising candidate for complete
characterization. Specifically, we investigate the class of PCSPs whose
polymorphisms are Polynomial Threshold Functions (PTFs) of bounded degree. We
obtain two complexity characterization results: (1) with a hardness condition
introduced in [ACMTCT'21], we establish a complete complexity dichotomy in the
case where coefficients of PTF representations are non-negative; (2) dropping
the non-negativity assumption, we show a hardness result for PTFs admitting
coordinates with significant influence, conditioned on the Rich 2-to-1
Conjecture proposed in [ITCS'21]. In order to prove the latter, we show that a
random 2-to-1 minor map retains significant coordinate influence over the
$p$-biased hypercube with constant probability.

### 2. [Unitary synthesis with fewer T gates](http://arxiv.org/pdf/2509.25702v1)

Authors: Xinyu Tan

We present a simple algorithm that implements an arbitrary $n$-qubit unitary
operator using a Clifford+T circuit with T-count $O(2^{4n/3} n^{2/3})$. This
improves upon the previous best known upper bound of $O(2^{3n/2} n)$, while the
best known lower bound remains $\Omega(2^n)$. Our construction is based on a
recursive application of the cosine-sine decomposition, together with a
generalization of the optimal diagonal unitary synthesis method by Gosset,
Kothari, and Wu to multi-controlled $k$-qubit unitaries.

### 3. [Physically-Motivated Guiding States for Local Hamiltonians](http://arxiv.org/pdf/2509.25815v1)

Authors: Gabriel Waite, Karl Lin, Samuel J Elman, Michael J Bremner

This work characterises families of guiding states for the Guided Local
Hamiltonian problem, revealing new connections between physical constraints and
computational complexity. Focusing on states motivated by Quantum Chemistry and
Hamiltonian Complexity, we extend prior BQP-hardness results beyond
semi-classical subset states. We demonstrate that broader state families
preserve hardness, while maintaining classical tractability under practical
parameter regimes. Crucially, we provide a constructive proof of BQP
containment for the canonical problem, showing the problem is BQP-complete when
provided with a polynomial-size classical description of the guiding state. Our
results show quantum advantage persists for physically meaningful state
classes, and classical methods remain viable when guiding states admit
appropriate descriptions. We identify a Goldilocks zone of guiding states that
are efficiently preparable, succinctly described, and sample-query accessible,
allowing for a meaningful comparison between quantum and classical approaches.
Our work furthers the complexity landscape for ground state estimation
problems, presenting steps toward experimentally relevant settings while
clarifying the boundaries of quantum advantage.

### 4. [On the Complexity of the Succinct State Local Hamiltonian Problem](http://arxiv.org/pdf/2509.25821v1)

Authors: Gabriel Waite, Karl Lin

We study the computational complexity of the Local Hamiltonian problem under
the promise that its ground state is succinctly represented. We show that the
Succinct State 3-Local Hamiltonian problem is (promise) MA-complete. Our proof
proceeds by systematically characterising succinct quantum states and modifying
the original MA-hardness reduction. In particular, we show that a broader class
of succinct states suffices to capture the hardness of the problem, extending
and strengthening prior results to classes of Hamiltonians with lower locality.

### 5. [The Guided Local Hamiltonian Problem for Stoquastic Hamiltonians](http://arxiv.org/pdf/2509.25829v1)

Authors: Gabriel Waite

We show that the Guided Local Hamiltonian problem for stoquastic Hamiltonians
is (promise) BPP-hard. The Guided Local Hamiltonian problem is a variant of the
Local Hamiltonian problem that incorporates an additional input known as a
guiding state, which is promised to overlap with the ground state. For a range
of local Hamiltonian families, this problem is (promise) BQP-hard, though for
stoquastic Hamiltonians, the complexity was previously unknown. Our results are
achieved by first reducing from quantum-inspired BPP circuits to 6-local
stoquastic Hamiltonians. We prove particular classes of quantum states, known
as semi-classical encoded subset states, can guide the estimation of the ground
state energy. Subsequent analysis shows the BPP-hardness is not dependent on
the locality, i.e., the result holds for 2-local stoquastic Hamiltonians.
Additional arguments further the BPP-hardness to Hamiltonians restricted to a
square lattice. We also find for stoquastic Hamiltonians with a fixed local
constraint on a subset of the system qubits, the Guided Local Hamiltonian
problem is BQP-hard.

### 6. [Strong random unitaries and fast scrambling](http://arxiv.org/pdf/2509.26310v1)

Authors: Thomas Schuster, Fermi Ma, Alex Lombardi, Fernando Brandao, Hsin-Yuan Huang

Understanding how fast physical systems can resemble Haar-random unitaries is
a fundamental question in physics. Many experiments of interest in quantum
gravity and many-body physics, including the butterfly effect in quantum
information scrambling and the Hayden-Preskill thought experiment, involve
queries to a random unitary $U$ alongside its inverse $U^\dagger$, conjugate
$U^*$, and transpose $U^T$. However, conventional notions of approximate
unitary designs and pseudorandom unitaries (PRUs) fail to capture these
experiments. In this work, we introduce and construct strong unitary designs
and strong PRUs that remain robust under all such queries. Our constructions
achieve the optimal circuit depth of $O(\log n)$ for systems of $n$ qubits. We
further show that strong unitary designs can form in circuit depth $O(\log^2
n)$ in circuits composed of independent two-qubit Haar-random gates, and that
strong PRUs can form in circuit depth $\text{poly}(\log n)$ in circuits with no
ancilla qubits. Our results provide an operational proof of the fast scrambling
conjecture from black hole physics: every observable feature of the fastest
scrambling quantum systems reproduces Haar-random behavior at logarithmic
times.

### Computational Engineering

### 1. [UncertainGen: Uncertainty-Aware Representations of DNA Sequences for Metagenomic Binning](http://arxiv.org/pdf/2509.26116v1)

Authors: Abdulkadir Celikkanat, Andres R. Masegosa, Mads Albertsen, Thomas D. Nielsen

Metagenomic binning aims to cluster DNA fragments from mixed microbial
samples into their respective genomes, a critical step for downstream analyses
of microbial communities. Existing methods rely on deterministic
representations, such as k-mer profiles or embeddings from large language
models, which fail to capture the uncertainty inherent in DNA sequences arising
from inter-species DNA sharing and from fragments with highly similar
representations. We present the first probabilistic embedding approach,
UncertainGen, for metagenomic binning, representing each DNA fragment as a
probability distribution in latent space. Our approach naturally models
sequence-level uncertainty, and we provide theoretical guarantees on embedding
distinguishability. This probabilistic embedding framework expands the feasible
latent space by introducing a data-adaptive metric, which in turn enables more
flexible separation of bins/clusters. Experiments on real metagenomic datasets
demonstrate the improvements over deterministic k-mer and LLM-based embeddings
for the binning task by offering a scalable and lightweight solution for
large-scale metagenomic analysis.

### 2. [Analyzing BEV Suitability and Charging Strategies Using Italian Driving Data](http://arxiv.org/pdf/2509.26262v1)

Authors: Homa Jamalof, Luca Vassio, Danilo Giordano, Marco Mellia, Claudio De Tommasi

Battery Electric Vehicles (BEVs) are rapidly evolving from a niche
alternative to an established option for private transportation, often
replacing Internal Combustion Engine (ICE) vehicles. Despite growing interest,
significant barriers remain, including range anxiety, the inconvenience
associated with public charging stations, and higher costs. This study analyses
extensive telemetry data collected from 10,441 users using ICE vehicles in an
Italian province to assess the potential for switching to BEVs without changing
current travel behaviour. We evaluate to what extent the BEV models can fulfil
their mobility needs under different charging scenarios. To do so, we replicate
trips and parking events, simulating and monitoring the battery state of
charge. The analysis reveals the compromises between charging behaviours and
limited BEV autonomy. Assuming access to overnight charging, at least 35% of
the users could already adopt even low-capacity BEVs.

### 3. [Importance of localized dilatation and distensibility in identifying determinants of thoracic aortic aneurysm with neural operators](http://arxiv.org/pdf/2509.26576v1)

Authors: David S. Li, Somdatta Goswami, Qianying Cao, Vivek Oommen, Roland Assi, Jay D. Humphrey, George E. Karniadakis

Thoracic aortic aneurysms (TAAs) arise from diverse mechanical and
mechanobiological disruptions to the aortic wall that increase the risk of
dissection or rupture. Evidence links TAA development to dysfunctions in the
aortic mechanotransduction axis, including loss of elastic fiber integrity and
cell-matrix connections. Because distinct insults create different mechanical
vulnerabilities, there is a critical need to identify interacting factors that
drive progression. Here, we use a finite element framework to generate
synthetic TAAs from hundreds of heterogeneous insults spanning varying degrees
of elastic fiber damage and impaired mechanosensing. From these simulations, we
construct spatial maps of localized dilatation and distensibility to train
neural networks that predict the initiating combined insult. We compare several
architectures (Deep Operator Networks, UNets, and Laplace Neural Operators) and
multiple input data formats to define a standard for future subject-specific
modeling. We also quantify predictive performance when networks are trained
using only geometric data (dilatation) versus both geometric and mechanical
data (dilatation plus distensibility). Across all networks, prediction errors
are significantly higher when trained on dilatation alone, underscoring the
added value of distensibility information. Among the tested models, UNet
consistently provides the highest accuracy across all data formats. These
findings highlight the importance of acquiring full-field measurements of both
dilatation and distensibility in TAA assessment to reveal the mechanobiological
drivers of disease and support the development of personalized treatment
strategies.

### 4. [Quasi-Monte Carlo methods for uncertainty quantification of tumor growth modeled by a parametric semi-linear parabolic reaction-diffusion equation](http://arxiv.org/pdf/2509.25753v1)

Authors: Alexander D. Gilbert, Frances Y. Kuo, Dirk Nuyens, Graham Pash, Ian H. Sloan, Karen E. Willcox

We study the application of a quasi-Monte Carlo (QMC) method to a class of
semi-linear parabolic reaction-diffusion partial differential equations used to
model tumor growth. Mathematical models of tumor growth are largely
phenomenological in nature, capturing infiltration of the tumor into
surrounding healthy tissue, proliferation of the existing tumor, and patient
response to therapies, such as chemotherapy and radiotherapy. Considerable
inter-patient variability, inherent heterogeneity of the disease, sparse and
noisy data collection, and model inadequacy all contribute to significant
uncertainty in the model parameters. It is crucial that these uncertainties can
be efficiently propagated through the model to compute quantities of interest
(QoIs), which in turn may be used to inform clinical decisions. We show that
QMC methods can be successful in computing expectations of meaningful QoIs.
Well-posedness results are developed for the model and used to show a
theoretical error bound for the case of uniform random fields. The theoretical
linear error rate, which is superior to that of standard Monte Carlo, is
verified numerically. Encouraging computational results are also provided for
lognormal random fields, prompting further theoretical development.

### 5. [Better with Less: Small Proprietary Models Surpass Large Language Models in Financial Transaction Understanding](http://arxiv.org/pdf/2509.25803v1)

Authors: Wanying Ding, Savinay Narendra, Xiran Shi, Adwait Ratnaparkhi, Chengrui Yang, Nikoo Sabzevar, Ziyan Yin

Analyzing financial transactions is crucial for ensuring regulatory
compliance, detecting fraud, and supporting decisions. The complexity of
financial transaction data necessitates advanced techniques to extract
meaningful insights and ensure accurate analysis. Since Transformer-based
models have shown outstanding performance across multiple domains, this paper
seeks to explore their potential in understanding financial transactions. This
paper conducts extensive experiments to evaluate three types of Transformer
models: Encoder-Only, Decoder-Only, and Encoder-Decoder models. For each type,
we explore three options: pretrained LLMs, fine-tuned LLMs, and small
proprietary models developed from scratch. Our analysis reveals that while
LLMs, such as LLaMA3-8b, Flan-T5, and SBERT, demonstrate impressive
capabilities in various natural language processing tasks, they do not
significantly outperform small proprietary models in the specific context of
financial transaction understanding. This phenomenon is particularly evident in
terms of speed and cost efficiency. Proprietary models, tailored to the unique
requirements of transaction data, exhibit faster processing times and lower
operational costs, making them more suitable for real-time applications in the
financial sector. Our findings highlight the importance of model selection
based on domain-specific needs and underscore the potential advantages of
customized proprietary models over general-purpose LLMs in specialized
applications. Ultimately, we chose to implement a proprietary decoder-only
model to handle the complex transactions that we previously couldn't manage.
This model can help us to improve 14% transaction coverage, and save more than
\$13 million annual cost.

### 6. [Bubble, Bubble, AI's Rumble: Why Global Financial Regulatory Incident Reporting is Our Shield Against Systemic Stumbles](http://arxiv.org/pdf/2509.26150v1)

Authors: Anchal Gupta, Gleb Pappyshev, James T Kwok

"Double, double toil and trouble; Fire burn and cauldron bubble." As
Shakespeare's witches foretold chaos through cryptic prophecies, modern capital
markets grapple with systemic risks concealed by opaque AI systems. According
to IMF, the August 5, 2024, plunge in Japanese and U.S. equities can be linked
to algorithmic trading yet ab-sent from existing AI incidents database
exemplifies this transparency crisis. Current AI incident databases, reliant on
crowdsourcing or news scraping, systematically over-look capital market
anomalies, particularly in algorithmic and high-frequency trading. We address
this critical gap by proposing a regulatory-grade global database that
elegantly synthesises post-trade reporting frameworks with proven incident
documentation models from healthcare and aviation. Our framework's temporal
data omission technique masking timestamps while preserving percent-age-based
metrics enables sophisticated cross-jurisdictional analysis of emerging risks
while safeguarding confidential business information. Synthetic data validation
(modelled after real life published incidents , sentiments, data) reveals
compelling pat-terns: systemic risks transcending geographical boundaries,
market manipulation clusters distinctly identifiable via K-means algorithms,
and AI system typology exerting significantly greater influence on trading
behaviour than geographical location, This tripartite solution empowers
regulators with unprecedented cross-jurisdictional oversight, financial
institutions with seamless compliance integration, and investors with critical
visibility into previously obscured AI-driven vulnerabilities. We call for
immediate action to strengthen risk management and foster resilience in
AI-driven financial markets against the volatile "cauldron" of AI-driven
systemic risks., promoting global financial stability through enhanced
transparency and coordinated oversight.

### Computational Geometry

### 1. [Analytic Conditions for Differentiable Collision Detection in Trajectory Optimization](http://arxiv.org/pdf/2509.26459v1)

Authors: Akshay Jaitly, Devesh K. Jha, Kei Ota, Yuki Shirai

Optimization-based methods are widely used for computing fast, diverse
solutions for complex tasks such as collision-free movement or planning in the
presence of contacts. However, most of these methods require enforcing
non-penetration constraints between objects, resulting in a non-trivial and
computationally expensive problem. This makes the use of optimization-based
methods for planning and control challenging. In this paper, we present a
method to efficiently enforce non-penetration of sets while performing
optimization over their configuration, which is directly applicable to problems
like collision-aware trajectory optimization. We introduce novel differentiable
conditions with analytic expressions to achieve this. To enforce non-collision
between non-smooth bodies using these conditions, we introduce a method to
approximate polytopes as smooth semi-algebraic sets. We present several
numerical experiments to demonstrate the performance of the proposed method and
compare the performance with other baseline methods recently proposed in the
literature.

### 2. [PFDepth: Heterogeneous Pinhole-Fisheye Joint Depth Estimation via Distortion-aware Gaussian-Splatted Volumetric Fusion](http://arxiv.org/pdf/2509.26008v1)

Authors: Zhiwei Zhang, Ruikai Xu, Weijian Zhang, Zhizhong Zhang, Xin Tan, Jingyu Gong, Yuan Xie, Lizhuang Ma

In this paper, we present the first pinhole-fisheye framework for
heterogeneous multi-view depth estimation, PFDepth. Our key insight is to
exploit the complementary characteristics of pinhole and fisheye imagery
(undistorted vs. distorted, small vs. large FOV, far vs. near field) for joint
optimization. PFDepth employs a unified architecture capable of processing
arbitrary combinations of pinhole and fisheye cameras with varied intrinsics
and extrinsics. Within PFDepth, we first explicitly lift 2D features from each
heterogeneous view into a canonical 3D volumetric space. Then, a core module
termed Heterogeneous Spatial Fusion is designed to process and fuse
distortion-aware volumetric features across overlapping and non-overlapping
regions. Additionally, we subtly reformulate the conventional voxel fusion into
a novel 3D Gaussian representation, in which learnable latent Gaussian spheres
dynamically adapt to local image textures for finer 3D aggregation. Finally,
fused volume features are rendered into multi-view depth maps. Through
extensive experiments, we demonstrate that PFDepth sets a state-of-the-art
performance on KITTI-360 and RealHet datasets over current mainstream depth
networks. To the best of our knowledge, this is the first systematic study of
heterogeneous pinhole-fisheye depth estimation, offering both technical novelty
and valuable empirical insights.

### Computation and Language

### 1. [The Media Bias Detector: A Framework for Annotating and Analyzing the News at Scale](http://arxiv.org/pdf/2509.25649v1)

Authors: Samar Haider, Amir Tohidi, Jenny S. Wang, Timothy Dörr, David M. Rothschild, Chris Callison-Burch, Duncan J. Watts

Mainstream news organizations shape public perception not only directly
through the articles they publish but also through the choices they make about
which topics to cover (or ignore) and how to frame the issues they do decide to
cover. However, measuring these subtle forms of media bias at scale remains a
challenge. Here, we introduce a large, ongoing (from January 1, 2024 to
present), near real-time dataset and computational framework developed to
enable systematic study of selection and framing bias in news coverage. Our
pipeline integrates large language models (LLMs) with scalable, near-real-time
news scraping to extract structured annotations -- including political lean,
tone, topics, article type, and major events -- across hundreds of articles per
day. We quantify these dimensions of coverage at multiple levels -- the
sentence level, the article level, and the publisher level -- expanding the
ways in which researchers can analyze media bias in the modern news landscape.
In addition to a curated dataset, we also release an interactive web platform
for convenient exploration of these data. Together, these contributions
establish a reusable methodology for studying media bias at scale, providing
empirical resources for future research. Leveraging the breadth of the corpus
over time and across publishers, we also present some examples (focused on the
150,000+ articles examined in 2024) that illustrate how this novel data set can
reveal insightful patterns in news coverage and bias, supporting academic
research and real-world efforts to improve media accountability.

### 2. [QFrBLiMP: a Quebec-French Benchmark of Linguistic Minimal Pairs](http://arxiv.org/pdf/2509.25664v1)

Authors: David Beauchemin, Pier-Luc Veilleux, Richard Khoury, Johanna-Pascale Roy

In this paper, we introduce the Quebec-French Benchmark of Linguistic Minimal
Pairs (QFrBLiMP), a corpus designed to evaluate the linguistic knowledge of
LLMs on prominent grammatical phenomena in Quebec-French. QFrBLiMP consists of
1,761 minimal pairs annotated with 20 linguistic phenomena. Specifically, these
minimal pairs have been created by manually modifying sentences extracted from
an official online resource maintained by a Qu\'ebec government institution.
Each pair is annotated by twelve Quebec-French native speakers, who select the
sentence they feel is grammatical amongst the two. These annotations are used
to compare the competency of LLMs with that of humans. We evaluate different
LLMs on QFrBLiMP and MultiBLiMP-Fr by observing the rate of higher
probabilities assigned to the sentences of each minimal pair for each category.
We find that while grammatical competence scales with model size, a clear
hierarchy of difficulty emerges. All benchmarked models consistently fail on
phenomena requiring deep semantic understanding, revealing a critical
limitation and a significant gap compared to human performance on these
specific tasks.

### 3. [Mitigating Biases in Language Models via Bias Unlearning](http://arxiv.org/pdf/2509.25673v1)

Authors: Dianqing Liu, Yi Liu, Guoqing Jin, Zhendong Mao

Many studies have shown various biases targeting different demographic groups
in language models, amplifying discrimination and harming fairness. Recent
parameter modification debiasing approaches significantly degrade core
capabilities such as text coherence and task accuracy. And Prompt-based
debiasing methods, only effective for predefined trigger words, fail to address
deeply embedded stereotypical associations in model parameters. In this paper,
we propose BiasUnlearn, a novel model debiasing framework which achieves
targeted debiasing via dual-pathway unlearning mechanisms coordinating
stereotype forgetting with anti-stereotype retention, while preventing bias
polarity reversal through adversarial forget set and dynamic dataset swapping.
We conducted extensive experiments with multiple language models across various
evaluation benchmarks. The results show that BiasUnlearn outperforms existing
methods in mitigating bias in language models while retaining language modeling
capabilities. Further experiments reveal that debiasing weights are
transferable across model variants, confirming that bias representations become
entrenched during pre-training and persist through fine-tuning phases.

### 4. [Atomic Thinking of LLMs: Decoupling and Exploring Mathematical Reasoning Abilities](http://arxiv.org/pdf/2509.25725v1)

Authors: Jiayi Kuang, Haojing Huang, Yinghui Li, Xinnian Liang, Zhikun Xu, Yangning Li, Xiaoyu Tan, Chao Qu, Meishan Zhang, Ying Shen, Philip S. Yu

Large Language Models (LLMs) have demonstrated outstanding performance in
mathematical reasoning capabilities. However, we argue that current large-scale
reasoning models primarily rely on scaling up training datasets with diverse
mathematical problems and long thinking chains, which raises questions about
whether LLMs genuinely acquire mathematical concepts and reasoning principles
or merely remember the training data. In contrast, humans tend to break down
complex problems into multiple fundamental atomic capabilities. Inspired by
this, we propose a new paradigm for evaluating mathematical atomic
capabilities. Our work categorizes atomic abilities into two dimensions: (1)
field-specific abilities across four major mathematical fields, algebra,
geometry, analysis, and topology, and (2) logical abilities at different
levels, including conceptual understanding, forward multi-step reasoning with
formal math language, and counterexample-driven backward reasoning. We propose
corresponding training and evaluation datasets for each atomic capability unit,
and conduct extensive experiments about how different atomic capabilities
influence others, to explore the strategies to elicit the required specific
atomic capability. Evaluation and experimental results on advanced models show
many interesting discoveries and inspirations about the different performances
of models on various atomic capabilities and the interactions between atomic
capabilities. Our findings highlight the importance of decoupling mathematical
intelligence into atomic components, providing new insights into model
cognition and guiding the development of training strategies toward a more
efficient, transferable, and cognitively grounded paradigm of "atomic
thinking".

### 5. [CATCH: A Novel Data Synthesis Framework for High Therapy Fidelity and Memory-Driven Planning Chain of Thought in AI Counseling](http://arxiv.org/pdf/2509.25733v1)

Authors: Mingyu Chen, Jingkai Lin, Zhaojie Chu, Xiaofen Xing, Yirong Chen, Xiangmin Xu

Recently, advancements in AI counseling based on large language models have
shown significant progress. However, existing studies employ a one-time
generation approach to synthesize multi-turn dialogue samples, resulting in low
therapy fidelity and failing to capture the decision-making rationale behind
each response. In this work, we propose CATCH, a novel data synthesis framework
designed to address these challenges. Specifically, to improve therapy
fidelity, we introduce the Progressive Dialogue Synthesis strategy, which
extracts goals, resources, and solutions from a client's self-report, organizes
them into structured outlines, and then incrementally generates stage-aligned
counseling dialogues. To capture decision-making rationale behind each
response, we propose the Memory-Driven Dynamic Planning thinking pattern that
integrates memory enhancement, global planning, and strategy reasoning; a
collaborative multi-agent optimizer then leverages MDP to attach explicit
chain-of-thought to each dialogue turn. Extensive experiments and human
evaluations demonstrate that CATCH significantly enhances fidelity and logical
coherence in AI counseling.

### 6. [Assessing Algorithmic Bias in Language-Based Depression Detection: A Comparison of DNN and LLM Approaches](http://arxiv.org/pdf/2509.25795v1)

Authors: Obed Junias, Prajakta Kini, Theodora Chaspari

This paper investigates algorithmic bias in language-based models for
automated depression detection, focusing on socio-demographic disparities
related to gender and race/ethnicity. Models trained using deep neural networks
(DNN) based embeddings are compared to few-shot learning approaches with large
language models (LLMs), evaluating both performance and fairness on clinical
interview transcripts from the Distress Analysis Interview Corpus/Wizard-of-Oz
(DAIC-WOZ). To mitigate bias, fairness-aware loss functions are applied to
DNN-based models, while in-context learning with varied prompt framing and shot
counts is explored for LLMs. Results indicate that LLMs outperform DNN-based
models in depression classification, particularly for underrepresented groups
such as Hispanic participants. LLMs also exhibit reduced gender bias compared
to DNN-based embeddings, though racial disparities persist. Among
fairness-aware techniques for mitigating bias in DNN-based embeddings, the
worst-group loss, which is designed to minimize loss for the worst-performing
demographic group, achieves a better balance between performance and fairness.
In contrast, the fairness-regularized loss minimizes loss across all groups but
performs less effectively. In LLMs, guided prompting with ethical framing helps
mitigate gender bias in the 1-shot setting. However, increasing the number of
shots does not lead to further reductions in disparities. For race/ethnicity,
neither prompting strategy nor increasing $N$ in $N$-shot learning effectively
reduces disparities.

### 7. [ReTAG: Retrieval-Enhanced, Topic-Augmented Graph-Based Global Sensemaking](http://arxiv.org/pdf/2509.25814v1)

Authors: Boyoung Kim, Dosung Lee, Sumin An, Jinseong Jeong, Paul Hongsuck Seo

Recent advances in question answering have led to substantial progress in
tasks such as multi-hop reasoning. However, global sensemaking-answering
questions by synthesizing information from an entire corpus remains a
significant challenge. A prior graph-based approach to global sensemaking lacks
retrieval mechanisms, topic specificity, and incurs high inference costs. To
address these limitations, we propose ReTAG, a Retrieval-Enhanced,
Topic-Augmented Graph framework that constructs topic-specific subgraphs and
retrieves the relevant summaries for response generation. Experiments show that
ReTAG improves response quality while significantly reducing inference time
compared to the baseline. Our code is available at
https://github.com/bykimby/retag.

### 8. [ReFACT: A Benchmark for Scientific Confabulation Detection with Positional Error Annotations](http://arxiv.org/pdf/2509.25868v1)

Authors: Yindong Wang, Martin Preiß, Margarita Bugueño, Jan Vincent Hoffbauer, Abdullatif Ghajar, Tolga Buz, Gerard de Melo

Large Language Models (LLMs) frequently confabulate scientific facts,severely
undermining their trustworthiness. Addressing this challenge requires
benchmarks that go beyond binary factuality and enable fine-grained evaluation.
We introduce \textbf{ReFACT} (\textit{Reddit False And Correct Texts}), a
benchmark of 1,001 expert-annotated question--answer pairs spanning diverse
scientific domains for the detection of scientific confabulation. Each instance
includes both a scientifically correct answer and a non-factual counterpart
annotated with \textbf{precise error spans and error-types}. ReFACT enables
multi-stage evaluation: (1) confabulation detection, (2) fine-grained error
localization, and (3) correction. We benchmark 9 state-of-the-art LLMs,
revealing limited performance ($\sim$50\% accuracy). Even top models such as
GPT-4o fail to distinguish factual from confabulated scientific answers,
raising concerns about the reliability of \textit{LLM-as-judge} evaluation
paradigms. Our findings highlight the need for fine-grained, human-validated
benchmarks to detect and correct scientific confabulation in domain-specific
contexts. Dataset is released on
\href{https://github.com/ddz5431/ReFACT}{GitHub}\footnote{We provide the
dataset at: https://github.com/ddz5431/ReFACT}.

### 9. [ASR Under Noise: Exploring Robustness for Sundanese and Javanese](http://arxiv.org/pdf/2509.25878v1)

Authors: Salsabila Zahirah Pranida, Muhammad Cendekia Airlangga, Rifo Ahmad Genadi, Shady Shehata

We investigate the robustness of Whisper-based automatic speech recognition
(ASR) models for two major Indonesian regional languages: Javanese and
Sundanese. While recent work has demonstrated strong ASR performance under
clean conditions, their effectiveness in noisy environments remains unclear. To
address this, we experiment with multiple training strategies, including
synthetic noise augmentation and SpecAugment, and evaluate performance across a
range of signal-to-noise ratios (SNRs). Our results show that noise-aware
training substantially improves robustness, particularly for larger Whisper
models. A detailed error analysis further reveals language-specific challenges,
highlighting avenues for future improvements

### 10. [Mem-α: Learning Memory Construction via Reinforcement Learning](http://arxiv.org/pdf/2509.25911v1)

Authors: Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu

Large language model (LLM) agents are constrained by limited context windows,
necessitating external memory systems for long-term information understanding.
Current memory-augmented agents typically depend on pre-defined instructions
and tools for memory updates. However, language models may lack the ability to
determine which information to store, how to structure it, and when to update
it, especially as memory systems become more complex. This results in
suboptimal memory construction and information loss. To this end, we propose
Mem-alpha, a reinforcement learning framework that trains agents to effectively
manage complex memory systems through interaction and feedback. We also
construct a specialized training dataset spanning diverse multi-turn
interaction patterns paired with comprehensive evaluation questions designed to
teach effective memory management. During training, agents process sequential
information chunks, learn to extract and store relevant content, then update
the memory system. The reward signal derives from downstream question-answering
accuracy over the full interaction history, directly optimizing for memory
construction. To illustrate the effectiveness of our training framework, we
design a memory architecture comprising core, episodic, and semantic
components, equipped with multiple tools for memory operations. Empirical
evaluation demonstrates that Mem-alpha achieves significant improvements over
existing memory-augmented agent baselines. Despite being trained exclusively on
instances with a maximum length of 30k tokens, our agents exhibit remarkable
generalization to sequences exceeding 400k tokens, over 13x the training
length, highlighting the robustness of Mem-alpha.

### Cryptography and Security

### 1. [Logic Solver Guided Directed Fuzzing for Hardware Designs](http://arxiv.org/pdf/2509.26509v1)

Authors: Raghul Saravanan, Sai Manoj P D

The ever-increasing complexity of design specifications for processors and
intellectual property (IP) presents a formidable challenge for early bug
detection in the modern IC design cycle. The recent advancements in hardware
fuzzing have proven effective in detecting bugs in RTL designs of cutting-edge
processors. The modern IC design flow involves incremental updates and
modifications to the hardware designs necessitating rigorous verification and
extending the overall verification period. To accelerate this process, directed
fuzzing has emerged focusing on generating targeted stimuli for specific
regions of the design, avoiding the need for exhaustive, full-scale
verification. However, a significant limitation of these hardware fuzzers lies
in their reliance on an equivalent SW model of the hardware which fails to
capture intrinsic hardware characteristics. To circumvent the aforementioned
challenges, this work introduces TargetFuzz, an innovative and scalable
targeted hardware fuzzing mechanism. It leverages SAT-based techniques to focus
on specific regions of the hardware design while operating at its native
hardware abstraction level, ensuring a more precise and comprehensive
verification process. We evaluated this approach across a diverse range of RTL
designs for various IP cores. Our experimental results demonstrate its
capability to effectively target and fuzz a broad spectrum of sites within
these designs, showcasing its extensive coverage and precision in addressing
targeted regions. TargetFuzz demonstrates its capability to effectively scale
30x greater in terms of handling target sites, achieving 100% state coverage
and 1.5x faster in terms of site coverage, and shows 90x improvement in target
state coverage compared to Coverage-Guided Fuzzing, demonstrating its potential
to advance the state-of-the-art in directed hardware fuzzing.

### 2. [Better Privilege Separation for Agents by Restricting Data Types](http://arxiv.org/pdf/2509.25926v1)

Authors: Dennis Jacob, Emad Alghamdi, Zhanhao Hu, Basel Alomair, David Wagner

Large language models (LLMs) have become increasingly popular due to their
ability to interact with unstructured content. As such, LLMs are now a key
driver behind the automation of language processing systems, such as AI agents.
Unfortunately, these advantages have come with a vulnerability to prompt
injections, an attack where an adversary subverts the LLM's intended
functionality with an injected task. Past approaches have proposed detectors
and finetuning to provide robustness, but these techniques are vulnerable to
adaptive attacks or cannot be used with state-of-the-art models. To this end we
propose type-directed privilege separation for LLMs, a method that
systematically prevents prompt injections. We restrict the ability of an LLM to
interact with third-party data by converting untrusted content to a curated set
of data types; unlike raw strings, each data type is limited in scope and
content, eliminating the possibility for prompt injections. We evaluate our
method across several case studies and find that designs leveraging our
principles can systematically prevent prompt injection attacks while
maintaining high utility.

### 3. [Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](http://arxiv.org/pdf/2509.26032v1)

Authors: Xiaobao Wang, Ruoxiao Sun, Yujun Zhang, Bingdao Feng, Dongxiao He, Luzhi Wang, Di Jin

Graph Neural Networks (GNNs) have demonstrated strong performance across
tasks such as node classification, link prediction, and graph classification,
but remain vulnerable to backdoor attacks that implant imperceptible triggers
during training to control predictions. While node-level attacks exploit local
message passing, graph-level attacks face the harder challenge of manipulating
global representations while maintaining stealth. We identify two main sources
of anomaly in existing graph classification backdoor methods: structural
deviation from rare subgraph triggers and semantic deviation caused by label
flipping, both of which make poisoned graphs easily detectable by anomaly
detection models. To address this, we propose DPSBA, a clean-label backdoor
framework that learns in-distribution triggers via adversarial training guided
by anomaly-aware discriminators. DPSBA effectively suppresses both structural
and semantic anomalies, achieving high attack success while significantly
improving stealth. Extensive experiments on real-world datasets validate that
DPSBA achieves a superior balance between effectiveness and detectability
compared to state-of-the-art baselines.

### 4. [SoK: Systematic analysis of adversarial threats against deep learning approaches for autonomous anomaly detection systems in SDN-IoT networks](http://arxiv.org/pdf/2509.26350v1)

Authors: Tharindu Lakshan Yasarathna, Nhien-An Le-Khac

Integrating SDN and the IoT enhances network control and flexibility.
DL-based AAD systems improve security by enabling real-time threat detection in
SDN-IoT networks. However, these systems remain vulnerable to adversarial
attacks that manipulate input data or exploit model weaknesses, significantly
degrading detection accuracy. Existing research lacks a systematic analysis of
adversarial vulnerabilities specific to DL-based AAD systems in SDN-IoT
environments. This SoK study introduces a structured adversarial threat model
and a comprehensive taxonomy of attacks, categorising them into data, model,
and hybrid-level threats. Unlike previous studies, we systematically evaluate
white, black, and grey-box attack strategies across popular benchmark datasets.
Our findings reveal that adversarial attacks can reduce detection accuracy by
up to 48.4%, with Membership Inference causing the most significant drop. C&W
and DeepFool achieve high evasion success rates. However, adversarial training
enhances robustness, and its high computational overhead limits the real-time
deployment of SDN-IoT applications. We propose adaptive countermeasures,
including real-time adversarial mitigation, enhanced retraining mechanisms, and
explainable AI-driven security frameworks. By integrating structured threat
models, this study offers a more comprehensive approach to attack
categorisation, impact assessment, and defence evaluation than previous
research. Our work highlights critical vulnerabilities in existing DL-based AAD
models and provides practical recommendations for improving resilience,
interpretability, and computational efficiency. This study serves as a
foundational reference for researchers and practitioners seeking to enhance
DL-based AAD security in SDN-IoT networks, offering a systematic adversarial
threat model and conceptual defence evaluation based on prior empirical
studies.

### 5. [DeepProv: Behavioral Characterization and Repair of Neural Networks via Inference Provenance Graph Analysis](http://arxiv.org/pdf/2509.26562v1)

Authors: Firas Ben Hmida, Abderrahmen Amich, Ata Kaboudi, Birhanu Eshete

Deep neural networks (DNNs) are increasingly being deployed in high-stakes
applications, from self-driving cars to biometric authentication. However,
their unpredictable and unreliable behaviors in real-world settings require new
approaches to characterize and ensure their reliability.
  This paper introduces DeepProv, a novel and customizable system designed to
capture and characterize the runtime behavior of DNNs during inference by using
their underlying graph structure. Inspired by system audit provenance graphs,
DeepProv models the computational information flow of a DNN's inference process
through Inference Provenance Graphs (IPGs). These graphs provide a detailed
structural representation of the behavior of DNN, allowing both empirical and
structural analysis. DeepProv uses these insights to systematically repair DNNs
for specific objectives, such as improving robustness, privacy, or fairness.
  We instantiate DeepProv with adversarial robustness as the goal of model
repair and conduct extensive case studies to evaluate its effectiveness. Our
results demonstrate its effectiveness and scalability across diverse
classification tasks, attack scenarios, and model complexities. DeepProv
automatically identifies repair actions at the node and edge-level within IPGs,
significantly enhancing the robustness of the model. In particular, applying
DeepProv repair strategies to just a single layer of a DNN yields an average
55% improvement in adversarial accuracy. Moreover, DeepProv complements
existing defenses, achieving substantial gains in adversarial robustness.
Beyond robustness, we demonstrate the broader potential of DeepProv as an
adaptable system to characterize DNN behavior in other critical areas, such as
privacy auditing and fairness analysis.

### 6. [SPATA: Systematic Pattern Analysis for Detailed and Transparent Data Cards](http://arxiv.org/pdf/2509.26640v1)

Authors: João Vitorino, Eva Maia, Isabel Praça, Carlos Soares

Due to the susceptibility of Artificial Intelligence (AI) to data
perturbations and adversarial examples, it is crucial to perform a thorough
robustness evaluation before any Machine Learning (ML) model is deployed.
However, examining a model's decision boundaries and identifying potential
vulnerabilities typically requires access to the training and testing datasets,
which may pose risks to data privacy and confidentiality. To improve
transparency in organizations that handle confidential data or manage critical
infrastructure, it is essential to allow external verification and validation
of AI without the disclosure of private datasets. This paper presents
Systematic Pattern Analysis (SPATA), a deterministic method that converts any
tabular dataset to a domain-independent representation of its statistical
patterns, to provide more detailed and transparent data cards. SPATA computes
the projection of each data instance into a discrete space where they can be
analyzed and compared, without risking data leakage. These projected datasets
can be reliably used for the evaluation of how different features affect ML
model robustness and for the generation of interpretable explanations of their
behavior, contributing to more trustworthy AI.

### 7. [STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents](http://arxiv.org/pdf/2509.25624v1)

Authors: Jing-Jing Li, Jianfeng He, Chao Shang, Devang Kulshreshtha, Xun Xian, Yi Zhang, Hang Su, Sandesh Swamy, Yanjun Qi

As LLMs advance into autonomous agents with tool-use capabilities, they
introduce security challenges that extend beyond traditional content-based LLM
safety concerns. This paper introduces Sequential Tool Attack Chaining (STAC),
a novel multi-turn attack framework that exploits agent tool use. STAC chains
together tool calls that each appear harmless in isolation but, when combined,
collectively enable harmful operations that only become apparent at the final
execution step. We apply our framework to automatically generate and
systematically evaluate 483 STAC cases, featuring 1,352 sets of
user-agent-environment interactions and spanning diverse domains, tasks, agent
types, and 10 failure modes. Our evaluations show that state-of-the-art LLM
agents, including GPT-4.1, are highly vulnerable to STAC, with attack success
rates (ASR) exceeding 90% in most cases. The core design of STAC's automated
framework is a closed-loop pipeline that synthesizes executable multi-step tool
chains, validates them through in-environment execution, and reverse-engineers
stealthy multi-turn prompts that reliably induce agents to execute the verified
malicious sequence. We further perform defense analysis against STAC and find
that existing prompt-based defenses provide limited protection. To address this
gap, we propose a new reasoning-driven defense prompt that achieves far
stronger protection, cutting ASR by up to 28.8%. These results highlight a
crucial gap: defending tool-enabled agents requires reasoning over entire
action sequences and their cumulative effects, rather than evaluating isolated
prompts or responses.

### 8. [The Impact of Scaling Training Data on Adversarial Robustness](http://arxiv.org/pdf/2509.25927v1)

Authors: Marco Zimmerli, Andreas Plesner, Till Aczel, Roger Wattenhofer

Deep neural networks remain vulnerable to adversarial examples despite
advances in architectures and training paradigms. We investigate how training
data characteristics affect adversarial robustness across 36 state-of-the-art
vision models spanning supervised, self-supervised, and contrastive learning
approaches, trained on datasets from 1.2M to 22B images. Models were evaluated
under six black-box attack categories: random perturbations, two types of
geometric masks, COCO object manipulations, ImageNet-C corruptions, and
ImageNet-R style shifts. Robustness follows a logarithmic scaling law with both
data volume and model size: a tenfold increase in data reduces attack success
rate (ASR) on average by ~3.2%, whereas a tenfold increase in model size
reduces ASR on average by ~13.4%. Notably, some self-supervised models trained
on curated datasets, such as DINOv2, outperform others trained on much larger
but less curated datasets, challenging the assumption that scale alone drives
robustness. Adversarial fine-tuning of ResNet50s improves generalization across
structural variations but not across color distributions. Human evaluation
reveals persistent gaps between human and machine vision. These results show
that while scaling improves robustness, data quality, architecture, and
training objectives play a more decisive role than raw scale in achieving
broad-spectrum adversarial resilience.

### 9. [Exact Bias of Linear TRNG Correctors -- Spectral Approach](http://arxiv.org/pdf/2509.26393v1)

Authors: Maciej Skorski, Francisco-Javier Soto, Onur Günlü

Using Fourier analysis, this paper establishes exact security bounds for
linear extractors in True Random Number Generators (TRNGs). We provide the
first near-optimal total variation security characterization by interpolating
between optimal $\ell_{\infty}$ and $\ell_2$ norm results, expressed through
code weight enumerators and input bias parameters. Our bounds improve security
assessments by an order of magnitude over previous approximations. By scanning
~20,000 codes, we reveal fundamental trade-offs between compression efficiency
and cryptographic security. For instance, we show that achieving 80 bits of
security can require sacrificing more than 50\% of the code rate when
correcting 10\% input bias. Our bounds enhance security evaluation of TRNG
post-processing schemes and quantify the inherent cost of randomness extraction
in hardware implementations.

### 10. [SeedPrints: Fingerprints Can Even Tell Which Seed Your Large Language Model Was Trained From](http://arxiv.org/pdf/2509.26404v1)

Authors: Yao Tong, Haonan Wang, Siquan Li, Kenji Kawaguchi, Tianyang Hu

Fingerprinting Large Language Models (LLMs) is essential for provenance
verification and model attribution. Existing methods typically extract post-hoc
signatures based on training dynamics, data exposure, or hyperparameters --
properties that only emerge after training begins. In contrast, we propose a
stronger and more intrinsic notion of LLM fingerprinting: SeedPrints, a method
that leverages random initialization biases as persistent, seed-dependent
identifiers present even before training. We show that untrained models exhibit
reproducible token selection biases conditioned solely on their parameters at
initialization. These biases are stable and measurable throughout training,
enabling our statistical detection method to recover a model's lineage with
high confidence. Unlike prior techniques, unreliable before convergence and
vulnerable to distribution shifts, SeedPrints remains effective across all
training stages and robust under domain shifts or parameter modifications.
Experiments on LLaMA-style and Qwen-style models show that SeedPrints achieves
seed-level distinguishability and can provide birth-to-lifecycle identity
verification akin to a biometric fingerprint. Evaluations on large-scale
pretrained models and fingerprinting benchmarks further confirm its
effectiveness under practical deployment scenarios. These results suggest that
initialization itself imprints a unique and persistent identity on neural
language models, forming a true ''Galtonian'' fingerprint.

### Computer Vision and Pattern Recognition

### 1. [LMOD+: A Comprehensive Multimodal Dataset and Benchmark for Developing and Evaluating Multimodal Large Language Models in Ophthalmology](http://arxiv.org/pdf/2509.25620v1)

Authors: Zhenyue Qin, Yang Liu, Yu Yin, Jinyu Ding, Haoran Zhang, Anran Li, Dylan Campbell, Xuansheng Wu, Ke Zou, Tiarnan D. L. Keenan, Emily Y. Chew, Zhiyong Lu, Yih-Chung Tham, Ninghao Liu, Xiuzhen Zhang, Qingyu Chen

Vision-threatening eye diseases pose a major global health burden, with
timely diagnosis limited by workforce shortages and restricted access to
specialized care. While multimodal large language models (MLLMs) show promise
for medical image interpretation, advancing MLLMs for ophthalmology is hindered
by the lack of comprehensive benchmark datasets suitable for evaluating
generative models. We present a large-scale multimodal ophthalmology benchmark
comprising 32,633 instances with multi-granular annotations across 12 common
ophthalmic conditions and 5 imaging modalities. The dataset integrates imaging,
anatomical structures, demographics, and free-text annotations, supporting
anatomical structure recognition, disease screening, disease staging, and
demographic prediction for bias evaluation. This work extends our preliminary
LMOD benchmark with three major enhancements: (1) nearly 50% dataset expansion
with substantial enlargement of color fundus photography; (2) broadened task
coverage including binary disease diagnosis, multi-class diagnosis, severity
classification with international grading standards, and demographic
prediction; and (3) systematic evaluation of 24 state-of-the-art MLLMs. Our
evaluations reveal both promise and limitations. Top-performing models achieved
~58% accuracy in disease screening under zero-shot settings, and performance
remained suboptimal for challenging tasks like disease staging. We will
publicly release the dataset, curation pipeline, and leaderboard to potentially
advance ophthalmic AI applications and reduce the global burden of
vision-threatening diseases.

### 2. [Anchor-free Cross-view Object Geo-localization with Gaussian Position Encoding and Cross-view Association](http://arxiv.org/pdf/2509.25623v1)

Authors: Xingtao Ling, Chenlin Fu, Yingying Zhu

Most existing cross-view object geo-localization approaches adopt
anchor-based paradigm. Although effective, such methods are inherently
constrained by predefined anchors. To eliminate this dependency, we first
propose an anchor-free formulation for cross-view object geo-localization,
termed AFGeo. AFGeo directly predicts the four directional offsets (left,
right, top, bottom) to the ground-truth box for each pixel, thereby localizing
the object without any predefined anchors. To obtain a more robust spatial
prior, AFGeo incorporates Gaussian Position Encoding (GPE) to model the click
point in the query image, mitigating the uncertainty of object position that
challenges object localization in cross-view scenarios. In addition, AFGeo
incorporates a Cross-view Object Association Module (CVOAM) that relates the
same object and its surrounding context across viewpoints, enabling reliable
localization under large cross-view appearance gaps. By adopting an anchor-free
localization paradigm that integrates GPE and CVOAM with minimal parameter
overhead, our model is both lightweight and computationally efficient,
achieving state-of-the-art performance on benchmark datasets.

### 3. [DescribeEarth: Describe Anything for Remote Sensing Images](http://arxiv.org/pdf/2509.25654v1)

Authors: Kaiyu Li, Zixuan Jiang, Xiangyong Cao, Jiayu Wang, Yuchen Xiao, Deyu Meng, Zhi Wang

Automated textual description of remote sensing images is crucial for
unlocking their full potential in diverse applications, from environmental
monitoring to urban planning and disaster management. However, existing studies
in remote sensing image captioning primarily focus on the image level, lacking
object-level fine-grained interpretation, which prevents the full utilization
and transformation of the rich semantic and structural information contained in
remote sensing images. To address this limitation, we propose Geo-DLC, a novel
task of object-level fine-grained image captioning for remote sensing. To
support this task, we construct DE-Dataset, a large-scale dataset contains 25
categories and 261,806 annotated instances with detailed descriptions of object
attributes, relationships, and contexts. Furthermore, we introduce
DE-Benchmark, a LLM-assisted question-answering based evaluation suite designed
to systematically measure model capabilities on the Geo-DLC task. We also
present DescribeEarth, a Multi-modal Large Language Model (MLLM) architecture
explicitly designed for Geo-DLC, which integrates a scale-adaptive focal
strategy and a domain-guided fusion module leveraging remote sensing
vision-language model features to encode high-resolution details and remote
sensing category priors while maintaining global context. Our DescribeEarth
model consistently outperforms state-of-the-art general MLLMs on DE-Benchmark,
demonstrating superior factual accuracy, descriptive richness, and grammatical
soundness, particularly in capturing intrinsic object features and surrounding
environmental attributes across simple, complex, and even out-of-distribution
remote sensing scenarios. All data, code and weights are released at
https://github.com/earth-insights/DescribeEarth.

### 4. [OmniDFA: A Unified Framework for Open Set Synthesis Image Detection and Few-Shot Attribution](http://arxiv.org/pdf/2509.25682v1)

Authors: Shiyu Wu, Shuyan Li, Jing Li, Jing Liu, Yequan Wang

AI-generated image (AIGI) detection and source model attribution remain
central challenges in combating deepfake abuses, primarily due to the
structural diversity of generative models. Current detection methods are prone
to overfitting specific forgery traits, whereas source attribution offers a
robust alternative through fine-grained feature discrimination. However,
synthetic image attribution remains constrained by the scarcity of large-scale,
well-categorized synthetic datasets, limiting its practicality and
compatibility with detection systems. In this work, we propose a new paradigm
for image attribution called open-set, few-shot source identification. This
paradigm is designed to reliably identify unseen generators using only limited
samples, making it highly suitable for real-world application. To this end, we
introduce OmniDFA (Omni Detector and Few-shot Attributor), a novel framework
for AIGI that not only assesses the authenticity of images, but also determines
the synthesis origins in a few-shot manner. To facilitate this work, we
construct OmniFake, a large class-aware synthetic image dataset that curates
$1.17$ M images from $45$ distinct generative models, substantially enriching
the foundational resources for research on both AIGI detection and attribution.
Experiments demonstrate that OmniDFA exhibits excellent capability in open-set
attribution and achieves state-of-the-art generalization performance on AIGI
detection. Our dataset and code will be made available.

### 5. [AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](http://arxiv.org/pdf/2509.25699v1)

Authors: Xiping Li, Jianghong Ma

Multimodal Chain-of-Thought (CoT) has emerged as a powerful technique for
enhancing the vision-language reasoning with interleaved information. However,
existing methods often rely on simplistic heuristics for constructing
interleaved CoT, typically depending on attention maps, which our empirical
analysis reveals can be unreliable. What's more, the shortcomings of their
passive and purposeless selection strategies and their arbitrary triggering
mechanisms in capturing the model's cognitive need for information are further
amplified. In this paper, we propose \textbf{AIMCoT}, an \textbf{A}ctive
\textbf{I}nformation-driven \textbf{M}ulti-modal
\textbf{C}hain-\textbf{o}f-\textbf{T}hought framework that addresses these
fundamental limitations. AIMCoT introduces three synergistic components: (1)
\textbf{Context-enhanced Attention-map Generation (CAG)}, which mitigates the
text-vision granularity imbalance, thereby producing more reliable attention
maps as a foundation. (2) \textbf{Active Visual Probing (AVP)}, which replaces
passive selection with a proactive, goal-oriented strategy grounded in
information theory to select image regions that help answer the questions
maximally. (3) \textbf{Dynamic Attention-shifting Trigger (DAT)}, which
intelligently determines the optimal moments to insert visual information by
monitoring the model's text-to-vision attention shifts. Extensive experiments
on three challenging benchmarks demonstrate that AIMCoT significantly
outperforms state-of-the-art methods across different settings. By actively
foraging for information and dynamically structuring its reasoning process,
AIMCoT represents a critical step towards more robust, effective, and
human-like multimodal reasoning. Our code is available at
https://anonymous.4open.science/r/AIMCoT.

### 6. [How Diffusion Models Memorize](http://arxiv.org/pdf/2509.25705v1)

Authors: Juyeop Kim, Songkuk Kim, Jong-Seok Lee

Despite their success in image generation, diffusion models can memorize
training data, raising serious privacy and copyright concerns. Although prior
work has sought to characterize, detect, and mitigate memorization, the
fundamental question of why and how it occurs remains unresolved. In this
paper, we revisit the diffusion and denoising process and analyze latent space
dynamics to address the question: "How do diffusion models memorize?" We show
that memorization is driven by the overestimation of training samples during
early denoising, which reduces diversity, collapses denoising trajectories, and
accelerates convergence toward the memorized image. Specifically: (i)
memorization cannot be explained by overfitting alone, as training loss is
larger under memorization due to classifier-free guidance amplifying
predictions and inducing overestimation; (ii) memorized prompts inject training
images into noise predictions, forcing latent trajectories to converge and
steering denoising toward their paired samples; and (iii) a decomposition of
intermediate latents reveals how initial randomness is quickly suppressed and
replaced by memorized content, with deviations from the theoretical denoising
schedule correlating almost perfectly with memorization severity. Together,
these results identify early overestimation as the central underlying mechanism
of memorization in diffusion models.

### 7. [ProbMed: A Probabilistic Framework for Medical Multimodal Binding](http://arxiv.org/pdf/2509.25711v1)

Authors: Yuan Gao, Sangwook Kim, Jianzhong You, Chris McIntosh

Medical decision-making requires integrating diverse medical information,
from imaging to clinical narratives. These medical modalities are often
acquired in a many-to-many manner. However, current medical vision-language
pretraining models (Med-VLPMs) fail to directly account for this many-to-many
mapping in their model training and embeddings. To address this, we present
Probabilistic Modality-Enhanced Diagnosis (ProbMED), a multimodal Med-VLPM that
employs probabilistic contrastive learning to model distributions over
embeddings rather than deterministic estimates. ProbMED aligns four distinct
modalities -- chest X-rays, electrocardiograms, echocardiograms, and clinical
text -- into a unified probabilistic embedding space. We use InfoNCE loss with
Hellinger distance to integrate inter-modality distributions. We introduce a
probabilistic synthetic sampling loss that captures modality-specific mean and
variance to improve intra-modality binding. Extensive experiments across 13
medical datasets demonstrate that our model outperforms current Med-VLPMs in
cross-modality retrieval, zero-shot, and few-shot classification. We also
demonstrate the robust integration of multiple modalities for prognostication,
showing improved intra- and inter-medical modality binding.

### 8. [SAGE: Spatial-visual Adaptive Graph Exploration for Visual Place Recognition](http://arxiv.org/pdf/2509.25723v1)

Authors: Shunpeng Chen, Changwei Wang, Rongtao Xu, Xingtian Pei, Yukun Song, Jinzhou Lin, Wenhao Xu, Jingyi Zhang, Li Guo, Shibiao Xu

Visual Place Recognition (VPR) requires robust retrieval of geotagged images
despite large appearance, viewpoint, and environmental variation. Prior methods
focus on descriptor fine-tuning or fixed sampling strategies yet neglect the
dynamic interplay between spatial context and visual similarity during
training. We present SAGE (Spatial-visual Adaptive Graph Exploration), a
unified training pipeline that enhances granular spatial-visual discrimination
by jointly improving local feature aggregation, organize samples during
training, and hard sample mining. We introduce a lightweight Soft Probing
module that learns residual weights from training data for patch descriptors
before bilinear aggregation, boosting distinctive local cues. During training
we reconstruct an online geo-visual graph that fuses geographic proximity and
current visual similarity so that candidate neighborhoods reflect the evolving
embedding landscape. To concentrate learning on the most informative place
neighborhoods, we seed clusters from high-affinity anchors and iteratively
expand them with a greedy weighted clique expansion sampler. Implemented with a
frozen DINOv2 backbone and parameter-efficient fine-tuning, SAGE achieves SOTA
across eight benchmarks. It attains 98.9%, 95.8%, 94.5%, and 96.0% Recall@1 on
SPED, Pitts30k-test, MSLS-val, and Nordland, respectively. Notably, our method
obtains 100% Recall@10 on SPED only using 4096D global descriptors. Code and
model will be available at: https://github.com/chenshunpeng/SAGE.

### 9. [LaTo: Landmark-tokenized Diffusion Transformer for Fine-grained Human Face Editing](http://arxiv.org/pdf/2509.25731v1)

Authors: Zhenghao Zhang, Ziying Zhang, Junchao Liao, Xiangyu Meng, Qiang Hu, Siyu Zhu, Xiaoyun Zhang, Long Qin, Weizhi Wang

Recent multimodal models for instruction-based face editing enable semantic
manipulation but still struggle with precise attribute control and identity
preservation. Structural facial representations such as landmarks are effective
for intermediate supervision, yet most existing methods treat them as rigid
geometric constraints, which can degrade identity when conditional landmarks
deviate significantly from the source (e.g., large expression or pose changes,
inaccurate landmark estimates). To address these limitations, we propose LaTo,
a landmark-tokenized diffusion transformer for fine-grained,
identity-preserving face editing. Our key innovations include: (1) a landmark
tokenizer that directly quantizes raw landmark coordinates into discrete facial
tokens, obviating the need for dense pixel-wise correspondence; (2) a
location-mapping positional encoding that integrates facial and image tokens
for unified processing, enabling flexible yet decoupled geometry-appearance
interactions with high efficiency and strong identity preservation; and (3) a
landmark predictor that leverages vision-language models to infer target
landmarks from instructions and source images, whose structured
chain-of-thought improves estimation accuracy and interactive control. To
mitigate data scarcity, we curate HFL-150K, to our knowledge the largest
benchmark for this task, containing over 150K real face pairs with fine-grained
instructions. Extensive experiments show that LaTo outperforms state-of-the-art
methods by 7.8% in identity preservation and 4.6% in semantic consistency. Code
and dataset will be made publicly available upon acceptance.

### 10. [The 1st Solution for MOSEv1 Challenge on LSVOS 2025: CGFSeg](http://arxiv.org/pdf/2509.25738v1)

Authors: Tingmin Li, Yixuan Li, Yang Yang

Video Object Segmentation (VOS) aims to track and segment specific objects
across entire video sequences, yet it remains highly challenging under complex
real-world scenarios. The MOSEv1 and LVOS dataset, adopted in the MOSEv1
challenge on LSVOS 2025, which is specifically designed to enhance the
robustness of VOS models in complex real-world scenarios, including long-term
object disappearances and reappearances, as well as the presence of small and
inconspicuous objects. In this paper, we present our improved method,
Confidence-Guided Fusion Segmentation (CGFSeg), for the VOS task in the MOSEv1
Challenge. During training, the feature extractor of SAM2 is frozen, while the
remaining components are fine-tuned to preserve strong feature extraction
ability and improve segmentation accuracy. In the inference stage, we introduce
a pixel-check strategy that progressively refines predictions by exploiting
complementary strengths of multiple models, thereby yielding robust final
masks. As a result, our method achieves a J&F score of 86.37% on the test set,
ranking 1st in the MOSEv1 Challenge at LSVOS 2025. These results highlight the
effectiveness of our approach in addressing the challenges of VOS task in
complex scenarios.

### Computers and Society

### 1. [What Drives Paper Acceptance? A Process-Centric Analysis of Modern Peer Review](http://arxiv.org/pdf/2509.25701v1)

Authors: Sangkeun Jung, Goun Pyeon, Inbum Heo, Hyungjin Ahn

Peer review is the primary mechanism for evaluating scientific contributions,
yet prior studies have mostly examined paper features or external metadata in
isolation. The emergence of open platforms such as OpenReview has transformed
peer review into a transparent and interactive process, recording not only
scores and comments but also rebuttals, reviewer-author exchanges, reviewer
disagreements, and meta-reviewer decisions. This provides unprecedented
process-level data for understanding how modern peer review operates. In this
paper, we present a large-scale empirical study of ICLR 2017-2025, encompassing
over 28,000 submissions. Our analysis integrates four complementary dimensions,
including the structure and language quality of papers (e.g., section patterns,
figure/table ratios, clarity), submission strategies and external metadata
(e.g., timing, arXiv posting, author count), the dynamics of author-reviewer
interactions (e.g., rebuttal frequency, responsiveness), and the patterns of
reviewer disagreement and meta-review mediation (e.g., score variance,
confidence weighting). Our results show that factors beyond scientific novelty
significantly shape acceptance outcomes. In particular, the rebuttal stage
emerges as a decisive phase: timely, substantive, and interactive
author-reviewer communication strongly increases the likelihood of acceptance,
often outweighing initial reviewer skepticism. Alongside this, clearer writing,
balanced visual presentation, earlier submission, and effective resolution of
reviewer disagreement also correlate with higher acceptance probabilities.
Based on these findings, we propose data-driven guidelines for authors,
reviewers, and meta-reviewers to enhance transparency and fairness in peer
review. Our study demonstrates that process-centric signals are essential for
understanding and improving modern peer review.

### 2. [A systematic comparison of Large Language Models for automated assignment assessment in programming education: Exploring the importance of architecture and vendor](http://arxiv.org/pdf/2509.26483v1)

Authors: Marcin Jukiewicz

This study presents the first large-scale, side-by-side comparison of
contemporary Large Language Models (LLMs) in the automated grading of
programming assignments. Drawing on over 6,000 student submissions collected
across four years of an introductory programming course, we systematically
analysed the distribution of grades, differences in mean scores and variability
reflecting stricter or more lenient grading, and the consistency and clustering
of grading patterns across models. Eighteen publicly available models were
evaluated: Anthropic (claude-3-5-haiku, claude-opus-4-1, claude-sonnet-4);
Deepseek (deepseek-chat, deepseek-reasoner); Google (gemini-2.0-flash-lite,
gemini-2.0-flash, gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro); and
OpenAI (gpt-4.1-mini, gpt-4.1-nano, gpt-4.1, gpt-4o-mini, gpt-4o, gpt-5-mini,
gpt-5-nano, gpt-5). Statistical tests, correlation and clustering analyses
revealed clear, systematic differences between and within vendor families, with
"mini" and "nano" variants consistently underperforming their full-scale
counterparts. All models displayed high internal agreement, measured by the
intraclass correlation coefficient, with the model consensus but only moderate
agreement with human teachers' grades, indicating a persistent gap between
automated and human assessment. These findings underscore that the choice of
model for educational deployment is not neutral and should be guided by
pedagogical goals, transparent reporting of evaluation metrics, and ongoing
human oversight to ensure accuracy, fairness and relevance.

### 3. [A Framework for Studying AI Agent Behavior: Evidence from Consumer Choice Experiments](http://arxiv.org/pdf/2509.25609v1)

Authors: Manuel Cherep, Chengtian Ma, Abigail Xu, Maya Shaked, Pattie Maes, Nikhil Singh

Environments built for people are increasingly operated by a new class of
economic actors: LLM-powered software agents making decisions on our behalf.
These decisions range from our purchases to travel plans to medical treatment
selection. Current evaluations of these agents largely focus on task
competence, but we argue for a deeper assessment: how these agents choose when
faced with realistic decisions. We introduce ABxLab, a framework for
systematically probing agentic choice through controlled manipulations of
option attributes and persuasive cues. We apply this to a realistic web-based
shopping environment, where we vary prices, ratings, and psychological nudges,
all of which are factors long known to shape human choice. We find that agent
decisions shift predictably and substantially in response, revealing that
agents are strongly biased choosers even without being subject to the cognitive
constraints that shape human biases. This susceptibility reveals both risk and
opportunity: risk, because agentic consumers may inherit and amplify human
biases; opportunity, because consumer choice provides a powerful testbed for a
behavioral science of AI agents, just as it has for the study of human
behavior. We release our framework as an open benchmark for rigorous, scalable
evaluation of agent decision-making.

### 4. [Decoding the Gender Gap: Addressing Gender Stereotypes and Psychological Barriers to Empower Women in Technology](http://arxiv.org/pdf/2509.26332v1)

Authors: Zahra Fakoor Harehdasht, Raziyeh Saki

Recently, the unequal presence of women compared to men in technology has
attracted the attention of researchers and practitioners across multiple
fields. It is time to regard this problem as a global crisis that not only
limits access to talent but also reduces the diversity of perspectives that
shape technological innovation. This article examines the psychological and
social barriers that influence this gap, as well as the interventions designed
to reduce it. Using a structured review, the findings assemble evidence on the
role of early gender stereotypes in the family and school and the continuation
of this crisis in educational and career choices, through to the psychological
challenges women face in professional settings, such as feelings of
self-undervaluation, occupational anxiety, a heightened fear of technology, and
structural limitations in educational environments. Special attention is paid
to Germany, where the technology gap is particularly evident and where multiple
national programs have been implemented to address it. The present review shows
that effective solutions require more than anti-discrimination policies: they
should include educational practices, organizational reforms, mentoring, and
psychological support. The article concludes by outlining practical and
research implications and introduces the NEURON project as a pilot
interdisciplinary initiative aimed at accelerating current empowerment efforts
and developing new programs for women in technology occupations.

### 5. [RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](http://arxiv.org/pdf/2509.25897v1)

Authors: Jisu Shin, Hoyun Song, Juhyun Oh, Changgeon Ko, Eunsu Kim, Chani Jung, Alice Oh

Humans often encounter role conflicts -- social dilemmas where the
expectations of multiple roles clash and cannot be simultaneously fulfilled. As
large language models (LLMs) become increasingly influential in human
decision-making, understanding how they behave in complex social situations is
essential. While previous research has evaluated LLMs' social abilities in
contexts with predefined correct answers, role conflicts represent inherently
ambiguous social dilemmas that require contextual sensitivity: the ability to
recognize and appropriately weigh situational cues that can fundamentally alter
decision priorities. To address this gap, we introduce RoleConflictBench, a
novel benchmark designed to evaluate LLMs' contextual sensitivity in complex
social dilemmas. Our benchmark employs a three-stage pipeline to generate over
13K realistic role conflict scenarios across 65 roles, systematically varying
their associated expectations (i.e., their responsibilities and obligations)
and situational urgency levels. By analyzing model choices across 10 different
LLMs, we find that while LLMs show some capacity to respond to these contextual
cues, this sensitivity is insufficient. Instead, their decisions are
predominantly governed by a powerful, inherent bias related to social roles
rather than situational information. Our analysis quantifies these biases,
revealing a dominant preference for roles within the Family and Occupation
domains, as well as a clear prioritization of male roles and Abrahamic
religions across most evaluatee models.

### 6. [Bubble, Bubble, AI's Rumble: Why Global Financial Regulatory Incident Reporting is Our Shield Against Systemic Stumbles](http://arxiv.org/pdf/2509.26150v1)

Authors: Anchal Gupta, Gleb Pappyshev, James T Kwok

"Double, double toil and trouble; Fire burn and cauldron bubble." As
Shakespeare's witches foretold chaos through cryptic prophecies, modern capital
markets grapple with systemic risks concealed by opaque AI systems. According
to IMF, the August 5, 2024, plunge in Japanese and U.S. equities can be linked
to algorithmic trading yet ab-sent from existing AI incidents database
exemplifies this transparency crisis. Current AI incident databases, reliant on
crowdsourcing or news scraping, systematically over-look capital market
anomalies, particularly in algorithmic and high-frequency trading. We address
this critical gap by proposing a regulatory-grade global database that
elegantly synthesises post-trade reporting frameworks with proven incident
documentation models from healthcare and aviation. Our framework's temporal
data omission technique masking timestamps while preserving percent-age-based
metrics enables sophisticated cross-jurisdictional analysis of emerging risks
while safeguarding confidential business information. Synthetic data validation
(modelled after real life published incidents , sentiments, data) reveals
compelling pat-terns: systemic risks transcending geographical boundaries,
market manipulation clusters distinctly identifiable via K-means algorithms,
and AI system typology exerting significantly greater influence on trading
behaviour than geographical location, This tripartite solution empowers
regulators with unprecedented cross-jurisdictional oversight, financial
institutions with seamless compliance integration, and investors with critical
visibility into previously obscured AI-driven vulnerabilities. We call for
immediate action to strengthen risk management and foster resilience in
AI-driven financial markets against the volatile "cauldron" of AI-driven
systemic risks., promoting global financial stability through enhanced
transparency and coordinated oversight.

### Databases

### 1. [PAT: Pattern-Perceptive Transformer for Error Detection in Relational Databases](http://arxiv.org/pdf/2509.25907v1)

Authors: Jian Fu, Xixian Han, Xiaolong Wan, Wenjian Wang

Error detection in relational databases is critical for maintaining data
quality and is fundamental to tasks such as data cleaning and assessment.
Current error detection studies mostly employ the multi-detector approach to
handle heterogeneous attributes in databases, incurring high costs.
Additionally, their data preprocessing strategies fail to leverage the
variable-length characteristic of data sequences, resulting in reduced
accuracy. In this paper, we propose an attribute-wise PAttern-perceptive
Transformer (PAT) framework for error detection in relational databases. First,
PAT introduces a learned pattern module that captures attribute-specific data
distributions through learned embeddings during model training. Second, the
Quasi-Tokens Arrangement (QTA) tokenizer is designed to divide the cell
sequence based on its length and word types, and then generate the
word-adaptive data tokens, meanwhile providing compact hyperparameters to
ensure efficiency. By interleaving data tokens with the attribute-specific
pattern tokens, PAT jointly learns shared data features across different
attributes and pattern features that are distinguishable and unique in each
specified attribute. Third, PAT visualizes the attention map to interpret its
error detection mechanism. Extensive experiments show that PAT achieves
excellent F1 scores compared to state-of-the-art data error detection methods.
Moreover, PAT significantly reduces the model parameters and FLOPs when
applying the compact QTA tokenizer.

### 2. [Experiversum: an Ecosystem for Curating and Enhancing Data-Driven Experimental Science](http://arxiv.org/pdf/2509.26102v1)

Authors: Genoveva Vargas-Solar, Umberto Costa, Jérôme Darmont, Javier Espinosa-Oviedo, Carmem Hara, Sabine Loudcher, Regina Motz, Martin A. Musicante, José-Luis Zechinelli-Martini

This paper introduces Experiversum, a lakehouse-based ecosystem that supports
the curation, documentation and reproducibility of exploratory experiments.
Experiversum enables structured research through iterative data cycles, while
capturing metadata and collaborative decisions. Demonstrated through case
studies in Earth, Life and Political Sciences, Experiversum promotes
transparent workflows and multi-perspective result interpretation. Experiversum
bridges exploratory and reproducible research, encouraging accountable and
robust data-driven practices across disciplines.

### 3. [The Grammar of FAIR: A Granular Architecture of Semantic Units for FAIR Semantics, Inspired by Biology and Linguistics](http://arxiv.org/pdf/2509.26434v1)

Authors: Lars Vogt, Barend Mons

The FAIR Principles aim to make data and knowledge Findable, Accessible,
Interoperable, and Reusable, yet current digital infrastructures often lack a
unifying semantic framework that bridges human cognition and
machine-actionability. In this paper, we introduce the Grammar of FAIR: a
granular and modular architecture for FAIR semantics built on the concept of
semantic units. Semantic units, comprising atomic statement units and composite
compound units, implement the principle of semantic modularisation, decomposing
data and knowledge into independently identifiable, semantically meaningful,
and machine-actionable units. A central metaphor guiding our approach is the
analogy between the hierarchy of level of organisation in biological systems
and the hierarchy of levels of organisation in information systems: both are
structured by granular building blocks that mediate across multiple
perspectives while preserving functional unity. Drawing further inspiration
from concept formation and natural language grammar, we show how these building
blocks map to FAIR Digitial Objects (FDOs), enabling format-agnostic semantic
transitivity from natural language token models to schema-based
representations. This dual biological-linguistic analogy provides a
semantics-first foundation for evolving cross-ecosystem infrastructures, paving
the way for the Internet of FAIR Data and Services (IFDS) and a future of
modular, AI-ready, and citation-granular scholarly communication.

### 4. [SING-SQL: A Synthetic Data Generation Framework for In-Domain Text-to-SQL Translation](http://arxiv.org/pdf/2509.25672v1)

Authors: Hasan Alp Caferoğlu, Mehmet Serhat Çelik, Özgür Ulusoy

Translating natural language questions into SQL has become a core challenge
in enabling non-technical users to query databases. While recent work has
explored large-scale synthetic data generation to improve model performance
through post-training, most efforts emphasize cross-domain generalization. This
leaves a gap for real-world enterprise scenarios, where models need to
specialize to a single database schema and organizations require to be able to
evaluate their Text-to-SQL systems on their own databases. To address this, we
introduce SING-SQL, a fully automated two-stage framework for generating
high-quality, high-coverage synthetic Text-to-SQL data for any target database,
without relying on SQL logs or manual annotations. Our approach hierarchically
partitions a database schema into sub-schemas, synthesizes SQL queries across
multiple complexity levels, and applies a quality-aware pipeline that includes
LLM-as-a-judge validation, executability checks, automatic repair, and column
balancing. We further release SingSQL-LM, a family of compact language models
fine-tuned on the synthetic data, achieving strong in-domain generalization. On
the subset of the BIRD benchmark, SingSQL-LM-3B-R64 reaches 82.87% Soft F1 and
73.03% EX upper bound with 32 candidates, outperforming the best 3B-scale
baseline by +16.21 in Soft F1 and +12.36 in EX. At the 1.5B scale,
SingSQL-LM-1.5B-R64 improves over prior systems by +9.30 in Soft F1 and +4.49
in EX. On synthetic evaluation sets, SingSQL-LMs exceed prior systems by wide
margins, establishing state-of-the-art performance among open models at
comparable scales. Our study of context management strategies reveals that
schema-free fine-tuning combined with schema-only inference provides the most
robust results. These findings establish SING-SQL as a scalable,
database-agnostic paradigm for producing and evaluating enterprise-grade
Text-to-SQL systems.

### 5. [RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search](http://arxiv.org/pdf/2509.25839v1)

Authors: Han Zhang, Dongfang Zhao

While high-dimensional embedding vectors are being increasingly employed in
various tasks like Retrieval-Augmented Generation and Recommendation Systems,
popular dimensionality reduction (DR) methods such as PCA and UMAP have rarely
been adopted for accelerating the retrieval process due to their inability of
preserving the nearest neighbor (NN) relationship among vectors. Empowered by
neural networks' optimization capability and the bounding effect of Rayleigh
quotient, we propose a Regularized Auto-Encoder (RAE) for k-NN preserving
dimensionality reduction. RAE constrains the network parameter variation
through regularization terms, adjusting singular values to control embedding
magnitude changes during reduction, thus preserving k-NN relationships. We
provide a rigorous mathematical analysis demonstrating that regularization
establishes an upper bound on the norm distortion rate of transformed vectors,
thereby offering provable guarantees for k-NN preservation. With modest
training overhead, RAE achieves superior k-NN recall compared to existing DR
approaches while maintaining fast retrieval efficiency.

### Distributed, Parallel, and Cluster Computing

### 1. [LAPIS: A Performance Portable, High Productivity Compiler Framework](http://arxiv.org/pdf/2509.25605v1)

Authors: Brian Kelley, Sivasankaran Rajamanickam

Portability, performance, and productivity are three critical dimensions for
evaluating a programming model or compiler infrastructure. Several modern
programming models for computational science focus on performance and
portability. On the other end, several machine learning focused programming
models focus on portability and productivity. A clear solution that is strong
in all three dimensions has yet to emerge. A second related problem arises when
use cases from computational science converge with machine learning. The
disparate popular frameworks of these fields require programmers to manually
integrate codes written in different frameworks. Finally, several programming
frameworks lack easy options for extensibility as any new computer architecture
change require complex changes to the programming models. We present LAPIS, an
MLIR-based compiler that addresses all three of these challenges. We
demonstrate that LAPIS can automatically lower sparse and dense linear algebra
kernels from computational science and artificial intelligence use cases. We
also show how LAPIS facilitates the integration of codes between PyTorch and
Kokkos. We compare kernel performance with the default MLIR implementations on
diverse architectures to demonstrate portability. By developing a dialect that
is built on the principles of the Kokkos ecosystem, LAPIS also allows
extensibility of the framework to new architectures.

### 2. [Parallax: Efficient LLM Inference Service over Decentralized Environment](http://arxiv.org/pdf/2509.26182v1)

Authors: Chris Tong, Youhe Jiang, Gufeng Chen, Tianyi Zhao, Sibian Lu, Wenjie Qu, Eric Yang, Lynn Ai, Binhang Yuan

Deploying a large language model (LLM) inference service remains costly
because centralized serving depends on specialized GPU clusters and
high-bandwidth interconnects in datacenters. An appealing alternative is to
leverage collaborative decentralized GPU pools. However, heterogeneity in GPU
and limited interconnected network bandwidth, along with potentially dynamic
availability, make efficient scheduling the central challenge in this scenario.
In this paper, we present Parallax, a decentralized LLM serving system that
turns a pool of heterogeneous GPUs into an efficient inference platform via a
two-phase scheduler. Parallax decomposes planning into (i) model allocation,
which places layers of each replica across diverse GPUs to jointly optimize
latency and throughput under memory and link-bandwidth constraints, and (ii)
request-time GPU pipeline selection, which stitches layers from different
replicas into end-to-end execution chains that balance load and adapt to
current conditions. We implement Parallax and evaluate it on open-source LLMs
deployed over real volunteer nodes. Parallax consistently reduces latency and
increases throughput relative to decentralized baselines, demonstrating that
principled scheduling can make volunteer compute a practical, affordable
substrate for LLM inference.
  Github Repo at: https://github.com/GradientHQ/parallax.

### 3. [I Like To Move It -- Computation Instead of Data in the Brain](http://arxiv.org/pdf/2509.26193v1)

Authors: Fabian Czappa, Marvin Kaster, Felix Wolf

The detailed functioning of the human brain is still poorly understood. Brain
simulations are a well-established way to complement experimental research, but
must contend with the computational demands of the approximately $10^{11}$
neurons and the $10^{14}$ synapses connecting them, the network of the latter
referred to as the connectome. Studies suggest that changes in the connectome
(i.e., the formation and deletion of synapses, also known as structural
plasticity) are essential for critical tasks such as memory formation and
learning. The connectivity update can be efficiently computed using a
Barnes-Hut-inspired approximation that lowers the computational complexity from
$O(n^2)$ to $O(n log n)$, where n is the number of neurons. However, updating
synapses, which relies heavily on RMA, and the spike exchange between neurons,
which requires all-to-all communication at every time step, still hinder
scalability. We present a new algorithm that significantly reduces the
communication overhead by moving computation instead of data. This shrinks the
time it takes to update connectivity by a factor of six and the time it takes
to exchange spikes by more than two orders of magnitude.

### 4. [Accelerating LLM Inference with Precomputed Query Storage](http://arxiv.org/pdf/2509.25919v1)

Authors: Jay H. Park, Youngju Cho, Choungsol Lee, Moonwook Oh, Euiseong Seo

Large language model (LLM) inference often suffers from high latency,
particularly in resource-constrained environments such as on-device or edge
deployments. To address this challenge, we present StorInfer, a novel
storage-assisted LLM inference system that accelerates response time by
precomputing and storing predictable query-response pairs offline. When a user
query semantically matches a precomputed query, StorInfer bypasses expensive
GPU inference and instantly returns the stored response, significantly reducing
latency and compute costs. To maximize coverage and effectiveness, StorInfer
employs an LLM-driven generator that adaptively produces diverse and
deduplicated queries based on a given knowledge base. This is achieved via two
techniques: adaptive query masking, which prevents regeneration of similar
queries, and adaptive sampling, which dynamically tunes generation parameters
to promote semantic diversity. The resulting query-response pairs are embedded
and indexed using a disk-backed vector database to enable fast,
similarity-based retrieval at runtime. Using this approach, we generated 150K
unique precomputed pairs (taking up to 830 MB of storage space), achieving up
to 17.3% latency reduction with no loss in response quality. Our evaluation
across multiple QA datasets demonstrates the practicality and scalability of
storage-assisted inference, especially in scenarios with predictable query
distributions. StorInfer highlights a promising direction in leveraging storage
as a primary enabler for efficient, low-latency LLM deployment.

### 5. [Enabling Time-Aware Priority Traffic Management over Distributed FPGA Nodes](http://arxiv.org/pdf/2509.26043v1)

Authors: Alberto Scionti, Paolo Savio, Francesco Lubrano, Federico Stirano, Antonino Nespola, Olivier Terzo, Corrado De Sio, Luca Sterpone

Network Interface Cards (NICs) greatly evolved from simple basic devices
moving traffic in and out of the network to complex heterogeneous systems
offloading host CPUs from performing complex tasks on in-transit packets. These
latter comprise different types of devices, ranging from NICs accelerating
fixed specific functions (e.g., on-the-fly data compression/decompression,
checksum computation, data encryption, etc.) to complex Systems-on-Chip (SoC)
equipped with both general purpose processors and specialized engines
(Smart-NICs). Similarly, Field Programmable Gate Arrays (FPGAs) moved from pure
reprogrammable devices to modern heterogeneous systems comprising
general-purpose processors, real-time cores and even AI-oriented engines.
Furthermore, the availability of high-speed network interfaces (e.g., SFPs)
makes modern FPGAs a good choice for implementing Smart-NICs. In this work, we
extended the functionalities offered by an open-source NIC implementation
(Corundum) by enabling time-aware traffic management in hardware, and using
this feature to control the bandwidth associated with different traffic
classes. By exposing dedicated control registers on the AXI bus, the driver of
the NIC can easily configure the transmission bandwidth of different
prioritized queues. Basically, each control register is associated with a
specific transmission queue (Corundum can expose up to thousands of
transmission and receiving queues), and sets up the fraction of time in a
transmission window which the queue is supposed to get access the output port
and transmit the packets. Queues are then prioritized and associated to
different traffic classes through the Linux QDISC mechanism. Experimental
evaluation demonstrates that the approach allows to properly manage the
bandwidth reserved to the different transmission flows.

### 6. [Efficient Distributed Training via Dual Batch Sizes and Cyclic Progressive Learning](http://arxiv.org/pdf/2509.26092v1)

Authors: Kuan-Wei Lu, Ding-Yong Hong, Pangfeng Liu, Jan-Jan Wu

Distributed machine learning is critical for training deep learning models on
large datasets and with numerous parameters. Current research primarily focuses
on leveraging additional hardware resources and powerful computing units to
accelerate the training process. As a result, larger batch sizes are often
employed to speed up training. However, training with large batch sizes can
lead to lower accuracy due to poor generalization. To address this issue, we
propose the dual batch size learning scheme, a distributed training method
built on the parameter server framework. This approach maximizes training
efficiency by utilizing the largest batch size that the hardware can support
while incorporating a smaller batch size to enhance model generalization. By
using two different batch sizes simultaneously, this method reduces testing
loss and enhances generalization, with minimal extra training time.
Additionally, to mitigate the time overhead caused by dual batch size learning,
we propose the cyclic progressive learning scheme. This technique gradually
adjusts image resolution from low to high during training, significantly
boosting training speed. By combining cyclic progressive learning with dual
batch size learning, our hybrid approach improves both model generalization and
training efficiency. Experimental results using ResNet-18 show that, compared
to conventional training methods, our method can improve accuracy by 3.3% while
reducing training time by 10.6% on CIFAR-100, and improve accuracy by 0.1%
while reducing training time by 35.7% on ImageNet.

### 7. [Efficient Construction of Large Search Spaces for Auto-Tuning](http://arxiv.org/pdf/2509.26253v1)

Authors: Floris-Jan Willemsen, Rob V. van Nieuwpoort, Ben van Werkhoven

Automatic performance tuning, or auto-tuning, accelerates high-performance
codes by exploring vast spaces of code variants. However, due to the large
number of possible combinations and complex constraints, constructing these
search spaces can be a major bottleneck. Real-world applications have been
encountered where the search space construction takes minutes to hours or even
days. Current state-of-the-art techniques for search space construction, such
as chain-of-trees, lack a formal foundation and only perform adequately on a
specific subset of search spaces.
  We show that search space construction for constraint-based auto-tuning can
be reformulated as a Constraint Satisfaction Problem (CSP). Building on this
insight with a CSP solver, we develop a runtime parser that translates
user-defined constraint functions into solver-optimal expressions, optimize the
solver to exploit common structures in auto-tuning constraints, and integrate
these and other advances in open-source tools. These contributions
substantially improve performance and accessibility while preserving
flexibility.
  We evaluate our approach using a diverse set of benchmarks, demonstrating
that our optimized solver reduces construction time by four orders of magnitude
versus brute-force enumeration, three orders of magnitude versus an unoptimized
CSP solver, and one to two orders of magnitude versus leading auto-tuning
frameworks built on chain-of-trees. We thus eliminate a critical scalability
barrier for auto-tuning and provide a drop-in solution that enables the
exploration of previously unattainable problem scales in auto-tuning and
related domains.

### 8. [CSnake: Detecting Self-Sustaining Cascading Failure via Causal Stitching of Fault Propagations](http://arxiv.org/pdf/2509.26529v1)

Authors: Shangshu Qian, Lin Tan, Yongle Zhang

Recent studies have revealed that self-sustaining cascading failures in
distributed systems frequently lead to widespread outages, which are
challenging to contain and recover from. Existing failure detection techniques
struggle to expose such failures prior to deployment, as they typically require
a complex combination of specific conditions to be triggered. This challenge
stems from the inherent nature of cascading failures, as they typically involve
a sequence of fault propagations, each activated by distinct conditions.
  This paper presents CSnake, a fault injection framework to expose
self-sustaining cascading failures in distributed systems. CSnake uses the
novel idea of causal stitching, which causally links multiple single-fault
injections in different tests to simulate complex fault propagation chains. To
identify these chains, CSnake designs a counterfactual causality analysis of
fault propagations - fault causality analysis (FCA): FCA compares the execution
trace of a fault injection run with its corresponding profile run (i.e., same
test w/o the injection) and identifies any additional faults triggered, which
are considered to have a causal relationship with the injected fault.
  To address the large search space of fault and workload combinations, CSnake
employs a three-phase allocation protocol of test budget that prioritizes
faults with unique and diverse causal consequences, increasing the likelihood
of uncovering conditional fault propagations. Furthermore, to avoid incorrectly
connecting fault propagations from workloads with incompatible conditions,
CSnake performs a local compatibility check that approximately checks the
compatibility of the path constraints associated with connected fault
propagations with low overhead.
  CSnake detected 15 bugs that cause self-sustaining cascading failures in five
systems, five of which have been confirmed with two fixed.

### 9. [TASP: Topology-aware Sequence Parallelism](http://arxiv.org/pdf/2509.26541v1)

Authors: Yida Wang, Ke Hong, Xiuhong Li, Yuanchao Xu, Wenxun Wang, Guohao Dai, Yu Wang

Long-context large language models (LLMs) face constraints due to the
quadratic complexity of the self-attention mechanism. The mainstream sequence
parallelism (SP) method, Ring Attention, attempts to solve this by distributing
the query into multiple query chunks across accelerators and enable each Q
tensor to access all KV tensors from other accelerators via the Ring AllGather
communication primitive. However, it exhibits low communication efficiency,
restricting its practical applicability. This inefficiency stems from the
mismatch between the Ring AllGather communication primitive it adopts and the
AlltoAll topology of modern accelerators. A Ring AllGather primitive is
composed of iterations of ring-styled data transfer, which can only utilize a
very limited fraction of an AlltoAll topology.
  Inspired by the Hamiltonian decomposition of complete directed graphs, we
identify that modern accelerator topology can be decomposed into multiple
orthogonal ring datapaths which can concurrently transfer data without
interference. Based on this, we further observe that the Ring AllGather
primitive can also be decomposed into the same number of concurrent ring-styled
data transfer at every iteration. Based on these insights, we propose TASP, a
topology-aware SP method for long-context LLMs that fully utilizes the
communication capacity of modern accelerators via topology decomposition and
primitive decomposition. Experimental results on both single-node and
multi-node NVIDIA H100 systems and a single-node AMD MI300X system demonstrate
that TASP achieves higher communication efficiency than Ring Attention on these
modern accelerator topologies and achieves up to 3.58 speedup than Ring
Attention and its variant Zigzag-Ring Attention. The code is available at
https://github.com/infinigence/HamiltonAttention.

### 10. [PAST: Pilot and Adaptive Orchestration for Timely and Resilient Service Delivery in Edge-Assisted UAV Networks under Spatio-Temporal Dynamics](http://arxiv.org/pdf/2509.25700v1)

Authors: Houyi Qi, Minghui Liwang, Liqun Fu, Sai Zou, Xinlei Yi, Wei Ni, Huaiyu Dai

Incentive-driven resource trading is essential for UAV applications with
intensive, time-sensitive computing demands. Traditional spot trading suffers
from negotiation delays and high energy costs, while conventional futures
trading struggles to adapt to the dynamic, uncertain UAV-edge environment. To
address these challenges, we propose PAST (pilot-and-adaptive stable trading),
a novel framework for edge-assisted UAV networks with spatio-temporal dynamism.
PAST integrates two complementary mechanisms: PilotAO (pilot trading agreements
with overbooking), a risk-aware, overbooking-enabled early-stage
decision-making module that establishes long-term, mutually beneficial
agreements and boosts resource utilization; and AdaptAO (adaptive trading
agreements with overbooking rate update), an intelligent adaptation module that
dynamically updates agreements and overbooking rates based on UAV mobility,
supply-demand variations, and agreement performance. Together, these mechanisms
enable both stability and flexibility, guaranteeing individual rationality,
strong stability, competitive equilibrium, and weak Pareto optimality.
Extensive experiments on real-world datasets show that PAST consistently
outperforms benchmark methods in decision-making overhead, task completion
latency, resource utilization, and social welfare. By combining predictive
planning with real-time adjustments, PAST offers a valuable reference on robust
and adaptive practice for improving low-altitude mission performance.

### Digital Libraries

### 1. [First Workshop on Building Innovative Research Systems for Digital Libraries (BIRDS 2025)](http://arxiv.org/pdf/2509.26001v1)

Authors: Christin Katharina Kreutz, Hermann Kroll

We propose the first workshop on Building Innovative Research Systems for
Digital Libraries (BIRDS) to take place at TPDL 2025 as a full-day workshop.
BIRDS addresses practitioners working in digital libraries and GLAMs as well as
researchers from computational domains such as data science, information
retrieval, natural language processing, and data modelling. Our
interdisciplinary workshop focuses on connecting members of both worlds. One of
today's biggest challenges is the increasing information flood. Large language
models like ChatGPT seem to offer good performance for answering questions on
the web. So, shall we just build upon that idea and use chatbots in digital
libraries? Or do we need to design and develop specialized and effective access
paths? Answering these questions requires to connect different communities,
practitioners from real digital libraries and researchers in the area of
computer science. In brief, our workshop's goal is thus to support researchers
and practitioners to build the next generation of innovative and effective
digital library systems.

### Discrete Mathematics

### 1. [Well-Quasi-Ordering Eulerian Digraphs Embeddable in Surfaces by Strong Immersion](http://arxiv.org/pdf/2509.26260v1)

Authors: Dario Cavallaro, Ken-ichi Kawarabayashi, Stephan Kreutzer

We prove that for every surface $\Sigma$, the class of Eulerian directed
graphs that are Eulerian embeddable into $\Sigma$ (in particular they have
degree at most $4$) is well-quasi-ordered by strong immersion. This result
marks one of the most versatile directed graph classes (besides tournaments)
for which we are aware of a positive well-quasi-ordering result regarding a
well-studied graph relation.
  Our result implies that the class of bipartite circle graphs is
well-quasi-ordered under the pivot-minor relation. Furthermore, this also
yields two other interesting applications, namely, a polynomial-time algorithm
for testing immersion closed properties of Eulerian-embeddable graphs into a
fixed surface, and a characterisation of the Erd\H{o}s-P\'osa property for
Eulerian digraphs of maximum degree four.
  Further, in order to prove the mentioned result, we prove that Eulerian
digraphs of carving width bounded by some constant $k$ (which correspond to
Eulerian digraphs with bounded treewidth and additionally bounded degree) are
well-quasi-ordered by strong immersion. We actually prove a stronger result
where we allow for vertices of the Eulerian digraphs to be labeled by elements
of some well-quasi-order $\Omega$. We complement these results with a proof
that the class of Eulerian planar digraphs of treewidth at most $3$ is not
well-quasi-ordered by strong immersion, noting that any antichain of bounded
treewidth cannot have bounded degree.

### 2. [Balanced Fibonacci word rectangles, and beyond](http://arxiv.org/pdf/2509.25994v1)

Authors: Jeffrey Shallit, Ingrid Vukusic

Following a recent paper of Anselmo et al., we consider $m \times n$
rectangular matrices formed from the Fibonacci word, and we show that their
balance properties can be solved with a finite automaton. We also generalize
the result to every Sturmian characteristic word corresponding to a quadratic
irrational.

### Data Structures and Algorithms

### 1. [A Polylogarithmic Competitive Algorithm for Stochastic Online Sorting and TSP](http://arxiv.org/pdf/2509.26073v1)

Authors: Andreas Kalavas, Charalampos Platanos, Thanos Tolias

In \emph{Online Sorting}, an array of $n$ initially empty cells is given. At
each time step $t$, an element $x_t \in [0,1]$ arrives and must be placed
irrevocably into an empty cell without any knowledge of future arrivals. We aim
to minimize the sum of absolute differences between pairs of elements placed in
consecutive array cells, seeking an online placement strategy that results in a
final array close to a sorted one. An interesting multidimensional
generalization, a.k.a. the \emph{Online Travelling Salesperson Problem}, arises
when the request sequence consists of points in the $d$-dimensional unit cube
and the objective is to minimize the sum of euclidean distances between points
in consecutive cells. Motivated by the recent work of (Abrahamsen, Bercea,
Beretta, Klausen and Kozma; ESA 2024), we consider the \emph{stochastic
version} of Online Sorting (\textit{resp.} Online TSP), where each element
(\textit{resp.} point) $x_t$ is an i.i.d. sample from the uniform distribution
on $[0, 1]$ (\textit{resp.} $[0,1]^d$). By carefully decomposing the request
sequence into a hierarchy of balls-into-bins instances, where the balls to bins
ratio is large enough so that bin occupancy is sharply concentrated around its
mean and small enough so that we can efficiently deal with the elements placed
in the same bin, we obtain an online algorithm that approximates the optimal
cost within a factor of $O(\log^2 n)$ with high probability. Our result
comprises an exponential improvement on the previously best known competitive
ratio of $\tilde{O}(n^{1/4})$ for Stochastic Online Sorting due to (Abrahamsen
et al.; ESA 2024) and $O(\sqrt{n})$ for (adversarial) Online TSP due to
(Bertram, ESA 2025).

### 2. [Improved Approximation for Broadcasting in k-cycle Graphs](http://arxiv.org/pdf/2509.26426v1)

Authors: Jeffrey Bringolf, Anne-Laure Ehresmann, Hovhannes A. Harutyunyan

Broadcasting is an information dissemination primitive where a message
originates at a node (called the originator) and is passed to all other nodes
in the network. Broadcasting research is motivated by efficient network design
and determining the broadcast times of standard network topologies. Verifying
the broadcast time of a node $v$ in an arbitrary network $G$ is known to be
NP-hard. Additionally, recent findings show that the broadcast time problem is
also NP-complete in general cactus graphs and some highly restricted
subfamilies of cactus graphs. These graph families are structurally similar to
$k$-cycle graphs, in which the broadcast time problem is also believed to be
NP-complete. In this paper, we present a simple $(1.5-\epsilon)$-approximation
algorithm for determining the broadcast time of networks modeled using
$k$-cycle graphs, where $\epsilon > 0$ depends on the structure of the graph.

### 3. [Efficient Approximation Algorithms for Fair Influence Maximization under Maximin Constraint](http://arxiv.org/pdf/2509.26579v1)

Authors: Xiaobin Rui, Zhixiao Wang, Chen Peng, Qiangpeng Fang, Wei Chen

Aiming to reduce disparities of influence across different groups, Fair
Influence Maximization (FIM) has recently garnered widespread attention. The
maximin constraint, a common notion of fairness adopted in the FIM problem,
imposes a direct and intuitive requirement that asks the utility (influenced
ratio within a group) of the worst-off group should be maximized. Although the
objective of FIM under maximin constraint is conceptually straightforward, the
development of efficient algorithms with strong theoretical guarantees remains
an open challenge. The difficulty arises from the fact that the maximin
objective does not satisfy submodularity, a key property for designing
approximate algorithms in traditional influence maximization settings. In this
paper, we address this challenge by proposing a two-step optimization framework
consisting of Inner-group Maximization (IGM) and Across-group Maximization
(AGM). We first prove that the influence spread within any individual group
remains submodular, enabling effective optimization within groups. Based on
this, IGM applies a greedy approach to pick high-quality seeds for each group.
In the second step, AGM coordinates seed selection across groups by introducing
two strategies: Uniform Selection (US) and Greedy Selection (GS). We prove that
AGM-GS holds a $(1 - 1/e - \varepsilon)$ approximation to the optimal solution
when groups are completely disconnected, while AGM-US guarantees a roughly
$\frac{1}{m}(1 - 1/e - \varepsilon)$ lower bound regardless of the group
structure, with $m$ denoting the number of groups

### 4. [Signal-Aware Workload Shifting Algorithms with Uncertainty-Quantified Predictors](http://arxiv.org/pdf/2509.26511v1)

Authors: Ezra Johnson, Adam Lechowicz, Mohammad Hajiesmaili

A wide range of sustainability and grid-integration strategies depend on
workload shifting, which aligns the timing of energy consumption with external
signals such as grid curtailment events, carbon intensity, or time-of-use
electricity prices. The main challenge lies in the online nature of the
problem: operators must make real-time decisions (e.g., whether to consume
energy now) without knowledge of the future. While forecasts of signal values
are typically available, prior work on learning-augmented online algorithms has
relied almost exclusively on simple point forecasts. In parallel, the
forecasting research has made significant progress in uncertainty
quantification (UQ), which provides richer and more fine-grained predictive
information. In this paper, we study how online workload shifting can leverage
UQ predictors to improve decision-making. We introduce $\texttt{UQ-Advice}$, a
learning-augmented algorithm that systematically integrates UQ forecasts
through a $\textit{decision uncertainty score}$ that measures how forecast
uncertainty affects optimal future decisions. By introducing
$\textit{UQ-robustness}$, a new metric that characterizes how performance
degrades with forecast uncertainty, we establish theoretical performance
guarantees for $\texttt{UQ-Advice}$. Finally, using trace-driven experiments on
carbon intensity and electricity price data, we demonstrate that
$\texttt{UQ-Advice}$ consistently outperforms robust baselines and existing
learning-augmented methods that ignore uncertainty.

### 5. [On Computing Top-$k$ Simple Shortest Paths from a Single Source](http://arxiv.org/pdf/2509.26094v1)

Authors: Mattia D'Emidio, Gabriele Di Stefano

We investigate the problem of computing the top-$k$ simple shortest paths in
weighted digraphs. While the single-pair variant -- finding the top-$k$ simple
shortest paths between two specified vertices -- has been extensively studied
over the past decades, with Yen's algorithm and its heuristic improvements
emerging as the most effective solving strategies, relatively little attention
has been devoted to the more general single-source version, where the goal is
determining top-$k$ simple shortest paths from a source vertex to all other
vertices. Motivated by the numerous practical applications of ranked shortest
paths, in this paper we provide new insights and algorithmic contributions to
this problem. In particular, we first present a theoretical characterization of
the structural properties of its solutions. Then, we introduce the first
polynomial-time algorithm specifically designed to handle it. On the one hand,
we prove our new algorithm is on par, in terms of time complexity, with the
best (and only) polynomial-time approach known in the literature to solve the
problem, that is applying the fastest single-pair algorithm independently to
each vertex pair formed by the source and the remaining vertices. On the other
hand, through an extensive experimental evaluation on both real-world and
synthetic graphs, we demonstrate that our algorithm consistently and
significantly outperforms the latter baseline in terms of running time,
achieving speed-ups of up to several orders of magnitude. These results
establish our new algorithm as the solution to be preferred for computing $k$
simple shortest paths from a single source in practical settings.

### Emerging Technologies

### 1. [CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search](http://arxiv.org/pdf/2509.25862v1)

Authors: Olga Krestinskaya, Mohammed E. Fouda, Ahmed Eltawil, Khaled N. Salama

To maximize hardware efficiency and performance accuracy in Compute-In-Memory
(CIM)-based neural network accelerators for Artificial Intelligence (AI)
applications, co-optimizing both software and hardware design parameters is
essential. Manual tuning is impractical due to the vast number of parameters
and their complex interdependencies. To effectively automate the design and
optimization of CIM-based neural network accelerators, hardware-aware neural
architecture search (HW-NAS) techniques can be applied. This work introduces
CIMNAS, a joint model-quantization-hardware optimization framework for CIM
architectures. CIMNAS simultaneously searches across software parameters,
quantization policies, and a broad range of hardware parameters, incorporating
device-, circuit-, and architecture-level co-optimizations. CIMNAS experiments
were conducted over a search space of 9.9x10^85 potential parameter
combinations with the MobileNet model as a baseline and RRAM-based CIM
architecture. Evaluated on the ImageNet dataset, CIMNAS achieved a reduction in
energy-delay-area product (EDAP) ranging from 90.1x to 104.5x, an improvement
in TOPS/W between 4.68x and 4.82x, and an enhancement in TOPS/mm^2 from 11.3x
to 12.78x relative to various baselines, all while maintaining an accuracy of
73.81%. The adaptability and robustness of CIMNAS are demonstrated by extending
the framework to support the SRAM-based ResNet50 architecture, achieving up to
an 819.5x reduction in EDAP. Unlike other state-of-the-art methods, CIMNAS
achieves EDAP-focused optimization without any accuracy loss, generating
diverse software-hardware parameter combinations for high-performance CIM-based
neural network designs. The source code of CIMNAS is available at
https://github.com/OlgaKrestinskaya/CIMNAS.

### 2. [Robust NbN on Si-SiGe hybrid superconducting-semiconducting microwave quantum circuit](http://arxiv.org/pdf/2509.26363v1)

Authors: Paniz Foshat, Samane Kalhor, Shima Poorgholam-khanjari, Douglas Paul, Martin Weides, Kaveh Delfanazari

Advancing large-scale quantum computing requires superconducting circuits
that combine long coherence times with compatibility with semiconductor
technology. We investigate niobium nitride (NbN) coplanar waveguide resonators
integrated with Si/SiGe quantum wells, creating a hybrid platform designed for
CMOS-compatible quantum hardware. Using temperature-dependent microwave
spectroscopy in the single-photon regime, we examine resonance frequency and
quality factor variations to probe the underlying loss mechanisms. Our analysis
identifies the roles of two-level systems, quasiparticles, and scattering
processes, and connects these losses to wafer properties and fabrication
methods. The devices demonstrate reproducible performance and stable operation
maintained for over two years, highlighting their robustness. These results
provide design guidelines for developing low-loss, CMOS-compatible
superconducting circuits and support progress toward resilient, scalable
architectures for quantum information processing.

### Formal Languages and Automata Theory

### 1. [Group Actions and Some Combinatorics on Words with $\mathbf{vtm}$](http://arxiv.org/pdf/2509.26613v1)

Authors: John Machacek

We introduce generalizations of powers and factor complexity via orbits of
group actions. These generalizations include concepts like abelian powers and
abelian complexity. It is shown that this notion of factor complexity cannot be
used to recognize Sturmian words in general. Within our framework, we establish
square avoidance results for the ternary squarefree Thue--Morse word
$\mathbf{vtm}$. These results go beyond the usual squarefreeness of
$\mathbf{vtm}$ and are proved using Walnut. Lastly, we establish a group action
factor complexity formula for $\mathbf{vtm}$ that is expressed in terms of the
abelian complexity of the period doubling word $\mathbf{pd}$.

### 2. [Balanced Fibonacci word rectangles, and beyond](http://arxiv.org/pdf/2509.25994v1)

Authors: Jeffrey Shallit, Ingrid Vukusic

Following a recent paper of Anselmo et al., we consider $m \times n$
rectangular matrices formed from the Fibonacci word, and we show that their
balance properties can be solved with a finite automaton. We also generalize
the result to every Sturmian characteristic word corresponding to a quadratic
irrational.

### 3. [Black-box Context-free Grammar Inference for Readable & Natural Grammars](http://arxiv.org/pdf/2509.26616v1)

Authors: Mohammad Rifat Arefin, Shanto Rahman, Christoph Csallner

Black-box context-free grammar inference is crucial for program analysis,
reverse engineering, and security, yet existing tools such as Arvada, TreeVada,
and Kedavra struggle with scalability, readability, and accuracy on large,
complex languages. We present NatGI, a novel LLM-guided grammar inference
framework that extends TreeVada's parse tree recovery with three key
innovations: bracket-guided bubble exploration, LLM-driven bubble generation
and non-terminal labeling, and hierarchical delta debugging (HDD) for
systematic tree simplification. Bracket-guided exploration leverages syntactic
cues such as parentheses to propose well-structured grammar fragments, while
LLM guidance produces meaningful non-terminal names and selects more promising
merges. Finally, HDD incrementally reduces unnecessary rules, which makes the
grammars both compact and interpretable. In our experiments, we evaluate NatGI
on a comprehensive benchmark suite ranging from small languages to larger ones
such as lua, c, and mysql. Our results show that NatGI consistently outperforms
strong baselines in terms of F1 score. On average, NatGI achieves an F1 score
of 0.57, which is 25pp (percentage points) higher than the best-performing
baseline, TreeVada. In the case of interpretability, our generated grammars
perform significantly better than those produced by existing approaches.
Leveraging LLM-based node renaming and bubble exploration, NatGI produces rules
with meaningful non-terminal names and compact structures that align more
closely with human intuition. As a result, developers and researchers can
achieve higher accuracy while still being able to easily inspect, verify, and
reason about the structure and semantics of the induced grammars.

### Graphics

### 1. [Palace: A Library for Interactive GPU-Accelerated Large Tensor Processing and Visualization](http://arxiv.org/pdf/2509.26213v1)

Authors: Dominik Drees, Benjamin Risse

Tensor datasets (two-, three-, or higher-dimensional) are fundamental to many
scientific fields utilizing imaging or simulation technologies. Advances in
these methods have led to ever-increasing data sizes and, consequently,
interest and development of out-of-core processing and visualization
techniques, although mostly as specialized solutions. Here we present Palace,
an open-source, cross-platform, general-purpose library for interactive and
accelerated out-of-core tensor processing and visualization. Through a
high-performance asynchronous concurrent architecture and a simple
compute-graph interface, Palace enables the interactive development of
out-of-core pipelines on workstation hardware. We demonstrate on benchmarks
that Palace outperforms or matches state-of-the-art systems for volume
rendering and hierarchical random-walker segmentation and demonstrate
applicability in use cases involving tensors from 2D images up to 4D time
series datasets.

### 2. [Vector sketch animation generation with differentialable motion trajectories](http://arxiv.org/pdf/2509.25857v1)

Authors: Xinding Zhu, Xinye Yang, Shuyang Zheng, Zhexin Zhang, Fei Gao, Jing Huang, Jiazhou Chen

Sketching is a direct and inexpensive means of visual expression. Though
image-based sketching has been well studied, video-based sketch animation
generation is still very challenging due to the temporal coherence requirement.
In this paper, we propose a novel end-to-end automatic generation approach for
vector sketch animation. To solve the flickering issue, we introduce a
Differentiable Motion Trajectory (DMT) representation that describes the
frame-wise movement of stroke control points using differentiable
polynomial-based trajectories. DMT enables global semantic gradient propagation
across multiple frames, significantly improving the semantic consistency and
temporal coherence, and producing high-framerate output. DMT employs a
Bernstein basis to balance the sensitivity of polynomial parameters, thus
achieving more stable optimization. Instead of implicit fields, we introduce
sparse track points for explicit spatial modeling, which improves efficiency
and supports long-duration video processing. Evaluations on DAVIS and LVOS
datasets demonstrate the superiority of our approach over SOTA methods.
Cross-domain validation on 3D models and text-to-video data confirms the
robustness and compatibility of our approach.

### 3. [GaussEdit: Adaptive 3D Scene Editing with Text and Image Prompts](http://arxiv.org/pdf/2509.26055v1)

Authors: Zhenyu Shu, Junlong Yu, Kai Chao, Shiqing Xin, Ligang Liu

This paper presents GaussEdit, a framework for adaptive 3D scene editing
guided by text and image prompts. GaussEdit leverages 3D Gaussian Splatting as
its backbone for scene representation, enabling convenient Region of Interest
selection and efficient editing through a three-stage process. The first stage
involves initializing the 3D Gaussians to ensure high-quality edits. The second
stage employs an Adaptive Global-Local Optimization strategy to balance global
scene coherence and detailed local edits and a category-guided regularization
technique to alleviate the Janus problem. The final stage enhances the texture
of the edited objects using a sophisticated image-to-image synthesis technique,
ensuring that the results are visually realistic and align closely with the
given prompts. Our experimental results demonstrate that GaussEdit surpasses
existing methods in editing accuracy, visual fidelity, and processing speed. By
successfully embedding user-specified concepts into 3D scenes, GaussEdit is a
powerful tool for detailed and user-driven 3D scene editing, offering
significant improvements over traditional methods.

### 4. [3DiFACE: Synthesizing and Editing Holistic 3D Facial Animation](http://arxiv.org/pdf/2509.26233v1)

Authors: Balamurugan Thambiraja, Malte Prinzler, Sadegh Aliakbarian, Darren Cosker, Justus Thies

Creating personalized 3D animations with precise control and realistic head
motions remains challenging for current speech-driven 3D facial animation
methods. Editing these animations is especially complex and time consuming,
requires precise control and typically handled by highly skilled animators.
Most existing works focus on controlling style or emotion of the synthesized
animation and cannot edit/regenerate parts of an input animation. They also
overlook the fact that multiple plausible lip and head movements can match the
same audio input. To address these challenges, we present 3DiFACE, a novel
method for holistic speech-driven 3D facial animation. Our approach produces
diverse plausible lip and head motions for a single audio input and allows for
editing via keyframing and interpolation. Specifically, we propose a
fully-convolutional diffusion model that can leverage the viseme-level
diversity in our training corpus. Additionally, we employ a speaking-style
personalization and a novel sparsely-guided motion diffusion to enable precise
control and editing. Through quantitative and qualitative evaluations, we
demonstrate that our method is capable of generating and editing diverse
holistic 3D facial animations given a single audio input, with control between
high fidelity and diversity. Code and models are available here:
https://balamuruganthambiraja.github.io/3DiFACE

### Computer Science and Game Theory

### 1. [Achieving Pareto Optimality in Games via Single-bit Feedback](http://arxiv.org/pdf/2509.25921v1)

Authors: Seref Taha Kiremitci, Ahmed Said Donmez, Muhammed O. Sayin

Efficient coordination in multi-agent systems often incurs high communication
overhead or slow convergence rates, making scalable welfare optimization
difficult. We propose Single-Bit Coordination Dynamics for Pareto-Efficient
Outcomes (SBC-PE), a decentralized learning algorithm requiring only a
single-bit satisfaction signal per agent each round. Despite this extreme
efficiency, SBC-PE guarantees convergence to the exact optimal solution in
arbitrary finite games. We establish explicit regret bounds, showing expected
regret grows only logarithmically with the horizon, i.e., O(log T). Compared
with prior payoff-based or bandit-style rules, SBC-PE uniquely combines minimal
signaling, general applicability, and finite-time guarantees. These results
show scalable welfare optimization is achievable under minimal communication
constraints.

### 2. [Quadratic Programming Approach for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games](http://arxiv.org/pdf/2509.25618v1)

Authors: Sam Ganzfried

There has been significant recent progress in algorithms for approximation of
Nash equilibrium in large two-player zero-sum imperfect-information games and
exact computation of Nash equilibrium in multiplayer strategic-form games.
While counterfactual regret minimization and fictitious play are scalable to
large games and have convergence guarantees in two-player zero-sum games, they
do not guarantee convergence to Nash equilibrium in multiplayer games. We
present an approach for exact computation of Nash equilibrium in multiplayer
imperfect-information games that solves a quadratically-constrained program
based on a nonlinear complementarity problem formulation from the sequence-form
game representation. This approach capitalizes on recent advances for solving
nonconvex quadratic programs. Our algorithm is able to quickly solve
three-player Kuhn poker after removal of dominated actions. Of the available
algorithms in the Gambit software suite, only the logit quantal response
approach is successfully able to solve the game; however, the approach takes
longer than our algorithm and also involves a degree of approximation. Our
formulation also leads to a new approach for computing Nash equilibrium in
multiplayer strategic-form games which we demonstrate to outperform a previous
quadratically-constrained program formulation.

### 3. [PAST: Pilot and Adaptive Orchestration for Timely and Resilient Service Delivery in Edge-Assisted UAV Networks under Spatio-Temporal Dynamics](http://arxiv.org/pdf/2509.25700v1)

Authors: Houyi Qi, Minghui Liwang, Liqun Fu, Sai Zou, Xinlei Yi, Wei Ni, Huaiyu Dai

Incentive-driven resource trading is essential for UAV applications with
intensive, time-sensitive computing demands. Traditional spot trading suffers
from negotiation delays and high energy costs, while conventional futures
trading struggles to adapt to the dynamic, uncertain UAV-edge environment. To
address these challenges, we propose PAST (pilot-and-adaptive stable trading),
a novel framework for edge-assisted UAV networks with spatio-temporal dynamism.
PAST integrates two complementary mechanisms: PilotAO (pilot trading agreements
with overbooking), a risk-aware, overbooking-enabled early-stage
decision-making module that establishes long-term, mutually beneficial
agreements and boosts resource utilization; and AdaptAO (adaptive trading
agreements with overbooking rate update), an intelligent adaptation module that
dynamically updates agreements and overbooking rates based on UAV mobility,
supply-demand variations, and agreement performance. Together, these mechanisms
enable both stability and flexibility, guaranteeing individual rationality,
strong stability, competitive equilibrium, and weak Pareto optimality.
Extensive experiments on real-world datasets show that PAST consistently
outperforms benchmark methods in decision-making overhead, task completion
latency, resource utilization, and social welfare. By combining predictive
planning with real-time adjustments, PAST offers a valuable reference on robust
and adaptive practice for improving low-altitude mission performance.

### Human-Computer Interaction

### 1. [Photographic Conviviality: A Synchronic and Symbiotic Photographic Experience through a Body Paint Workshop](http://arxiv.org/pdf/2509.25968v1)

Authors: Chinatsu Ozawa, Tatsuya Minagawa, Yoichi Ochiai

This study explores "Photo Tattooing," merging photography and body
ornamentation, and introduces the concept of "Photographic Conviviality." Using
our instant camera that prints images onto mesh screens for immediate body art,
we examine how this integration affects personal expression and challenges
traditional photography. Workshops revealed that this fusion redefines
photography's role, fostering intimacy and shared experiences, and opens new
avenues for self-expression by transforming static images into dynamic,
corporeal experiences.

### 2. [Dia-Lingle: A Gamified Interface for Dialectal Data Collection](http://arxiv.org/pdf/2509.26210v1)

Authors: Jiugeng Sun, Rita Sevastjanova, Sina Ahmadi, Rico Sennrich, Mennatallah El-Assady

Dialects suffer from the scarcity of computational textual resources as they
exist predominantly in spoken rather than written form and exhibit remarkable
geographical diversity. Collecting dialect data and subsequently integrating it
into current language technologies present significant obstacles. Gamification
has been proven to facilitate remote data collection processes with great ease
and on a substantially wider scale. This paper introduces Dia-Lingle, a
gamified interface aimed to improve and facilitate dialectal data collection
tasks such as corpus expansion and dialect labelling. The platform features two
key components: the first challenges users to rewrite sentences in their
dialects, identifies them through a classifier and solicits feedback, and the
other one asks users to match sentences to their geographical locations.
Dia-Lingle combines active learning with gamified difficulty levels,
strategically encouraging prolonged user engagement while efficiently enriching
the dialect corpus. Usability evaluation shows that our interface demonstrates
high levels of user satisfaction. We provide the link to Dia-Lingle:
https://dia-lingle.ivia.ch/, and demo video: https://youtu.be/0QyJsB8ym64.

### 3. [From Code to Concept: Evaluating Multiple Coordinated Views in Introductory Programming](http://arxiv.org/pdf/2509.26466v1)

Authors: Naaz Sibia, Valeria Ramirez Osorio, Jessica Wen, Rutwa Engineer, Angela Zavaleta Bernuy, Andrew Petersen, Michael Liut, Carolina Nobre

Novice programmers often struggle to understand how code executes and to form
the abstract mental models necessary for effective problem-solving, challenges
that are amplified in large, diverse introductory courses where students'
backgrounds, language proficiencies, and prior experiences vary widely. This
study examines whether interactive, multi-representational visualizations,
combining synchronized code views, memory diagrams, and conceptual analogies,
can help manage cognitive load and foster engagement more effectively than
single-visual or text-only approaches. Over a 12-week deployment in a
high-enrolment introductory Python course (N = 829), students who relied solely
on text-based explanations reported significantly higher immediate mental
effort than those using visual aids, although overall cognitive load did not
differ significantly among conditions. The multi-representational approach
consistently yielded higher engagement than both single-visual and text-only
methods. Usage logs indicated that learners' interaction patterns varied with
topic complexity, and predictive modelling suggested that early experiences of
high cognitive load were associated with lower longer-term perceptions of
clarity and helpfulness. Individual differences, including language proficiency
and prior programming experience, moderated these patterns. By integrating
multiple external representations with scaffolded support adapted to diverse
learner profiles, our findings highlight design considerations for creating
visualization tools that more effectively support novices learning to program.

### 4. [The Invisible Mentor: Inferring User Actions from Screen Recordings to Recommend Better Workflows](http://arxiv.org/pdf/2509.26557v1)

Authors: Litao Yan, Andrew Head, Ken Milne, Vu Le, Sumit Gulwani, Chris Parnin, Emerson Murphy-Hill

Many users struggle to notice when a more efficient workflow exists in
feature-rich tools like Excel. Existing AI assistants offer help only after
users describe their goals or problems, which can be effortful and imprecise.
We present InvisibleMentor, a system that turns screen recordings of task
completion into vision-grounded reflections on tasks. It detects issues such as
repetitive edits and recommends more efficient alternatives based on observed
behavior. Unlike prior systems that rely on logs, APIs, or user prompts,
InvisibleMentor operates directly on screen recordings. It uses a two-stage
pipeline: a vision-language model reconstructs actions and context, and a
language model generates structured, high-fidelity suggestions. In evaluation,
InvisibleMentor accurately identified inefficient workflows, and participants
found its suggestions more actionable, tailored, and more helpful for learning
and improvement compared to a prompt-based spreadsheet assistant.

### 5. [Exploring Large Language Model as an Interactive Sports Coach: Lessons from a Single-Subject Half Marathon Preparation](http://arxiv.org/pdf/2509.26593v1)

Authors: Kichang Lee

Large language models (LLMs) are emerging as everyday assistants, but their
role as longitudinal virtual coaches is underexplored. This two-month single
subject case study documents LLM guided half marathon preparation
(July-September 2025). Using text based interactions and consumer app logs, the
LLM acted as planner, explainer, and occasional motivator. Performance improved
from sustaining 2 km at 7min 54sec per km to completing 21.1 km at 6min 30sec
per km, with gains in cadence, pace HR coupling, and efficiency index trends.
While causal attribution is limited without a control, outcomes demonstrate
safe, measurable progress. At the same time, gaps were evident, no realtime
sensor integration, text only feedback, motivation support that was user
initiated, and limited personalization or safety guardrails. We propose design
requirements for next generation systems, persistent athlete models with
explicit guardrails, multimodal on device sensing, audio, haptic, visual
feedback, proactive motivation scaffolds, and privacy-preserving
personalization. This study offers grounded evidence and a design agenda for
evolving LLMs from retrospective advisors to closed-loop coaching companions.

### 6. [Supporting Creative Ownership through Deep Learning-Based Music Variation](http://arxiv.org/pdf/2509.25834v1)

Authors: Stephen James Krol, Maria Teresa Llano, Jon McCormack

This paper investigates the importance of personal ownership in musical AI
design, examining how practising musicians can maintain creative control over
the compositional process. Through a four-week ecological evaluation, we
examined how a music variation tool, reliant on the skill of musicians,
functioned within a composition setting. Our findings demonstrate that the
dependence of the tool on the musician's ability, to provide a strong initial
musical input and to turn moments into complete musical ideas, promoted
ownership of both the process and artefact. Qualitative interviews further
revealed the importance of this personal ownership, highlighting tensions
between technological capability and artistic identity. These findings provide
insight into how musical AI can support rather than replace human creativity,
highlighting the importance of designing tools that preserve the humanness of
musical expression.

### 7. [Believing without Seeing: Quality Scores for Contextualizing Vision-Language Model Explanations](http://arxiv.org/pdf/2509.25844v1)

Authors: Keyu He, Tejas Srinivasan, Brihi Joshi, Xiang Ren, Jesse Thomason, Swabha Swayamdipta

When people query Vision-Language Models (VLMs) but cannot see the
accompanying visual context (e.g. for blind and low-vision users), augmenting
VLM predictions with natural language explanations can signal which model
predictions are reliable. However, prior work has found that explanations can
easily convince users that inaccurate VLM predictions are correct. To remedy
undesirable overreliance on VLM predictions, we propose evaluating two
complementary qualities of VLM-generated explanations via two quality scoring
functions. We propose Visual Fidelity, which captures how faithful an
explanation is to the visual context, and Contrastiveness, which captures how
well the explanation identifies visual details that distinguish the model's
prediction from plausible alternatives. On the A-OKVQA and VizWiz tasks, these
quality scoring functions are better calibrated with model correctness than
existing explanation qualities. We conduct a user study in which participants
have to decide whether a VLM prediction is accurate without viewing its visual
context. We observe that showing our quality scores alongside VLM explanations
improves participants' accuracy at predicting VLM correctness by 11.1%,
including a 15.4% reduction in the rate of falsely believing incorrect
predictions. These findings highlight the utility of explanation quality scores
in fostering appropriate reliance on VLM predictions.

### 8. [NeuroTTT: Bridging Pretraining-Downstream Task Misalignment in EEG Foundation Models via Test-Time Training](http://arxiv.org/pdf/2509.26301v1)

Authors: Suli Wang, Yangshen Deng, Zhenghua Bao, Xinyu Zhan, Yiqun Duan

Large-scale foundation models for EEG signals offer a promising path to
generalizable brain-computer interface (BCI) applications, but they often
suffer from misalignment between pretraining objectives and downstream tasks,
as well as significant cross-subject distribution shifts. This paper addresses
these challenges by introducing a two-stage alignment strategy that bridges the
gap between generic pretraining and specific EEG decoding tasks. First, we
propose NeuroTTT: a domain-specific self-supervised fine-tuning paradigm that
augments the foundation model with task-relevant self-supervised objectives,
aligning latent representations to important spectral, spatial, and temporal
EEG features without requiring additional labeled data. Second, we incorporate
test-time training (TTT) at inference, we perform (i) self-supervised test-time
training on individual unlabeled test samples and (ii) prediction entropy
minimization (Tent), which updates only normalization statistics to continually
calibrate the model to each new input on the fly. Our approach, which, to our
knowledge, is the first to unify domain-tuned self-supervision with test-time
training in large-scale EEG foundation models, yields substantially improved
robustness and accuracy across diverse BCI tasks (imagined speech, stress
detection, motor imagery). Using CBraMod and LaBraM as backbones, our method
pushes their performance to a markedly higher level. Results on three diverse
tasks demonstrate that the proposed alignment strategy achieves
state-of-the-art performance, outperforming conventional fine-tuning and
adaptation methods. Our code is available at
https://github.com/wsl2000/NeuroTTT.

### 9. [Decoding the Gender Gap: Addressing Gender Stereotypes and Psychological Barriers to Empower Women in Technology](http://arxiv.org/pdf/2509.26332v1)

Authors: Zahra Fakoor Harehdasht, Raziyeh Saki

Recently, the unequal presence of women compared to men in technology has
attracted the attention of researchers and practitioners across multiple
fields. It is time to regard this problem as a global crisis that not only
limits access to talent but also reduces the diversity of perspectives that
shape technological innovation. This article examines the psychological and
social barriers that influence this gap, as well as the interventions designed
to reduce it. Using a structured review, the findings assemble evidence on the
role of early gender stereotypes in the family and school and the continuation
of this crisis in educational and career choices, through to the psychological
challenges women face in professional settings, such as feelings of
self-undervaluation, occupational anxiety, a heightened fear of technology, and
structural limitations in educational environments. Special attention is paid
to Germany, where the technology gap is particularly evident and where multiple
national programs have been implemented to address it. The present review shows
that effective solutions require more than anti-discrimination policies: they
should include educational practices, organizational reforms, mentoring, and
psychological support. The article concludes by outlining practical and
research implications and introduces the NEURON project as a pilot
interdisciplinary initiative aimed at accelerating current empowerment efforts
and developing new programs for women in technology occupations.

### 10. [EEG-based AI-BCI Wheelchair Advancement: Hybrid Deep Learning with Motor Imagery for Brain Computer Interface](http://arxiv.org/pdf/2509.25667v1)

Authors: Bipul Thapa, Biplov Paneru, Bishwash Paneru, Khem Narayan Poudyal

This paper presents an Artificial Intelligence (AI) integrated novel approach
to Brain-Computer Interface (BCI)-based wheelchair development, utilizing a
motor imagery right-left-hand movement mechanism for control. The system is
designed to simulate wheelchair navigation based on motor imagery right and
left-hand movements using electroencephalogram (EEG) data. A pre-filtered
dataset, obtained from an open-source EEG repository, was segmented into arrays
of 19x200 to capture the onset of hand movements. The data was acquired at a
sampling frequency of 200Hz. The system integrates a Tkinter-based interface
for simulating wheelchair movements, offering users a functional and intuitive
control system. We propose a BiLSTM-BiGRU model that shows a superior test
accuracy of 92.26% as compared with various machine learning baseline models,
including XGBoost, EEGNet, and a transformer-based model. The Bi-LSTM-BiGRU
attention-based model achieved a mean accuracy of 90.13% through
cross-validation, showcasing the potential of attention mechanisms in BCI
applications.

### Information Retrieval

### 1. [Fading to Grow: Growing Preference Ratios via Preference Fading Discrete Diffusion for Recommendation](http://arxiv.org/pdf/2509.26063v1)

Authors: Guoqing Hu, An Zhang. Shuchang Liu, Wenyu Mao, Jiancan Wu, Xun Yang, Xiang Li, Lantao Hu, Han Li, Kun Gai, Xiang Wang

Recommenders aim to rank items from a discrete item corpus in line with user
interests, yet suffer from extremely sparse user preference data. Recent
advances in diffusion models have inspired diffusion-based recommenders, which
alleviate sparsity by injecting noise during a forward process to prevent the
collapse of perturbed preference distributions. However, current
diffusion-based recommenders predominantly rely on continuous Gaussian noise,
which is intrinsically mismatched with the discrete nature of user preference
data in recommendation. In this paper, building upon recent advances in
discrete diffusion, we propose PreferGrow, a discrete diffusion-based
recommender system that models preference ratios by fading and growing user
preferences over the discrete item corpus. PreferGrow differs from existing
diffusion-based recommenders in three core aspects: (1) Discrete modeling of
preference ratios: PreferGrow models relative preference ratios between item
pairs, rather than operating in the item representation or raw score simplex.
This formulation aligns naturally with the discrete and ranking-oriented nature
of recommendation tasks. (2) Perturbing via preference fading: Instead of
injecting continuous noise, PreferGrow fades user preferences by replacing the
preferred item with alternatives -- physically akin to negative sampling --
thereby eliminating the need for any prior noise assumption. (3) Preference
reconstruction via growing: PreferGrow reconstructs user preferences by
iteratively growing the preference signals from the estimated ratios.
PreferGrow offers a well-defined matrix-based formulation with theoretical
guarantees on Markovianity and reversibility, and it demonstrates consistent
performance gains over state-of-the-art diffusion-based recommenders across
five benchmark datasets, highlighting both its theoretical soundness and
empirical effectiveness.

### 2. [Items Proxy Bridging: Enabling Frictionless Critiquing in Knowledge Graph Recommendations](http://arxiv.org/pdf/2509.26107v1)

Authors: Huanyu Zhang, Xiaoxuan Shen, Yu Lei, Baolin Yi, Jianfang Liu, Yinao xie

Modern recommender systems place great inclination towards facilitating user
experience, as more applications enabling users to critique and then refine
recommendations immediately. Considering the real-time requirements,
critique-able recommender systems typically straight modify the model
parameters and update the recommend list through analyzing the user critiquing
keyphrases in the inference phase. Current critiquing methods require first
constructing a specially designated model which establish direct correlations
between users and keyphrases during the training phase allowing for innovative
recommendations upon the critiquing,restricting the applicable scenarios.
Additionally, all these approaches ignore the catastrophic forgetting problem,
where the cumulative changes in parameters during continuous multi-step
critiquing may lead to a collapse in model performance. Thus, We conceptualize
a proxy bridging users and keyphrases, proposing a streamlined yet potent Items
Proxy Generic Critiquing Framework (IPGC) framework, which can serve as a
universal plugin for most knowledge graph recommender models based on
collaborative filtering (CF) strategies. IPGC provides a new paradigm for
frictionless integration of critique mechanisms to enable iterative
recommendation refinement in mainstream recommendation scenarios. IPGC
describes the items proxy mechanism for transforming the critiquing
optimization objective of user-keyphrase pairs into user-item pairs, adapting
it for general CF recommender models without the necessity of specifically
designed user-keyphrase correlation module. Furthermore, an anti-forgetting
regularizer is introduced in order to efficiently mitigate the catastrophic
forgetting problem of the model as a prior for critiquing optimization.

### 3. [Leveraging Scene Context with Dual Networks for Sequential User Behavior Modeling](http://arxiv.org/pdf/2509.26172v1)

Authors: Xu Chen, Yunmeng Shu, Yuangang Pan, Jinsong Lan, Xiaoyong Zhu, Shuai Xiao, Haojin Zhu, Ivor W. Tsang, Bo Zheng

Modeling sequential user behaviors for future behavior prediction is crucial
in improving user's information retrieval experience. Recent studies highlight
the importance of incorporating contextual information to enhance prediction
performance. One crucial but usually neglected contextual information is the
scene feature which we define as sub-interfaces within an app, created by
developers to provide specific functionalities, such as ``text2product search"
and ``live" modules in e-commence apps. Different scenes exhibit distinct
functionalities and usage habits, leading to significant distribution gap in
user engagement across them. Popular sequential behavior models either ignore
the scene feature or merely use it as attribute embeddings, which cannot
effectively capture the dynamic interests and interplay between scenes and
items when modeling user sequences. In this work, we propose a novel Dual
Sequence Prediction networks (DSPnet) to effectively capture the dynamic
interests and interplay between scenes and items for future behavior
prediction. DSPnet consists of two parallel networks dedicated to learn users'
dynamic interests over items and scenes, and a sequence feature enhancement
module to capture the interplay for enhanced future behavior prediction.
Further, we introduce a Conditional Contrastive Regularization (CCR) loss to
capture the invariance of similar historical sequences. Theoretical analysis
suggests that DSPnet is a principled way to learn the joint relationships
between scene and item sequences. Extensive experiments are conducted on one
public benchmark and two collected industrial datasets. The method has been
deployed online in our system, bringing a 0.04 point increase in CTR, 0.78\%
growth in deals, and 0.64\% rise in GMV. The codes are available at this
anonymous github:
\textcolor{blue}{https://anonymous.4open.science/r/DSPNet-ForPublish-2506/}.

### 4. [Informed Dataset Selection](http://arxiv.org/pdf/2509.26448v1)

Authors: Abdullah Abbas, Michael Heep, Theodor Sperle

The selection of datasets in recommender systems research lacks a systematic
methodology. Researchers often select datasets based on popularity rather than
empirical suitability. We developed the APS Explorer, a web application that
implements the Algorithm Performance Space (APS) framework for informed dataset
selection. The system analyzes 96 datasets using 28 algorithms across three
metrics (nDCG, Hit Ratio, Recall) at five K-values. We extend the APS framework
with a statistical based classification system that categorizes datasets into
five difficulty levels based on quintiles. We also introduce a
variance-normalized distance metric based on Mahalanobis distance to measure
similarity. The APS Explorer was successfully developed with three interactive
modules for visualizing algorithm performance, direct comparing algorithms, and
analyzing dataset metadata. This tool shifts the process of selecting datasets
from intuition-based to evidence-based practices, and it is publicly available
at datasets.recommender-systems.com.

### 5. [HiFIRec: Towards High-Frequency yet Low-Intention Behaviors for Multi-Behavior Recommendation](http://arxiv.org/pdf/2509.25755v1)

Authors: Ruiqi Luo, Ran Jin, Zhenglong Li, Kaixi Hu, Xiaohui Tao, Lin Li

Multi-behavior recommendation leverages multiple types of user-item
interactions to address data sparsity and cold-start issues, providing
personalized services in domains such as healthcare and e-commerce. Most
existing methods utilize graph neural networks to model user intention in a
unified manner, which inadequately considers the heterogeneity across different
behaviors. Especially, high-frequency yet low-intention behaviors may
implicitly contain noisy signals, and frequent patterns that are plausible
while misleading, thereby hindering the learning of user intentions. To this
end, this paper proposes a novel multi-behavior recommendation method, HiFIRec,
that corrects the effect of high-frequency yet low-intention behaviors by
differential behavior modeling. To revise the noisy signals, we hierarchically
suppress it across layers by extracting neighborhood information through
layer-wise neighborhood aggregation and further capturing user intentions
through adaptive cross-layer feature fusion. To correct plausible frequent
patterns, we propose an intensity-aware non-sampling strategy that dynamically
adjusts the weights of negative samples. Extensive experiments on two
benchmarks show that HiFIRec relatively improves HR@10 by 4.21%-6.81% over
several state-of-the-art methods.

### 6. [Using GPT to build a Project Management assistant for Jira environments](http://arxiv.org/pdf/2509.26014v1)

Authors: Joel Garcia-Escribano, Arkaitz Carbajo, Mikel Egaña Aranguren, Unai Lopez-Novoa

In the domain of Project Management, the sheer volume of data is a challenge
that project managers continually have to deal with. Effectively steering
projects from inception to completion requires handling of diverse information
streams, including timelines, budgetary considerations, and task dependencies.
To navigate this data-driven landscape with precision and agility, project
managers must rely on efficient and sophisticated tools. These tools have
become essential, as they enable project managers to streamline communication,
optimize resource allocation, and make informed decisions in real-time.
However, many of these tools have steep learning curves and require using
complex programming languages to retrieve the exact data that project managers
need. In this work we present JiraGPT Next, a software that uses the GPT Large
Language Model to ease the process by which project managers deal with large
amounts of data. It is conceived as an add-on for Jira, one of the most popular
Project Management tools, and provides a natural language interface to retrieve
information. This work presents the design decisions behind JiraGPT Next and an
evaluation of the accuracy of GPT in this context, including the effects of
providing different prompts to complete a particular task.

### 7. [Self-supervised learning for phase retrieval](http://arxiv.org/pdf/2509.26203v1)

Authors: Victor Sechaud, Patrice Abry, Laurent Jacques, Julián Tachella

In recent years, deep neural networks have emerged as a solution for inverse
imaging problems. These networks are generally trained using pairs of images:
one degraded and the other of high quality, the latter being called 'ground
truth'. However, in medical and scientific imaging, the lack of fully sampled
data limits supervised learning. Recent advances have made it possible to
reconstruct images from measurement data alone, eliminating the need for
references. However, these methods remain limited to linear problems, excluding
non-linear problems such as phase retrieval. We propose a self-supervised
method that overcomes this limitation in the case of phase retrieval by using
the natural invariance of images to translations.

### 8. [Analyzing BEV Suitability and Charging Strategies Using Italian Driving Data](http://arxiv.org/pdf/2509.26262v1)

Authors: Homa Jamalof, Luca Vassio, Danilo Giordano, Marco Mellia, Claudio De Tommasi

Battery Electric Vehicles (BEVs) are rapidly evolving from a niche
alternative to an established option for private transportation, often
replacing Internal Combustion Engine (ICE) vehicles. Despite growing interest,
significant barriers remain, including range anxiety, the inconvenience
associated with public charging stations, and higher costs. This study analyses
extensive telemetry data collected from 10,441 users using ICE vehicles in an
Italian province to assess the potential for switching to BEVs without changing
current travel behaviour. We evaluate to what extent the BEV models can fulfil
their mobility needs under different charging scenarios. To do so, we replicate
trips and parking events, simulating and monitoring the battery state of
charge. The analysis reveals the compromises between charging behaviours and
limited BEV autonomy. Assuming access to overnight charging, at least 35% of
the users could already adopt even low-capacity BEVs.

### 9. [SQUARE: Semantic Query-Augmented Fusion and Efficient Batch Reranking for Training-free Zero-Shot Composed Image Retrieval](http://arxiv.org/pdf/2509.26330v1)

Authors: Ren-Di Wu, Yu-Yen Lin, Huei-Fang Yang

Composed Image Retrieval (CIR) aims to retrieve target images that preserve
the visual content of a reference image while incorporating user-specified
textual modifications. Training-free zero-shot CIR (ZS-CIR) approaches, which
require no task-specific training or labeled data, are highly desirable, yet
accurately capturing user intent remains challenging. In this paper, we present
SQUARE, a novel two-stage training-free framework that leverages Multimodal
Large Language Models (MLLMs) to enhance ZS-CIR. In the Semantic
Query-Augmented Fusion (SQAF) stage, we enrich the query embedding derived from
a vision-language model (VLM) such as CLIP with MLLM-generated captions of the
target image. These captions provide high-level semantic guidance, enabling the
query to better capture the user's intent and improve global retrieval quality.
In the Efficient Batch Reranking (EBR) stage, top-ranked candidates are
presented as an image grid with visual marks to the MLLM, which performs joint
visual-semantic reasoning across all candidates. Our reranking strategy
operates in a single pass and yields more accurate rankings. Experiments show
that SQUARE, with its simplicity and effectiveness, delivers strong performance
on four standard CIR benchmarks. Notably, it maintains high performance even
with lightweight pre-trained, demonstrating its potential applicability.

### 10. [MR$^2$-Bench: Going Beyond Matching to Reasoning in Multimodal Retrieval](http://arxiv.org/pdf/2509.26378v1)

Authors: Junjie Zhou, Ze Liu, Lei Xiong, Jin-Ge Yao, Yueze Wang, Shitao Xiao, Fenfen Lin, Miguel Hu Chen, Zhicheng Dou, Siqi Bao, Defu Lian, Yongping Xiong, Zheng Liu

Multimodal retrieval is becoming a crucial component of modern AI
applications, yet its evaluation lags behind the demands of more realistic and
challenging scenarios. Existing benchmarks primarily probe surface-level
semantic correspondence (e.g., object-text matching) while failing to assess
the deeper reasoning required to capture complex relationships between visual
and textual information. To address this gap, we introduce MR$^2$-Bench, a
reasoning-intensive benchmark for multimodal retrieval. MR$^2$-Bench presents
the following critical values: 1) all tasks are reasoning-driven, going beyond
shallow matching to effectively assess models' capacity for logical, spatial,
and causal inference; 2) it features diverse multimodal data, such as natural
images, diagrams, and visual puzzles, enabling comprehensive evaluation across
content types; 3) it supports complex queries and documents containing multiple
images and covers diverse retrieval scenarios, more accurately reflecting
real-world applications. Our benchmark contains 1,309 curated queries, derived
either from manual collection and annotation or from selective consolidation of
public datasets. Despite achieving strong results on existing benchmarks,
current state-of-the-art models still struggle on MR$^2$-Bench: for example,
the leading Seed1.6-Embedding model attains a Recall@1 of 77.78 on MMEB, but
only 9.91 on MR$^2$-Bench. This substantial performance gap highlights both the
increased challenge posed by our benchmark and the pressing need for further
advances in reasoning-intensive multimodal retrieval. The dataset and
evaluation code will be made publicly available at
https://github.com/VectorSpaceLab/MR2-Bench.

### Machine Learning

### 1. [Effective Model Pruning](http://arxiv.org/pdf/2509.25606v1)

Authors: Yixuan Wang, Dan Guralnik, Saiedeh Akbari, Warren Dixon

We introduce Effective Model Pruning (EMP), a context-agnostic,
parameter-free rule addressing a fundamental question about pruning: how many
entries to keep. EMP does not prescribe how to score the parameters or prune
the models; instead, it supplies a universal adaptive threshold that can be
applied to any pruning criterion: weight magnitude, attention score, KAN
importance score, or even feature-level signals such as image pixel, and used
on structural parts or weights of the models. Given any score vector s, EMP
maps s to a built-in effective number N_eff which is inspired by the Inverse
Simpson index of contributors. Retaining the N_eff highest scoring entries and
zeroing the remainder yields sparse models with performance comparable to the
original dense networks across MLPs, CNNs, Transformers/LLMs, and KAN, in our
experiments. By leveraging the geometry of the simplex, we derive a tight lower
bound on the preserved mass s_eff (the sum of retained scores) over the
corresponding ordered probability simplex associated with the score vector s.
We further verify the effectiveness of N_eff by pruning the model with a scaled
threshold \b{eta}*N_eff across a variety of criteria and models. Experiments
suggest that the default \b{eta} = 1 yields a robust threshold for model
pruning while \b{eta} not equal to 1 still serves as an optional adjustment to
meet specific sparsity requirements.

### 2. [Layer-wise dynamic rank for compressing large language models](http://arxiv.org/pdf/2509.25622v1)

Authors: Zhendong Mi, Bian Sun, Grace Li Zhang, Shaoyi Huang

Large language models (LLMs) have rapidly scaled in size, bringing severe
memory and computational challenges that hinder their deployment. Singular
Value Decomposition (SVD)-based compression has emerged as an appealing
post-training compression technique for LLMs, yet most existing methods apply a
uniform compression ratio across all layers, implicitly assuming homogeneous
information included in various layers. This overlooks the substantial
intra-layer heterogeneity observed in LLMs, where middle layers tend to encode
richer information while early and late layers are more redundant. In this
work, we revisit the existing SVD-based compression method and propose D-Rank,
a framework with layer-wise balanced Dynamic Rank allocation for LLMs
compression. We first introduce effective rank as a principled metric to
measure the information density of weight matrices, and then allocate ranks via
a Lagrange multiplier-based optimization scheme to adaptively assign more
capacity to groups with higher information density under a fixed compression
ratio. Moreover, we rebalance the allocated ranks across attention layers to
account for their varying importance and extend D-Rank to latest LLMs with
grouped-query attention. Extensive experiments on various LLMs with different
scales across multiple compression ratios demonstrate that D-Rank consistently
outperforms SVD-LLM, ASVD, and Basis Sharing, achieving more than 15 lower
perplexity with LLaMA-3-8B model on C4 datasets at 20% compression ratio and up
to 5% higher zero-shot reasoning accuracy with LLaMA-7B model at 40%
compression ratio while achieving even higher throughput.

### 3. [Swift: An Autoregressive Consistency Model for Efficient Weather Forecasting](http://arxiv.org/pdf/2509.25631v1)

Authors: Jason Stock, Troy Arcomano, Rao Kotamarthi

Diffusion models offer a physically grounded framework for probabilistic
weather forecasting, but their typical reliance on slow, iterative solvers
during inference makes them impractical for subseasonal-to-seasonal (S2S)
applications where long lead-times and domain-driven calibration are essential.
To address this, we introduce Swift, a single-step consistency model that, for
the first time, enables autoregressive finetuning of a probability flow model
with a continuous ranked probability score (CRPS) objective. This eliminates
the need for multi-model ensembling or parameter perturbations. Results show
that Swift produces skillful 6-hourly forecasts that remain stable for up to 75
days, running $39\times$ faster than state-of-the-art diffusion baselines while
achieving forecast skill competitive with the numerical-based, operational IFS
ENS. This marks a step toward efficient and reliable ensemble forecasting from
medium-range to seasonal-scales.

### 4. [How Does Preconditioning Guide Feature Learning in Deep Neural Networks?](http://arxiv.org/pdf/2509.25637v1)

Authors: Kotaro Yoshida, Atsushi Nitanda

Preconditioning is widely used in machine learning to accelerate convergence
on the empirical risk, yet its role on the expected risk remains underexplored.
In this work, we investigate how preconditioning affects feature learning and
generalization performance. We first show that the input information available
to the model is conveyed solely through the Gram matrix defined by the
preconditioner's metric, thereby inducing a controllable spectral bias on
feature learning. Concretely, instantiating the preconditioner as the $p$-th
power of the input covariance matrix and within a single-index teacher model,
we prove that in generalization, the exponent $p$ and the alignment between the
teacher and the input spectrum are crucial factors. We further investigate how
the interplay between these factors influences feature learning from three
complementary perspectives: (i) Robustness to noise, (ii) Out-of-distribution
generalization, and (iii) Forward knowledge transfer. Our results indicate that
the learned feature representations closely mirror the spectral bias introduced
by the preconditioner -- favoring components that are emphasized and exhibiting
reduced sensitivity to those that are suppressed. Crucially, we demonstrate
that generalization is significantly enhanced when this spectral bias is
aligned with that of the teacher.

### 5. [Growing Winning Subnetworks, Not Pruning Them: A Paradigm for Density Discovery in Sparse Neural Networks](http://arxiv.org/pdf/2509.25665v1)

Authors: Qihang Yao, Constantine Dovrolis

The lottery ticket hypothesis suggests that dense networks contain sparse
subnetworks that can be trained in isolation to match full-model performance.
Existing approaches-iterative pruning, dynamic sparse training, and pruning at
initialization-either incur heavy retraining costs or assume the target density
is fixed in advance. We introduce Path Weight Magnitude Product-biased Random
growth (PWMPR), a constructive sparse-to-dense training paradigm that grows
networks rather than pruning them, while automatically discovering their
operating density. Starting from a sparse seed, PWMPR adds edges guided by
path-kernel-inspired scores, mitigates bottlenecks via randomization, and stops
when a logistic-fit rule detects plateauing accuracy. Experiments on CIFAR,
TinyImageNet, and ImageNet show that PWMPR approaches the performance of
IMP-derived lottery tickets-though at higher density-at substantially lower
cost (~1.5x dense vs. 3-4x for IMP). These results establish growth-based
density discovery as a promising paradigm that complements pruning and dynamic
sparsity.

### 6. [Guiding Mixture-of-Experts with Temporal Multimodal Interactions](http://arxiv.org/pdf/2509.25678v1)

Authors: Xing Han, Hsing-Huan Chung, Joydeep Ghosh, Paul Pu Liang, Suchi Saria

Mixture-of-Experts (MoE) architectures have become pivotal for large-scale
multimodal models. However, their routing mechanisms typically overlook the
informative, time-varying interaction dynamics between modalities. This
limitation hinders expert specialization, as the model cannot explicitly
leverage intrinsic modality relationships for effective reasoning. To address
this, we propose a novel framework that guides MoE routing using quantified
temporal interaction. A multimodal interaction-aware router learns to dispatch
tokens to experts based on the nature of their interactions. This dynamic
routing encourages experts to acquire generalizable interaction-processing
skills rather than merely learning task-specific features. Our framework builds
on a new formulation of temporal multimodal interaction dynamics, which are
used to guide expert routing. We first demonstrate that these temporal
multimodal interactions reveal meaningful patterns across applications, and
then show how they can be leveraged to improve both the design and performance
of MoE-based models. Comprehensive experiments on challenging multimodal
benchmarks validate our approach, demonstrating both enhanced performance and
improved interpretability.

### 7. [Minimalist Explanation Generation and Circuit Discovery](http://arxiv.org/pdf/2509.25686v1)

Authors: Pirzada Suhail, Aditya Anand, Amit Sethi

Machine learning models, by virtue of training, learn a large repertoire of
decision rules for any given input, and any one of these may suffice to justify
a prediction. However, in high-dimensional input spaces, such rules are
difficult to identify and interpret. In this paper, we introduce an
activation-matching based approach to generate minimal and faithful
explanations for the decisions of pre-trained image classifiers. We aim to
identify minimal explanations that not only preserve the model's decision but
are also concise and human-readable. To achieve this, we train a lightweight
autoencoder to produce binary masks that learns to highlight the decision-wise
critical regions of an image while discarding irrelevant background. The
training objective integrates activation alignment across multiple layers,
consistency at the output label, priors that encourage sparsity, and
compactness, along with a robustness constraint that enforces faithfulness. The
minimal explanations so generated also lead us to mechanistically interpreting
the model internals. In this regard we also introduce a circuit readout
procedure wherein using the explanation's forward pass and gradients, we
identify active channels and construct a channel-level graph, scoring
inter-layer edges by ingress weight magnitude times source activation and
feature-to-class links by classifier weight magnitude times feature activation.
Together, these contributions provide a practical bridge between minimal
input-level explanations and a mechanistic understanding of the internal
computations driving model decisions.

### 8. [Physics-Informed Learning for Human Whole-Body Kinematics Prediction via Sparse IMUs](http://arxiv.org/pdf/2509.25704v1)

Authors: Cheng Guo, Giuseppe L'Erario, Giulio Romualdi, Mattia Leonori, Marta Lorenzini, Arash Ajoudani, Daniele Pucci

Accurate and physically feasible human motion prediction is crucial for safe
and seamless human-robot collaboration. While recent advancements in human
motion capture enable real-time pose estimation, the practical value of many
existing approaches is limited by the lack of future predictions and
consideration of physical constraints. Conventional motion prediction schemes
rely heavily on past poses, which are not always available in real-world
scenarios. To address these limitations, we present a physics-informed learning
framework that integrates domain knowledge into both training and inference to
predict human motion using inertial measurements from only 5 IMUs. We propose a
network that accounts for the spatial characteristics of human movements.
During training, we incorporate forward and differential kinematics functions
as additional loss components to regularize the learned joint predictions. At
the inference stage, we refine the prediction from the previous iteration to
update a joint state buffer, which is used as extra inputs to the network.
Experimental results demonstrate that our approach achieves high accuracy,
smooth transitions between motions, and generalizes well to unseen subjects

### 9. [Adaptive Graph Coarsening for Efficient GNN Training](http://arxiv.org/pdf/2509.25706v1)

Authors: Rostyslav Olshevskyi, Madeline Navarro, Santiago Segarra

We propose an adaptive graph coarsening method to jointly learn graph neural
network (GNN) parameters and merge nodes via K-means clustering during
training. As real-world graphs grow larger, processing them directly becomes
increasingly challenging and sometimes infeasible. Tailoring algorithms to
large-scale data may sacrifice performance, so we instead consider graph
reduction to decrease the amount of data used during training. In particular,
we propose a method to simultaneously train a GNN and coarsen its graph by
partitioning nodes via K-means clustering based on their embeddings. Unlike
past graph coarsening works, our approach allows us to merge nodes during
training. Not only does this preclude coarsening as a preprocessing step, but
our node clusters can adapt to the learning task instead of relying solely on
graph connectivity and features. Thus, our method is amenable to scenarios that
are challenging for other methods, such as heterophilic data. We validate our
approach on both homophilic and heterophilic node classification datasets. We
further visualize relationships between node embeddings and their corresponding
clusters to illustrate that our coarsened graph adapts to the learning task
during training.

### 10. [Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking](http://arxiv.org/pdf/2509.25712v1)

Authors: Dengming Zhang, Xiaowen Ma, Zhenliang Ni, Zhenkai Wu, Han Shu, Xin Jiang, Xinghao Chen

Model merging, which combines multiple domain-specialized experts into a
single model, offers a practical path to endow Large Language Models (LLMs) and
Multimodal Large Language Models (MLLMs) with broad capabilities without the
cost of joint training or serving many models. However, training-free methods
rely on hand-tuned coefficients, whereas training-based methods primarily align
parameters rather than downstream task behavior and typically treat all layers
uniformly, ignoring inter-layer heterogeneity. We introduce Expert Merging, a
training-light method that learns a small set of layer-wise coefficients using
only unlabeled calibration data. The coefficients are optimized to explicitly
align the merged model's hidden states and logits with those of the
corresponding experts, with a coefficient regularizer for stability and
task-weighted losses for controllable trade-offs. To capture inter-layer
variation, Expert Merging++ augments this design with importance-guided
chunking: a normalized layer-importance metric, derived from learned
coefficients, task-vector magnitudes, and parameter counts, allocates more
chunk-wise coefficients to high-importance layers while keeping low-importance
layers lightweight. The result is a label-free, parameter-efficient, and
scalable approach to multi-expert model merging across LLMs and MLLMs. Across
MLLM backbones (InternVL and Qwen2-VL) and the LLM backbone (Mistral), our
method surpasses strong training-free and training-based merging baselines,
with Expert Merging++ delivering further gains and, in some cases, even
exceeding supervised Mixture Training. The source code is available at
https://github.com/Littleor/ExpertMerging.

### Neural and Evolutionary Computing

### 1. [Analysis of a Spatialized Brain-Body-Environment System](http://arxiv.org/pdf/2509.25640v1)

Authors: Denizhan Pak, Quan Le Thien, Christopher J. Agostino

The brain-body-environment framework studies adaptive behavior through
embodied and situated agents, emphasizing interactions between brains,
biomechanics, and environmental dynamics. However, many models often treat the
brain as a network of coupled ordinary differential equations (ODEs),
neglecting finer spatial properties which can not only increase model
complexity but also constrain observable neural dynamics. To address this
limitation, we propose a spatially extended approach using partial differential
equations (PDEs) for both the brain and body. As a case study, we revisit a
previously developed model of a child swinging, now incorporating spatial
dynamics. By considering the spatio-temporal properties of the brain and body,
we analyze how input location and propagation along a PDE influence behavior.
This approach offers new insights into the role of spatial organization in
adaptive behavior, bridging the gap between abstract neural models and the
physical constraints of embodied systems. Our results highlight the importance
of spatial dynamics in understanding brain-body-environment interactions.

### 2. [Scaling Equilibrium Propagation to Deeper Neural Network Architectures](http://arxiv.org/pdf/2509.26003v1)

Authors: Sankar Vinayak. E. P, Gopalakrishnan Srinivasan

Equilibrium propagation has been proposed as a biologically plausible
alternative to the backpropagation algorithm. The local nature of gradient
computations, combined with the use of convergent RNNs to reach equilibrium
states, make this approach well-suited for implementation on neuromorphic
hardware. However, previous studies on equilibrium propagation have been
restricted to networks containing only dense layers or relatively small
architectures with a few convolutional layers followed by a final dense layer.
These networks have a significant gap in accuracy compared to similarly sized
feedforward networks trained with backpropagation. In this work, we introduce
the Hopfield-Resnet architecture, which incorporates residual (or skip)
connections in Hopfield networks with clipped $\mathrm{ReLU}$ as the activation
function. The proposed architectural enhancements enable the training of
networks with nearly twice the number of layers reported in prior works. For
example, Hopfield-Resnet13 achieves 93.92\% accuracy on CIFAR-10, which is
$\approx$3.5\% higher than the previous best result and comparable to that
provided by Resnet13 trained using backpropagation.

### 3. [CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search](http://arxiv.org/pdf/2509.25862v1)

Authors: Olga Krestinskaya, Mohammed E. Fouda, Ahmed Eltawil, Khaled N. Salama

To maximize hardware efficiency and performance accuracy in Compute-In-Memory
(CIM)-based neural network accelerators for Artificial Intelligence (AI)
applications, co-optimizing both software and hardware design parameters is
essential. Manual tuning is impractical due to the vast number of parameters
and their complex interdependencies. To effectively automate the design and
optimization of CIM-based neural network accelerators, hardware-aware neural
architecture search (HW-NAS) techniques can be applied. This work introduces
CIMNAS, a joint model-quantization-hardware optimization framework for CIM
architectures. CIMNAS simultaneously searches across software parameters,
quantization policies, and a broad range of hardware parameters, incorporating
device-, circuit-, and architecture-level co-optimizations. CIMNAS experiments
were conducted over a search space of 9.9x10^85 potential parameter
combinations with the MobileNet model as a baseline and RRAM-based CIM
architecture. Evaluated on the ImageNet dataset, CIMNAS achieved a reduction in
energy-delay-area product (EDAP) ranging from 90.1x to 104.5x, an improvement
in TOPS/W between 4.68x and 4.82x, and an enhancement in TOPS/mm^2 from 11.3x
to 12.78x relative to various baselines, all while maintaining an accuracy of
73.81%. The adaptability and robustness of CIMNAS are demonstrated by extending
the framework to support the SRAM-based ResNet50 architecture, achieving up to
an 819.5x reduction in EDAP. Unlike other state-of-the-art methods, CIMNAS
achieves EDAP-focused optimization without any accuracy loss, generating
diverse software-hardware parameter combinations for high-performance CIM-based
neural network designs. The source code of CIMNAS is available at
https://github.com/OlgaKrestinskaya/CIMNAS.

### 4. [Real-time Noise Detection and Classification in Single-Channel EEG: A Lightweight Machine Learning Approach for EMG, White Noise, and EOG Artifacts](http://arxiv.org/pdf/2509.26058v1)

Authors: Hossein Enshaei, Pariya Jebreili, Sayed Mahmoud Sakahei

Electroencephalogram (EEG) artifact detection in real-world settings faces
significant challenges such as computational inefficiency in multi-channel
methods, poor robustness to simultaneous noise, and trade-offs between accuracy
and complexity in deep learning models. We propose a hybrid spectral-temporal
framework for real-time detection and classification of ocular (EOG), muscular
(EMG), and white noise artifacts in single-channel EEG. This method, in
contrast to other approaches, combines time-domain low-pass filtering
(targeting low-frequency EOG) and frequency-domain power spectral density (PSD)
analysis (capturing broad-spectrum EMG), followed by PCA-optimized feature
fusion to minimize redundancy while preserving discriminative information. This
feature engineering strategy allows a lightweight multi-layer perceptron (MLP)
architecture to outperform advanced CNNs and RNNs by achieving 99% accuracy at
low SNRs (SNR -7) dB and >90% accuracy in moderate noise (SNR 4 dB).
Additionally, this framework addresses the unexplored problem of simultaneous
multi-source contamination(EMG+EOG+white noise), where it maintains 96%
classification accuracy despite overlapping artifacts. With 30-second training
times (97% faster than CNNs) and robust performance across SNR levels, this
framework bridges the gap between clinical applicability and computational
efficiency, which enables real-time use in wearable brain-computer interfaces.
This work also challenges the ubiquitous dependence on model depth for EEG
artifact detection by demonstrating that domain-informed feature fusion
surpasses complex architecture in noisy scenarios.

### 5. [A general optimization framework for mapping local transition-state networks](http://arxiv.org/pdf/2509.26269v1)

Authors: Qichen Xu, Anna Delin

Understanding how complex systems transition between states requires mapping
the energy landscape that governs these changes. Local transition-state
networks reveal the barrier architecture that explains observed behaviour and
enables mechanism-based prediction across computational chemistry, biology, and
physics, yet current practice either prescribes endpoints or randomly samples
only a few saddles around an initial guess. We present a general optimization
framework that systematically expands local coverage by coupling a
multi-objective explorer with a bilayer minimum-mode kernel. The inner layer
uses Hessian-vector products to recover the lowest-curvature subspace (smallest
k eigenpairs), the outer layer optimizes on a reflected force to reach index-1
saddles, then a two-sided descent certifies connectivity. The GPU-based
pipeline is portable across autodiff backends and eigensolvers and, on large
atomistic-spin tests, matches explicit-Hessian accuracy while cutting peak
memory and wall time by orders of magnitude. Applied to a DFT-parameterized
N\'eel-type skyrmionic model, it recovers known routes and reveals previously
unreported mechanisms, including meron-antimeron-mediated N\'eel-type
skyrmionic duplication, annihilation, and chiral-droplet formation, enabling up
to 32 pathways between biskyrmion (Q=2) and biantiskyrmion (Q=-2). The same
core transfers to Cartesian atoms, automatically mapping canonical
rearrangements of a Ni(111) heptamer, underscoring the framework's generality.

### 6. [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](http://arxiv.org/pdf/2509.26507v1)

Authors: Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz

The relationship between computing systems and the brain has served as
motivation for pioneering theoreticians since John von Neumann and Alan Turing.
Uniform, scale-free biological networks, such as the brain, have powerful
properties, including generalizing over time, which is the main barrier for
Machine Learning on the path to Universal Reasoning Models.
  We introduce `Dragon Hatchling' (BDH), a new Large Language Model
architecture based on a scale-free biologically inspired network of \$n\$
locally-interacting neuron particles. BDH couples strong theoretical
foundations and inherent interpretability without sacrificing Transformer-like
performance.
  BDH is a practical, performant state-of-the-art attention-based state space
sequence learning architecture. In addition to being a graph model, BDH admits
a GPU-friendly formulation. It exhibits Transformer-like scaling laws:
empirically BDH rivals GPT2 performance on language and translation tasks, at
the same number of parameters (10M to 1B), for the same training data.
  BDH can be represented as a brain model. The working memory of BDH during
inference entirely relies on synaptic plasticity with Hebbian learning using
spiking neurons. We confirm empirically that specific, individual synapses
strengthen connection whenever BDH hears or reasons about a specific concept
while processing language inputs. The neuron interaction network of BDH is a
graph of high modularity with heavy-tailed degree distribution. The BDH model
is biologically plausible, explaining one possible mechanism which human
neurons could use to achieve speech.
  BDH is designed for interpretability. Activation vectors of BDH are sparse
and positive. We demonstrate monosemanticity in BDH on language tasks.
Interpretability of state, which goes beyond interpretability of neurons and
model parameters, is an inherent feature of the BDH architecture.

### Networking and Internet Architecture

### 1. [Oh-Trust: Overbooking and Hybrid Trading-empowered Resource Scheduling with Smart Reputation Update over Dynamic Edge Networks](http://arxiv.org/pdf/2509.25683v1)

Authors: Houyi Qi, Minghui Liwang, Liqun Fu, Xianbin Wang, Huaiyu Dai, Xiaoyu Xia

Incentive-driven computing resource sharing is crucial for meeting the
ever-growing demands of emerging mobile applications. Although conventional
spot trading offers a solution, it frequently leads to excessive overhead due
to the need for real-time trading related interactions. Likewise, traditional
futures trading, which depends on historical data, is susceptible to risks from
network dynamics. This paper explores a dynamic and uncertain edge network
comprising a computing platform, e.g., an edge server, that offers computing
services as resource seller, and various types of mobile users with diverse
resource demands as buyers, including fixed buyers (FBs) and uncertain
occasional buyers (OBs) with fluctuating needs. To facilitate efficient and
timely computing services, we propose an overbooking- and hybrid
trading-empowered resource scheduling mechanism with reputation update, termed
Oh-Trust. Particularly, our Oh-Trust incentivizes FBs to enter futures trading
by signing long-term contracts with the seller, while simultaneously attracting
OBs to spot trading, enhancing resource utilization and profitability for both
parties. Crucially, to adapt to market fluctuations, a smart reputation
updating mechanism is integrated, allowing for the timely renewal of long-term
contracts to optimize trading performance. Extensive simulations using
real-world datasets demonstrate the effectiveness of Oh-Trust across multiple
evaluation metrics.

### 2. [From Literature to Insights: Methodological Guidelines for Survey Writing in Communications Research](http://arxiv.org/pdf/2509.25828v1)

Authors: Dusit Niyato, Octavia A. Dobre, Trung Q. Duong, George K. Karagiannidis, Robert Schober

The rapid growth of communications and networking research has created an
unprecedented demand for high-quality survey and tutorial papers that can
synthesize vast bodies of literature into coherent understandings and
actionable insights. However, writing impactful survey papers presents
multifaceted challenges that demand substantial effort beyond traditional
research article composition. This article provides a systematic, practical
roadmap for prospective authors in the communications research community,
drawing upon extensive editorial experience from premier venues such as the
IEEE Communications Surveys & Tutorials. We present structured guidelines
covering seven essential aspects: strategic topic selection with novelty and
importance, systematic literature collection, effective structural
organization, critical review writing, tutorial content development with
emphasis on case studies, comprehensive illustration design that enhances
comprehension, and identification of future directions. Our goal is to enable
junior researchers to craft exceptional survey and tutorial articles that
enhance understanding and accelerate innovation within the communications and
networking research ecosystem.

### 3. [User-Centric Comparison of 5G NTN and DVB-S2/RCS2 Using OpenAirInterface and OpenSAND](http://arxiv.org/pdf/2509.26013v1)

Authors: Sumit Kumar, Juan Carlos Estrada-Jimenez, Ion Turcanu

The integration of satellite networks into next-generation mobile
communication systems has gained considerable momentum with the advent of 5G
Non-Terrestrial Networks (5G-NTN). Since established technologies like
DVB-S2/RCS2 are already widely used for satellite broadband, a detailed
comparison with emerging 5G NTN solutions is necessary to understand their
relative merits and guide deployment decisions. This paper presents a
user-centric, end-to-end evaluation of these technologies under realistic
traffic conditions, showing how differences in architecture and protocols
impact application-layer performance. Utilizing the 6G Sandbox platform, we
employ OpenAirInterface to emulate 5G NTN and OpenSAND for DVB-S2/RCS2,
replicating transparent payload GEO satellite scenarios under uniform downlink
conditions. A range of real-world applications, such as web browsing, file
downloads, and video streaming, are tested across both systems and
systematically analyzed. While the emulation lacks real-time capability, it
reveals key strengths and limitations of each approach, helping identify
suitable deployment scenarios for 5G NTN and DVB-S2/RCS2.

### 4. [Knowledge Defined Networking for 6G: A Reinforcement Learning Example for Resource Management](http://arxiv.org/pdf/2509.26075v1)

Authors: Erol Koçoğlu, Mehmet Ozdem, Tuğçe Bilen

6G networks are expected to revolutionize connectivity, offering significant
improvements in speed, capacity, and smart automation. However, existing
network designs will struggle to handle the demands of 6G, which include much
faster speeds, a huge increase in connected devices, lower energy consumption,
extremely quick response times, and better mobile broadband. To solve this
problem, incorporating the artificial intelligence (AI) technologies has been
proposed. This idea led to the concept of Knowledge-Defined Networking (KDN).
KDN promises many improvements, such as resource management, routing,
scheduling, clustering, and mobility prediction. The main goal of this study is
to optimize resource management using Reinforcement Learning.

### 5. [Flexible-Sector 6DMA Base Station: Modeling and Design](http://arxiv.org/pdf/2509.26086v1)

Authors: Yunli Li, Xiaoming Shi, Xiaodan Shao, Jie Xu, Rui Zhang

Six-dimensional movable antenna (6DMA) has emerged as a promising new
technology for future wireless networks, which can adaptively adjust the
three-dimensional (3D) positions and 3D rotations of antennas/antenna arrays
for performance enhancement. This paper proposes a novel cost-effective
6DMA-based base station (BS) architecture, termed the \textit{flexible-sector}
BS, which allows the deployed antennas to flexibly rotate and move along a
circular track, thus enabling common sector rotation and flexible antenna
allocation across sectors to adapt to the spatial user distribution
efficiently. In particular, we focus on the uplink transmission in a
single-cell system, where the flexible-sector BS receives independent messages
from multiple users. We introduce an angular-domain user distribution model,
which captures the users' spatial clustering or hot-spot distribution
effectively. Assuming the zero-forcing (ZF) based receiver applied at the BS to
decode multiuser signals, we derive the average sum rate achievable for the
users as a function of the common rotation of sectors and the antenna
allocation over them. Moreover, we develop a two-step algorithm to jointly
optimize the common sector rotation and antenna allocation to maximize the
average sum rate of all users. It is shown that the optimal antenna number in
each sector linearly increases with the number of users in it. It is also
revealed that under the most favorable user distribution, the achievable sum
rate gain increases in the order of $\log_{2}(B)$ in the regime of
asymptotically large number of antennas, where $B$ denotes the number of
sectors. Numerically results also show that as $B$ increases, the proposed
flexible-sector BS achieves higher sum rate, and it outperforms other benchmark
schemes, such as the traditional fixed-sector BS as well as the BS with sector
rotation or antenna allocation optimization only.

### 6. [Target Wake Time Scheduling for Time-sensitive and Energy-efficient Wi-Fi Networks](http://arxiv.org/pdf/2509.26245v1)

Authors: Fabio Busacca, Corrado Puligheddu, Francesco Raviglione, Riccardo Rusca, Claudio Casetti, Carla Fabiana Chiasserini, Sergio Palazzo

Time Sensitive Networking (TSN) is fundamental for the reliable, low-latency
networks that will enable the Industrial Internet of Things (IIoT). Wi-Fi has
historically been considered unfit for TSN, as channel contention and
collisions prevent deterministic transmission delays. However, this issue can
be overcome by using Target Wake Time (TWT), which enables the access point to
instruct Wi-Fi stations to wake up and transmit in non-overlapping TWT Service
Periods (SPs), and sleep in the remaining time. In this paper, we first
formulate the TWT Acceptance and Scheduling Problem (TASP), with the objective
to schedule TWT SPs that maximize traffic throughput and energy efficiency
while respecting Age of Information (AoI) constraints. Then, due to TASP being
NP-hard, we propose the TASP Efficient Resolver (TASPER), a heuristic strategy
to find near-optimal solutions efficiently. Using a TWT simulator based on
ns-3, we compare TASPER to several baselines, including HSA, a state-of-the-art
solution originally designed for WirelessHART networks. We demonstrate that
TASPER obtains up to 24.97% lower mean transmission rejection cost and saves up
to 14.86% more energy compared to the leading baseline, ShortestFirst, in a
challenging, large-scale scenario. Additionally, when compared to HSA, TASPER
also reduces the energy consumption by 34% and reduces the mean rejection cost
by 26%. Furthermore, we validate TASPER on our IIoT testbed, which comprises 10
commercial TWT-compatible stations, observing that our solution admits more
transmissions than the best baseline strategy, without violating any AoI
deadline.

### 7. [Introducing Large Language Models in the Design Flow of Time Sensitive Networking](http://arxiv.org/pdf/2509.26368v1)

Authors: Rubi Debnath, Luxi Zhao, Mohammadreza Barzegaran, Sebastian Steinhorst

The growing demand for real-time, safety-critical systems has significantly
increased both the adoption and complexity of Time Sensitive Networking (TSN).
Configuring an optimized TSN network is highly challenging, requiring careful
planning, design, verification, validation, and deployment. Large Language
Models (LLMs) have recently demonstrated strong capabilities in solving complex
tasks, positioning them as promising candidates for automating end-to-end TSN
deployment, referred to as TSN orchestration. This paper outlines the steps
involved in TSN orchestration and the associated challenges. To assess the
capabilities of existing LLM models, we conduct an initial proof-of-concept
case study focused on TSN configuration across multiple models. Building on
these insights, we propose an LLM-assisted orchestration framework. Unlike
prior research on LLMs in computer networks, which has concentrated on general
configuration and management, TSN-specific orchestration has not yet been
investigated. We present the building blocks for automating TSN using LLMs,
describe the proposed pipeline, and analyze opportunities and limitations for
real-world deployment. Finally, we highlight key challenges and research
directions, including the development of TSN-focused datasets, standardized
benchmark suites, and the integration of external tools such as Network
Calculus (NC) engines and simulators. This work provides the first roadmap
toward assessing the feasibility of LLM-assisted TSN orchestration.

### 8. [User-Centric Communication Service Provision for Edge-Assisted Mobile Augmented Reality](http://arxiv.org/pdf/2509.25905v1)

Authors: Conghao Zhou, Jie Gao, Shisheng Hu, Nan Cheng, Weihua Zhuang, Xuemin Shen

Future 6G networks are envisioned to facilitate edge-assisted mobile
augmented reality (MAR) via strengthening the collaboration between MAR devices
and edge servers. In order to provide immersive user experiences, MAR devices
must timely upload camera frames to an edge server for simultaneous
localization and mapping (SLAM)-based device pose tracking. In this paper, to
cope with user-specific and non-stationary uplink data traffic, we develop a
digital twin (DT)-based approach for user-centric communication service
provision for MAR. Specifically, to establish DTs for individual MAR devices,
we first construct a data model customized for MAR that captures the intricate
impact of the SLAM-based frame uploading mechanism on the user-specific data
traffic pattern. We then define two DT operation functions that cooperatively
enable adaptive switching between different data-driven models for capturing
non-stationary data traffic. Leveraging the user-oriented data management
introduced by DTs, we propose an algorithm for network resource management that
ensures the timeliness of frame uploading and the robustness against inherent
inaccuracies in data traffic modeling for individual MAR devices. Trace-driven
simulation results demonstrate that the user-centric communication service
provision achieves a 14.2% increase in meeting the camera frame uploading delay
requirement in comparison with the slicing-based communication service
provision widely used for 5G.

### 9. [OpenID Connect for Agents (OIDC-A) 1.0: A Standard Extension for LLM-Based Agent Identity and Authorization](http://arxiv.org/pdf/2509.25974v1)

Authors: Subramanya Nagabhushanaradhya

OpenID Connect for Agents (OIDC-A) 1.0 is an extension to OpenID Connect Core
1.0 that provides a comprehensive framework for representing, authenticating,
and authorizing LLM-based agents within the OAuth 2.0 ecosystem. As autonomous
AI agents become increasingly prevalent in digital systems, there is a critical
need for standardized protocols to establish agent identity, verify agent
attestation, represent delegation chains, and enable fine-grained authorization
based on agent attributes. This specification defines standard claims,
endpoints, and protocols that address these requirements while maintaining
compatibility with existing OAuth 2.0 and OpenID Connect infrastructure. The
proposed framework introduces mechanisms for agent identity representation,
delegation chain validation, attestation verification, and capability-based
authorization, providing a foundation for secure and trustworthy
agent-to-service interactions in modern distributed systems.

### 10. [Toward an Unbiased Collective Memory for Efficient LLM-Based Agentic 6G Cross-Domain Management](http://arxiv.org/pdf/2509.26200v1)

Authors: Hatim Chergui, Miguel Catalan Cid, Pouria Sayyad Khodashenas, Daniel Camps Mur, Christos Verikoukis

This paper introduces a novel framework for proactive cross-domain resource
orchestration in 6G RAN-Edge networks, featuring large language model
(LLM)-augmented agents. The system comprises specialized RAN (energy
efficiency) and Edge (latency assurance) agents that engage in iterative
negotiation, supported by advanced reasoning and planning capabilities. Agents
dynamically interact with a digital twin (DT) to test their proposals and
leverage a long-term collective memory where their joint successful and failed
agreements along with the related network contexts are distilled into
strategies to either follow or avoid and subsequently stored. Given that agents
are subject to a plethora of cognitive distortions when retrieving those past
experiences -- such as primacy, recency, confirmation and availability biases
-- we propose in this work a novel unbiased memory design (A reusable mockup
version of the unbiased memory source code is available for non-commercial use
at https://github.com/HatimChergui/unbiased-collective-memory). featuring (i)
semantic retrieval of past strategies via Jaccard similarity; (ii) learning
from failures through amplified weighting of SLA violations and mandatory
inclusion of failed negotiation cases to mitigate confirmation bias; (iii)
diversity enforcement to minimize availability bias and (iv) recency and
primacy weighting with slow decay to counteract temporal biases. Evaluation
results showcase the impact of existing biases and how the unbiased memory
allows to tackle them by learning from both successful and failed strategies,
either present or old, resulting in $\times 4.5$ and $\times 3.5$ reductions of
unresolved negotiations compared to non-memory and vanilla memory baselines,
respectively, while totally mitigating SLA violations as well as improving
latency and energy saving distributions.

### Robotics

### 1. [Hierarchical Diffusion Motion Planning with Task-Conditioned Uncertainty-Aware Priors](http://arxiv.org/pdf/2509.25685v1)

Authors: Amelie Minji Kim, Anqi Wu, Ye Zhao

We propose a novel hierarchical diffusion planner that embeds task and motion
structure directly in the noise model. Unlike standard diffusion-based planners
that use zero-mean, isotropic Gaussian noise, we employ a family of
task-conditioned structured Gaussians whose means and covariances are derived
from Gaussian Process Motion Planning (GPMP): sparse, task-centric key states
or their associated timings (or both) are treated as noisy observations to
produce a prior instance. We first generalize the standard diffusion process to
biased, non-isotropic corruption with closed-form forward and posterior
expressions. Building on this, our hierarchy separates prior instantiation from
trajectory denoising: the upper level instantiates a task-conditioned
structured Gaussian (mean and covariance), and the lower level denoises the
full trajectory under that fixed prior. Experiments on Maze2D goal-reaching and
KUKA block stacking show improved success rates, smoother trajectories, and
stronger task alignment compared to isotropic baselines. Ablation studies
indicate that explicitly structuring the corruption process offers benefits
beyond simply conditioning the neural network. Overall, our method concentrates
probability mass of prior near feasible, smooth, and semantically meaningful
trajectories while maintaining tractability. Our project page is available at
https://hta-diffusion.github.io.

### 2. [OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation](http://arxiv.org/pdf/2509.25687v1)

Authors: Xinda Xue, Junjun Hu, Minghua Luo, Xie Shichao, Jintao Chen, Zixun Xie, Quan Kuichen, Guo Wei, Mu Xu, Zedong Chu

Embodied navigation presents a core challenge for intelligent robots,
requiring the comprehension of visual environments, natural language
instructions, and autonomous exploration. Existing models often fall short in
offering a unified solution across diverse navigation paradigms, resulting in
low success rates and limited generalization. We introduce OmniNav, a unified
framework addressing instruct-goal, object-goal, point-goal navigation, and
frontier-based exploration within a single architecture. Our approach features
a lightweight, low-latency policy that accurately predicts continuous-space
waypoints (coordinates and orientations). This policy surpasses action-chunk
methods in precision and supports real-world deployment at control frequencies
up to 5 Hz. Architecturally, OmniNav employs a fast-slow system design: a fast
module generates waypoints using short-horizon visual context and subtasks,
while a slow module performs deliberative planning with long-horizon
observations and candidate frontiers to select subsequent subgoals and
subtasks. This collaboration enhances path efficiency and maintains trajectory
coherence, particularly in exploration and memory-intensive scenarios.
Crucially, we identify that the primary bottleneck isn't merely navigation
policy learning, but a robust understanding of general instructions and
objects. To boost generalization, OmniNav integrates large-scale,
general-purpose training datasets, including those for image captioning and
visual recognition, into a joint multi-task regimen. This significantly
improves success rates and robustness. Extensive experiments confirm OmniNav's
state-of-the-art performance across various navigation benchmarks, with
real-world deployment further validating its efficacy. OmniNav provides
practical insights for embodied navigation, charting a scalable path towards
versatile, highly generalizable robotic intelligence.

### 3. [VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning](http://arxiv.org/pdf/2509.25718v1)

Authors: Si-Cheng Wang, Tian-Yu Xiang, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Shuang-Yi Wang, Ao-Qun Jin, Zeng-Guang Hou

Reinforcement learning (RL) is a promising avenue for post-training
vision-language-action (VLA) models, but practical deployment is hindered by
sparse rewards and unstable training. This work mitigates these challenges by
introducing an action chunk based on proximal policy optimization (PPO) with
behavior cloning using self-collected demonstrations. Aggregating consecutive
actions into chunks improves the temporal consistency of the policy and the
density of informative feedback. In addition, an auxiliary behavior cloning
loss is applied with a dynamically updated demonstration buffer that
continually collects high-quality task trials during training. The relative
weight between the action-chunked PPO objective and the self behavior clone
auxiliary loss is adapted online to stabilize the post-training process.
Experiments on the MetaWorld benchmark indicate improved performance over
supervised fine-tuning, achieving a high success rate (0.93) and few steps to
success (42.17). These results demonstrate the viability of RL for VLA
post-training and help lay the groundwork for downstream VLA applications.

### 4. [TacRefineNet: Tactile-Only Grasp Refinement Between Arbitrary In-Hand Object Poses](http://arxiv.org/pdf/2509.25746v1)

Authors: Shuaijun Wang, Haoran Zhou, Diyun Xiang, Yangwei You

Despite progress in both traditional dexterous grasping pipelines and recent
Vision-Language-Action (VLA) approaches, the grasp execution stage remains
prone to pose inaccuracies, especially in long-horizon tasks, which undermines
overall performance. To address this "last-mile" challenge, we propose
TacRefineNet, a tactile-only framework that achieves fine in-hand pose
refinement of known objects in arbitrary target poses using multi-finger
fingertip sensing. Our method iteratively adjusts the end-effector pose based
on tactile feedback, aligning the object to the desired configuration. We
design a multi-branch policy network that fuses tactile inputs from multiple
fingers along with proprioception to predict precise control updates. To train
this policy, we combine large-scale simulated data from a physics-based tactile
model in MuJoCo with real-world data collected from a physical system.
Comparative experiments show that pretraining on simulated data and fine-tuning
with a small amount of real data significantly improves performance over
simulation-only training. Extensive real-world experiments validate the
effectiveness of the method, achieving millimeter-level grasp accuracy using
only tactile input. To our knowledge, this is the first method to enable
arbitrary in-hand pose refinement via multi-finger tactile sensing alone.
Project website is available at https://sites.google.com/view/tacrefinenet

### 5. [Best of Sim and Real: Decoupled Visuomotor Manipulation via Learning Control in Simulation and Perception in Real](http://arxiv.org/pdf/2509.25747v1)

Authors: Jialei Huang, Zhaoheng Yin, Yingdong Hu, Shuo Wang, Xingyu Lin, Yang Gao

Sim-to-real transfer remains a fundamental challenge in robot manipulation
due to the entanglement of perception and control in end-to-end learning. We
present a decoupled framework that learns each component where it is most
reliable: control policies are trained in simulation with privileged state to
master spatial layouts and manipulation dynamics, while perception is adapted
only at deployment to bridge real observations to the frozen control policy.
Our key insight is that control strategies and action patterns are universal
across environments and can be learned in simulation through systematic
randomization, while perception is inherently domain-specific and must be
learned where visual observations are authentic. Unlike existing end-to-end
approaches that require extensive real-world data, our method achieves strong
performance with only 10-20 real demonstrations by reducing the complex
sim-to-real problem to a structured perception alignment task. We validate our
approach on tabletop manipulation tasks, demonstrating superior data efficiency
and out-of-distribution generalization compared to end-to-end baselines. The
learned policies successfully handle object positions and scales beyond the
training distribution, confirming that decoupling perception from control
fundamentally improves sim-to-real transfer.

### 6. [Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](http://arxiv.org/pdf/2509.25822v1)

Authors: Jing Wang, Weiting Peng, Jing Tang, Zeyu Gong, Xihua Wang, Bo Tao, Li Cheng

Existing imitation learning methods decouple perception and action, which
overlooks the causal reciprocity between sensory representations and action
execution that humans naturally leverage for adaptive behaviors. To bridge this
gap, we introduce Action--Guided Diffusion Policy (DP--AG), a unified
representation learning that explicitly models a dynamic interplay between
perception and action through probabilistic latent dynamics. DP--AG encodes
latent observations into a Gaussian posterior via variational inference and
evolves them using an action-guided SDE, where the Vector-Jacobian Product
(VJP) of the diffusion policy's noise predictions serves as a structured
stochastic force driving latent updates. To promote bidirectional learning
between perception and action, we introduce a cycle--consistent contrastive
loss that organizes the gradient flow of the noise predictor into a coherent
perception--action loop, enforcing mutually consistent transitions in both
latent updates and action refinements. Theoretically, we derive a variational
lower bound for the action-guided SDE, and prove that the contrastive objective
enhances continuity in both latent and action trajectories. Empirically, DP--AG
significantly outperforms state--of--the--art methods across simulation
benchmarks and real-world UR5 manipulation tasks. As a result, our DP--AG
offers a promising step toward bridging biological adaptability and artificial
policy learning.

### 7. [Reinforced Embodied Planning with Verifiable Reward for Real-World Robotic Manipulation](http://arxiv.org/pdf/2509.25852v1)

Authors: Zitong Bo, Yue Hu, Jinming Ma, Mingliang Zhou, Junhui Yin, Yachen Kang, Yuqi Liu, Tong Wu, Diyun Xiang, Hao Chen

Enabling robots to execute long-horizon manipulation tasks from free-form
language instructions remains a fundamental challenge in embodied AI. While
vision-language models (VLMs) have shown promise as high-level planners, their
deployment in the real world is hindered by two gaps: (i) the scarcity of
large-scale, sequential manipulation data that couples natural language with
multi-step action plans, and (ii) the absence of dense, interpretable rewards
for fine-tuning VLMs on planning objectives. To address these issues, we
propose REVER, a framework that empowers VLMs to generate and validate
long-horizon manipulation plans from natural language instructions in
real-world scenarios. Under REVER we train and release RoboFarseer, a VLM
incentivized to emit chain-of-thought that perform temporal and spatial
reasoning, ensuring physically plausible and logically coherent plans. To
obtain training data, we leverage the Universal Manipulation Interface
framework to capture hardware-agnostic demonstrations of atomic skills. An
automated annotation engine converts each demonstration into
vision-instruction-plan triplet. We introduce a verifiable reward that scores
the generated plan by its ordered bipartite matching overlap with the
ground-truth skill sequence. At run time, the fine-tuned VLM functions both as
a planner and as a monitor, verifying step-wise completion. RoboFarseer matches
or exceeds the performance of proprietary models that are orders of magnitude
larger, while on open-ended planning it surpasses the best baseline by more
than 40%. In real-world, long-horizon tasks, the complete system boosts overall
success by roughly 60% compared with the same low-level controller without the
planner. We will open-source both the dataset and the trained model upon
publication.

### 8. [State Estimation for Compliant and Morphologically Adaptive Robots](http://arxiv.org/pdf/2509.25945v1)

Authors: Valentin Yuryev, Max Polzin, Josie Hughes

Locomotion robots with active or passive compliance can show robustness to
uncertain scenarios, which can be promising for agricultural, research and
environmental industries. However, state estimation for these robots is
challenging due to the lack of rigid-body assumptions and kinematic changes
from morphing. We propose a method to estimate typical rigid-body states
alongside compliance-related states, such as soft robot shape in different
morphologies and locomotion modes. Our neural network-based state estimator
uses a history of states and a mechanism to directly influence unreliable
sensors. We test our framework on the GOAT platform, a robot capable of passive
compliance and active morphing for extreme outdoor terrain. The network is
trained on motion capture data in a novel compliance-centric frame that
accounts for morphing-related states. Our method predicts shape-related
measurements within 4.2% of the robot's size, velocities within 6.3% and 2.4%
of the top linear and angular speeds, respectively, and orientation within 1.5
degrees. We also demonstrate a 300% increase in travel range during a motor
malfunction when using our estimator for closed-loop autonomous outdoor
operation.

### 9. [Towards Intuitive Human-Robot Interaction through Embodied Gesture-Driven Control with Woven Tactile Skins](http://arxiv.org/pdf/2509.25951v1)

Authors: ChunPing Lam, Xiangjia Chen, Chenming Wu, Hao Chen, Binzhi Sun, Guoxin Fang, Charlie C. L. Wang, Chengkai Dai, Yeung Yam

This paper presents a novel human-robot interaction (HRI) framework that
enables intuitive gesture-driven control through a capacitance-based woven
tactile skin. Unlike conventional interfaces that rely on panels or handheld
devices, the woven tactile skin integrates seamlessly with curved robot
surfaces, enabling embodied interaction and narrowing the gap between human
intent and robot response. Its woven design combines fabric-like flexibility
with structural stability and dense multi-channel sensing through the
interlaced conductive threads. Building on this capability, we define a
gesture-action mapping of 14 single- and multi-touch gestures that cover
representative robot commands, including task-space motion and auxiliary
functions. A lightweight convolution-transformer model designed for gesture
recognition in real time achieves an accuracy of near-100%, outperforming prior
baseline approaches. Experiments on robot arm tasks, including pick-and-place
and pouring, demonstrate that our system reduces task completion time by up to
57% compared with keyboard panels and teach pendants. Overall, our proposed
framework demonstrates a practical pathway toward more natural and efficient
embodied HRI.

### 10. [MUVLA: Learning to Explore Object Navigation via Map Understanding](http://arxiv.org/pdf/2509.25966v1)

Authors: Peilong Han, Fan Jia, Min Zhang, Yutao Qiu, Hongyao Tang, Yan Zheng, Tiancai Wang, Jianye Hao

In this paper, we present MUVLA, a Map Understanding Vision-Language-Action
model tailored for object navigation. It leverages semantic map abstractions to
unify and structure historical information, encoding spatial context in a
compact and consistent form. MUVLA takes the current and history observations,
as well as the semantic map, as inputs and predicts the action sequence based
on the description of goal object. Furthermore, it amplifies supervision
through reward-guided return modeling based on dense short-horizon progress
signals, enabling the model to develop a detailed understanding of action value
for reward maximization. MUVLA employs a three-stage training pipeline:
learning map-level spatial understanding, imitating behaviors from
mixed-quality demonstrations, and reward amplification. This strategy allows
MUVLA to unify diverse demonstrations into a robust spatial representation and
generate more rational exploration strategies. Experiments on HM3D and Gibson
benchmarks demonstrate that MUVLA achieves great generalization and learns
effective exploration behaviors even from low-quality or partially successful
trajectories.

### Software Engineering

### 1. [M&SCheck: Towards a Checklist to Support Software Engineering Newcomers to the Modeling and Simulation Area](http://arxiv.org/pdf/2509.25625v1)

Authors: Luiza Martins de Freitas Cintra, Philipp Zech, Mohamad Kassab, Eliomar Araújo Lima, Sofia Larissa da Costa Paiva, Valdemar Vicente Graciano Neto

The advent of increasingly complex and dynamic ecosystems, such as digital
twins (DT), smart cities and Industry 4.0 and 5.0, has made evident the need to
include modeling and simulation (M&S) in the software development life cycle.
Such disruptive systems include simulation models in their own architecture
(such as DT) or require the use of simulation models to represent the high
degree of movement and the multiplicity of interactions that occur between the
involved systems. However, when software engineers (particularly the newcomers)
need to use M&S in their projects, they often pose themselves an important
question: which formalism should I use? In this direction, the main
contribution of this paper is the establishment of a preliminary checklist with
questions to assist beginners in M&S in choosing the most appropriate paradigm
to solve their problems. The checklist is based on three main formalisms: DEVS,
System Dynamics and Agent-Based Simulation. A pilot study was carried out and
an expert was consulted. The preliminary results show (i) conformance between
the suggestion given by the checklist and the formalism selected in the
original studies used as input for evaluating the checklist, and (ii) a
positive feedback from the expert.

### 2. [Explainable Fault Localization for Programming Assignments via LLM-Guided Annotation](http://arxiv.org/pdf/2509.25676v1)

Authors: Fang Liu, Tianze Wang, Li Zhang, Zheyu Yang, Jing Jiang, Zian Sun

Providing timely and personalized guidance for students' programming
assignments, offers significant practical value for helping students complete
assignments and enhance their learning. In recent years, various automated
Fault Localization (FL) techniques have demonstrated promising results in
identifying errors in programs. However, existing FL techniques face challenges
when applied to educational contexts. Most approaches operate at the method
level without explanatory feedback, resulting in granularity too coarse for
students who need actionable insights to identify and fix their errors. While
some approaches attempt line-level fault localization, they often depend on
predicting line numbers directly in numerical form, which is ill-suited to
LLMs. To address these challenges, we propose FLAME, a fine-grained,
explainable Fault Localization method tailored for programming assignments via
LLM-guided Annotation and Model Ensemble. FLAME leverages rich contextual
information specific to programming assignments to guide LLMs in identifying
faulty code lines. Instead of directly predicting line numbers, we prompt the
LLM to annotate faulty code lines with detailed explanations, enhancing both
localization accuracy and educational value. To further improve reliability, we
introduce a weighted multi-model voting strategy that aggregates results from
multiple LLMs to determine the suspiciousness of each code line. Extensive
experimental results demonstrate that FLAME outperforms state-of-the-art fault
localization baselines on programming assignments, successfully localizing 207
more faults at top-1 over the best-performing baseline. Beyond educational
contexts, FLAME also generalizes effectively to general-purpose software
codebases, outperforming all baselines on the Defects4J benchmark.

### 3. [Are Classical Clone Detectors Good Enough For the AI Era?](http://arxiv.org/pdf/2509.25754v1)

Authors: Ajmain Inqiad Alam, Palash Roy, Farouq Al-omari, Chanchal Roy, Banani Roy, Kevin Schneider

The increasing adoption of AI-generated code has reshaped modern software
development, introducing syntactic and semantic variations in cloned code.
Unlike traditional human-written clones, AI-generated clones exhibit systematic
syntactic patterns and semantic differences learned from large-scale training
data. This shift presents new challenges for classical code clone detection
(CCD) tools, which have historically been validated primarily on human-authored
codebases and optimized to detect syntactic (Type 1-3) and limited semantic
clones. Given that AI-generated code can produce both syntactic and complex
semantic clones, it is essential to evaluate the effectiveness of classical CCD
tools within this new paradigm. In this paper, we systematically evaluate nine
widely used CCD tools using GPTCloneBench, a benchmark containing
GPT-3-generated clones. To contextualize and validate our results, we further
test these detectors on established human-authored benchmarks, BigCloneBench
and SemanticCloneBench, to measure differences in performance between
traditional and AI-generated clones. Our analysis demonstrates that classical
CCD tools, particularly those enhanced by effective normalization techniques,
retain considerable effectiveness against AI-generated clones, while some
exhibit notable performance variation compared to traditional benchmarks. This
paper contributes by (1) evaluating classical CCD tools against AI-generated
clones, providing critical insights into their current strengths and
limitations; (2) highlighting the role of normalization techniques in improving
detection accuracy; and (3) delivering detailed scalability and execution-time
analyses to support practical CCD tool selection.

### 4. [LogPilot: Intent-aware and Scalable Alert Diagnosis for Large-scale Online Service Systems](http://arxiv.org/pdf/2509.25874v1)

Authors: Zhihan Jiang, Jinyang Liu, Yichen Li, Haiyu Huang, Xiao He, Tieying Zhang, Jianjun Chen, Yi Li, Rui Shi, Michael R. Lyu

Effective alert diagnosis is essential for ensuring the reliability of
large-scale online service systems. However, on-call engineers are often
burdened with manually inspecting massive volumes of logs to identify root
causes. While various automated tools have been proposed, they struggle in
practice due to alert-agnostic log scoping and the inability to organize
complex data effectively for reasoning. To overcome these limitations, we
introduce LogPilot, an intent-aware and scalable framework powered by Large
Language Models (LLMs) for automated log-based alert diagnosis. LogPilot
introduces an intent-aware approach, interpreting the logic in alert
definitions (e.g., PromQL) to precisely identify causally related logs and
requests. To achieve scalability, it reconstructs each request's execution into
a spatiotemporal log chain, clusters similar chains to identify recurring
execution patterns, and provides representative samples to the LLMs for
diagnosis. This clustering-based approach ensures the input is both rich in
diagnostic detail and compact enough to fit within the LLM's context window.
Evaluated on real-world alerts from Volcano Engine Cloud, LogPilot improves the
usefulness of root cause summarization by 50.34% and exact localization
accuracy by 54.79% over state-of-the-art methods. With a diagnosis time under
one minute and a cost of only $0.074 per alert, LogPilot has been successfully
deployed in production, offering an automated and practical solution for
service alert diagnosis.

### 5. [Red Teaming Program Repair Agents: When Correct Patches can Hide Vulnerabilities](http://arxiv.org/pdf/2509.25894v1)

Authors: Simin Chen, Yixin He, Suman Jana, Baishakhi Ray

LLM-based agents are increasingly deployed for software maintenance tasks
such as automated program repair (APR). APR agents automatically fetch GitHub
issues and use backend LLMs to generate patches that fix the reported bugs.
However, existing work primarily focuses on the functional correctness of
APR-generated patches, whether they pass hidden or regression tests, while
largely ignoring potential security risks. Given the openness of platforms like
GitHub, where any user can raise issues and participate in discussions, an
important question arises: Can an adversarial user submit a valid issue on
GitHub that misleads an LLM-based agent into generating a functionally correct
but vulnerable patch? To answer this question, we propose SWExploit, which
generates adversarial issue statements designed to make APR agents produce
patches that are functionally correct yet vulnerable. SWExploit operates in
three main steps: (1) program analysis to identify potential injection points
for vulnerable payloads; (2) adversarial issue generation to provide misleading
reproduction and error information while preserving the original issue
semantics; and (3) iterative refinement of the adversarial issue statements
based on the outputs of the APR agents. Empirical evaluation on three agent
pipelines and five backend LLMs shows that SWExploit can produce patches that
are both functionally correct and vulnerable (the attack success rate on the
correct patch could reach 0.91, whereas the baseline ASRs are all below 0.20).
Based on our evaluation, we are the first to challenge the traditional
assumption that a patch passing all tests is inherently reliable and secure,
highlighting critical limitations in the current evaluation paradigm for APR
agents.

### 6. [Evaluating the impact of code smell refactoring on the energy consumption of Android applications](http://arxiv.org/pdf/2509.26031v1)

Authors: Hina Anwar, Dietmar Pfahl, Satish N. Srirama

Energy consumption of mobile apps is a domain that is receiving a lot of
attention from researchers. Recent studies indicate that the energy consumption
of mobile devices could be improved by improving the quality of mobile apps.
Frequent refactoring is one way of achieving this goal. In this paper, we
explore the performance and energy impact of several common code refactorings
in Android apps. Experimental results indicate that some code smell
refactorings positively impact the energy consumption of Android apps.
Refactoring of the code smells "Duplicated code" and "Type checking" reduce
energy consumption by up to 10.8%. Significant reduction in energy consumption,
however, does not seem to be directly related to the increase or decrease of
execution time. In addition, the energy impact over permutations of code smell
refactorings in the selected Android apps was small. When analyzing the order
in which refactorings were made across code smell types, it turned out that
some permutations resulted in a reduction and some in an increase of energy
consumption for the analyzed apps. More research needs to be done to
investigate how factors like size and age of software apps, experience, and
number of contributors to app development correlate with (a) the number and
type of code smells found and (b) the impact of energy consumption and
performance after refactoring.

### 7. [A Multi-Language Object-Oriented Programming Benchmark for Large Language Models](http://arxiv.org/pdf/2509.26111v1)

Authors: Shuai Wang, Liang Ding, Li Shen, Yong Luo, Han Hu, Lefei Zhang, Fu Lin

Establishing fair and robust benchmarks is essential for evaluating
intelligent code generation by large language models (LLMs). Our survey of 35
existing benchmarks uncovers three major imbalances: 85.7% focus on a single
programming language; 94.3% target only function-level or statement-level
tasks; and over 80% include fewer than ten test cases on average. To address
these gaps, we propose MultiOOP, a multi-language object-oriented programming
benchmark covering six popular languages (Python, PHP, C++, C#, Java,
JavaScript) with 267 tasks per language. We design a translator that extends an
existing single-language OOP benchmark and the pass@o metric to a multilingual
setting. Moreover, we propose an automated framework for augmenting test cases
to ensure the reliability of the evaluation results. We evaluate 14 mainstream
LLMs under zero-shot prompting and report three key findings: 1) Substantial
performance degradation: pass@1 scores on MultiOOP drop by up to 65.6
percentage points compared to function-level tasks (e.g., HumanEval). 2)
Cross-language variability: GPT-4o mini achieves pass@1 of 48.06% in Python but
only 0.12%-15.26% in other languages, indicating limited multilingual
generalization. 3) Conceptual gaps: pass@o scores are consistently 1.1-19.2
points lower than pass@k, demonstrating that LLMs often generate executable
code without fully capturing core OOP concepts. Our benchmark, metric
extensions, and evaluation scripts will be publicly released to foster a more
balanced and comprehensive assessment of LLMs in object-oriented code
generation. Our code and data will be released at
https://github.com/alphadl/OOP-eval and
https://huggingface.co/datasets/codeai-dteam/MultiOOP respectively.

### 8. [Understanding Collective Social Behavior in OSS Communities: A Co-editing Network Analysis of Activity Cascades](http://arxiv.org/pdf/2509.26173v1)

Authors: Lisi Qarkaxhija, Maximilian Carparo, Stefan Menzel, Bernhard Sendhoff, Ingo Scholtes

Understanding the collective social behavior of software developers is
crucial to model and predict the long-term dynamics and sustainability of Open
Source Software (OSS) communities. To this end, we analyze temporal activity
patterns of developers, revealing an inherently ``bursty'' nature of commit
contributions. To investigate the social mechanisms behind this phenomenon, we
adopt a network-based modelling framework that captures developer interactions
through co-editing networks. Our framework models social interactions, where a
developer editing the code of other developers triggers accelerated activity
among collaborators. Using a large data set on 50 major OSS communities, we
further develop a method that identifies activity cascades, i.e. the
propagation of developer activity in the underlying co-editing network. Our
results suggest that activity cascades are a statistically significant
phenomenon in more than half of the studied projects. We further show that our
insights can be used to develop a simple yet practical churn prediction method
that forecasts which developers are likely to leave a project. Our work sheds
light on the emergent collective social dynamics in OSS communities and
highlights the importance of activity cascades to understand developer churn
and retention in collaborative software projects.

### 9. [Hamster: A Large-Scale Study and Characterization of Developer-Written Tests](http://arxiv.org/pdf/2509.26204v1)

Authors: Rangeet Pan, Tyler Stennett, Raju Pavuluri, Nate Levin, Alessandro Orso, Saurabh Sinha

Automated test generation (ATG), which aims to reduce the cost of manual test
suite development, has been investigated for decades and has produced countless
techniques based on a variety of approaches: symbolic analysis, search-based,
random and adaptive-random, learning-based, and, most recently,
large-language-model-based approaches. However, despite this large body of
research, there is still a gap in our understanding of the characteristics of
developer-written tests and, consequently, in our assessment of how well ATG
techniques and tools can generate realistic and representative tests. To bridge
this gap, we conducted an extensive empirical study of developer-written tests
for Java applications, covering 1.7 million test cases from open-source
repositories. Our study is the first of its kind in studying aspects of
developer-written tests that are mostly neglected in the existing literature,
such as test scope, test fixtures and assertions, types of inputs, and use of
mocking. Based on the characterization, we then compare existing tests with
those generated by two state-of-the-art ATG tools. Our results highlight that a
vast majority of developer-written tests exhibit characteristics that are
beyond the capabilities of current ATG tools. Finally, based on the insights
gained from the study, we identify promising research directions that can help
bridge the gap between current tool capabilities and more effective tool
support for developer testing practices. We hope that this work can set the
stage for new advances in the field and bring ATG tools closer to generating
the types of tests developers write.

### 10. [UniSage: A Unified and Post-Analysis-Aware Sampling for Microservices](http://arxiv.org/pdf/2509.26336v1)

Authors: Zhouruixing Zhu, Zhihan Jiang, Tianyi Yang, Pinjia He

Traces and logs are essential for observability and fault diagnosis in modern
distributed systems. However, their ever-growing volume introduces substantial
storage overhead and complicates troubleshooting. Existing approaches typically
adopt a sample-before-analysis paradigm: even when guided by data heuristics,
they inevitably discard failure-related information and hinder transparency in
diagnosing system behavior. To address this, we introduce UniSage, the first
unified framework to sample both traces and logs using a post-analysis-aware
paradigm. Instead of discarding data upfront, UniSagefirst performs lightweight
and multi-modal anomaly detection and root cause analysis (RCA) on the complete
data stream. This process yields fine-grained, service-level diagnostic
insights that guide a dual-pillar sampling strategy for handling both normal
and anomalous scenarios: an analysis-guided sampler prioritizes data implicated
by RCA, while an edge-case-based sampler ensures rare but critical behaviors
are captured. Together, these pillars ensure comprehensive coverage of critical
signals without excessive redundancy. Extensive experiments demonstrate that
UniSage significantly outperforms state-of-the-art baselines. At a 2.5%
sampling rate, it captures 56.5% of critical traces and 96.25% of relevant
logs, while improving the accuracy (AC@1) of downstream root cause analysis by
42.45%. Furthermore, its efficient pipeline processes 10 minutes of telemetry
data in under 5 seconds, demonstrating its practicality for production
environments.

### Social and Information Networks

### 1. [HiFIRec: Towards High-Frequency yet Low-Intention Behaviors for Multi-Behavior Recommendation](http://arxiv.org/pdf/2509.25755v1)

Authors: Ruiqi Luo, Ran Jin, Zhenglong Li, Kaixi Hu, Xiaohui Tao, Lin Li

Multi-behavior recommendation leverages multiple types of user-item
interactions to address data sparsity and cold-start issues, providing
personalized services in domains such as healthcare and e-commerce. Most
existing methods utilize graph neural networks to model user intention in a
unified manner, which inadequately considers the heterogeneity across different
behaviors. Especially, high-frequency yet low-intention behaviors may
implicitly contain noisy signals, and frequent patterns that are plausible
while misleading, thereby hindering the learning of user intentions. To this
end, this paper proposes a novel multi-behavior recommendation method, HiFIRec,
that corrects the effect of high-frequency yet low-intention behaviors by
differential behavior modeling. To revise the noisy signals, we hierarchically
suppress it across layers by extracting neighborhood information through
layer-wise neighborhood aggregation and further capturing user intentions
through adaptive cross-layer feature fusion. To correct plausible frequent
patterns, we propose an intensity-aware non-sampling strategy that dynamically
adjusts the weights of negative samples. Extensive experiments on two
benchmarks show that HiFIRec relatively improves HR@10 by 4.21%-6.81% over
several state-of-the-art methods.

### 2. [Basic Cycle Ratio: Cost-Effective Ranking of Influential Spreaders from Local and Global Perspectives](http://arxiv.org/pdf/2509.26220v1)

Authors: Wenxin Zheng, Wenfeng Shi, Tianlong Fan, Linyuan Lv

Spreading processes are fundamental to complex networks. Identifying
influential spreaders with dual local and global roles presents a crucial yet
challenging task. To address this, our study proposes a novel method, the Basic
Cycle Ratio (BCR), for assessing node importance. BCR leverages basic cycles
and the cycle ratio to uniquely capture a node's local significance within its
immediate neighborhood and its global role in maintaining network cohesion. We
evaluated BCR on six diverse real-world social networks. Our method
outperformed traditional centrality measures and other cycle-based approaches,
proving more effective at selecting powerful spreaders and enhancing
information diffusion. Besides, BCR offers a cost-effective and practical
solution for social network applications.

### 3. [MHINDR -- a DSM5 based mental health diagnosis and recommendation framework using LLM](http://arxiv.org/pdf/2509.25992v1)

Authors: Vaishali Agarwal, Sachin Thukral, Arnab Chatterjee

Mental health forums offer valuable insights into psychological issues,
stressors, and potential solutions. We propose MHINDR, a large language model
(LLM) based framework integrated with DSM-5 criteria to analyze user-generated
text, dignose mental health conditions, and generate personalized interventions
and insights for mental health practitioners. Our approach emphasizes on the
extraction of temporal information for accurate diagnosis and symptom
progression tracking, together with psychological features to create
comprehensive mental health summaries of users. The framework delivers
scalable, customizable, and data-driven therapeutic recommendations, adaptable
to diverse clinical contexts, patient needs, and workplace well-being programs.

### Systems and Control

### 1. [Pinching-Antenna Systems (PASS)-Enabled UAV Delivery](http://arxiv.org/pdf/2509.25698v1)

Authors: Suyu Lv, Meng Li, Qi Li, Yuanwei Liu

A pinching-antenna systems (PASS)-enabled unmanned aerial vehicle (UAV)
delivery framework is proposed, which exploits the capability of PASS to
establish a strong line-of-sight link and reduce free-space pathloss.Aiming at
minimizing the communication energy consumption in one cycle, a double-layer
optimization (DLO) algorithm is developed by jointly optimizing the UAV
delivery sequence and the pinching antenna (PA) activation vector. More
specifically, at the outer layer, a hierarchical alternating optimization (HAO)
scheme is proposed to tackle the NP-hard problem of delivery sequence planning,
where a genetic algorithm performs global exploration to generate candidate
solutions at the top-level, while a dynamic programming performs local
refinement to obtain elite solutions at the lower-level. With determined UAV
trajectory, at the inner layer, focus is placed on addressing the highly
coupled mixed-integer nonlinear programming problem of PA activation vector
optimization, where a pair of algorithms are proposed: 1) Branch-and-Bound
(BnB) algorithm for finding global optimum; 2) incremental search and local
refinement (ISLR) algorithm for reducing computational complexity. Simulation
results indicate that: i) The proposed HAO-based delivery sequence planning
scheme can effectively reduce the total flight distance, thereby decreasing
flight time and communication energy consumption; ii) Both the proposed BnB and
ISLR algorithms can achieve energy-efficient PA activation, with the former
exhibiting better performance and the latter having lower complexity; iii) PASS
outperforms the conventional multi-antenna systems, especially with higher
communication rate requirements.

### 2. [Dynamic Causal Attack Graph based Cyber-security Risk Assessment Framework for CTCS System](http://arxiv.org/pdf/2509.25786v1)

Authors: Zikai Zhang

Protecting the security of the train control system is a critical issue to
ensure the safe and reliable operation of high-speed trains. Scientific
modeling and analysis for the security risk is a promising way to guarantee
system security. However, the representation and assessment of the
multi-staged, causally related, and temporal-dynamic changed attack
dependencies are difficult in the train control system. To solve the above
challenges, a security assessment framework based on the Dynamical Causality
Attack Graph (DCAG) model is introduced in this paper. Firstly, the DCAG model
is generated based on the attack graph with consideration of temporal attack
propagation and multi-stage attack event causality propagation. Then, the DCAG
model is analyzed based on Bayesian inference and logic gateway-based
inference. Through the case analysis of the CTCS-3 system, the security
assessment framework is validated. With the DCAG-based security assessment
framework, we can not only perform appropriate security risk quantification
calculations, but also explore the importance of different attacks on system
security risks, which is helpful in adjusting the cyber security defense
policy.

### 3. [Assessment of East-West (E-W) and South-North (S-N) facing Vertical Bifacial Photovoltaic Modules for Agrivoltaics and Dual-Land Use Applications in India](http://arxiv.org/pdf/2509.26396v1)

Authors: Nishant Kumar, Shravan Kumar Singh, Nikhil Chander

Deploying vertical bifacial PV modules can play a significant role in
agrivoltaics, fencing walls, noise barriers, building integrated photovoltaics
(BIPV), solar PV for electric vehicles, and many other applications. This
research work presents the performance comparison of vertical bifacial
photovoltaic (VBPV) modules facing East-West (E-W) and South-North (S-N)
directions. Also, the VBPV modules are compared with vertical and tilted
south-facing monofacial PV modules. Six PV modules (monofacial and bifacial)
were installed at the rooftop of IIT Bhilai academic building, Raipur
(21.16{\deg} N, 81.65{\deg} E), India, and studied for a year from May 2022 to
April 2023. The results show that the E-W facing VBPV module gives two
production peaks, one in the morning and another in the evening, as compared to
the single notable rise at midday observed for a monofacial module. From a
series of experiments, 19 days of data were collected over the one-year period
from May 2022 to April 2023, with specific inclusion of important days like
solstices and equinoxes. In addition, the energy generation results are
compared with PVsyst simulations, while also addressing the limitations of the
PVsyst simulation of vertical PV modules. E-W bifacial generation is higher
than S-N bifacial and south-facing monofacial modules from February to April.
The VBPV modules in E-W and S-N orientations present a promising opportunity
for expanding the agrivoltaics sector in tropical and sub-tropical countries,
like India. This has huge implications for addressing the sustainable
development goals by simultaneously contributing to sustainable land
management, green energy generation, energy security and water conservation in
the vast geo-climatic expanse of tropics.

### 4. [Anticipatory Structure in the Propagation of Signal](http://arxiv.org/pdf/2509.26481v1)

Authors: M. R. Sayeh, R. E. Auxier

We here report the development of a structure that shows the proteresis
phenomenon in more general setting and set out its philosophical implications.
In this case, the questions relate to how we are to interpret what will happen
in the future, and the procollection (the counterpart of recollection) of
not-yet-experienced phenomena that, when expressed, will be whatever has built
up in fully determinate form already, ahead of the event. If such a process
really exists, as our evidence confirms, not just as phenomenon but as a fact,
then a gap exists between the actualized form of the future and its concrete
expression when the event does happen. Such a fact, as hard to imagine as it
is, may be intelligible, even interpretable and susceptible to mathematical
and/or logical modeling. We build upon neglected theories and formulae that
present time in a way that makes our results interpretable. A proteretic device
is here described which shifts the input signal (event) to the future; and it
is an anticipatory structure. The proteretic characteristic of neurons should
also be capable of demonstration; and its neuronal behavior is possibly the
reason for the fast perception/thought processes in spite of slow behaving
neurons. That capacity may also account for why it is possible for animals
(including humans) to interact with the environment by slightly seeing (in the
sense of perceiving and/or sensing) the future. Exploiting this new proteretic
technology, faster computers and more efficient cellphones, among other things,
will be designed and built.

### 5. [Neural Network-based Co-design of Output-Feedback Control Barrier Function and Observer](http://arxiv.org/pdf/2509.26597v1)

Authors: Vaishnavi Jagabathula, Ahan Basu, Pushpak Jagtap

Control Barrier Functions (CBFs) provide a powerful framework for ensuring
safety in dynamical systems. However, their application typically relies on
full state information, which is often violated in real-world scenarios due to
the availability of partial state information. In this work, we propose a
neural network-based framework for the co-design of a safety controller,
observer, and CBF for partially observed continuous-time systems. By
formulating barrier conditions over an augmented state space, our approach
ensures safety without requiring bounded estimation errors or handcrafted
barrier functions. All components are jointly trained by formulating
appropriate loss functions, and we introduce a validity condition to provide
formal safety guarantees beyond the training data. Finally, we demonstrate the
effectiveness of the proposed approach through several case studies.

### 6. [Robust Safety-Critical Control of Integrator Chains with Mismatched Perturbations via Linear Time-Varying Feedback](http://arxiv.org/pdf/2509.26629v1)

Authors: Imtiaz Ur Rehman Moussa Labbadi, Amine Abadi, Lew Lew Yan Voon

In this paper, we propose a novel safety-critical control framework for a
chain of integrators subject to both matched and mismatched perturbations. The
core of our approach is a linear, time-varying state-feedback design that
simultaneously enforces stability and safety constraints. By integrating
backstepping techniques with a quadratic programming (QP) formulation, we
develop a systematic procedure to guarantee safety under time-varying gains. We
provide rigorous theoretical guarantees for the double integrator case, both in
the presence and absence of perturbations, and outline general proofs for
extending the methodology to higher-order chains of integrators. This proposed
framework thus bridges robustness and safety-critical performance, while
overcoming the limitations of existing prescribed-time approaches.

### 7. [Unsupervised Detection of Spatiotemporal Anomalies in PMU Data Using Transformer-Based BiGAN](http://arxiv.org/pdf/2509.25612v1)

Authors: Muhammad Imran Hossain, Jignesh Solanki, Sarika Khushlani Solanki

Ensuring power grid resilience requires the timely and unsupervised detection
of anomalies in synchrophasor data streams. We introduce T-BiGAN, a novel
framework that integrates window-attention Transformers within a bidirectional
Generative Adversarial Network (BiGAN) to address this challenge. Its
self-attention encoder-decoder architecture captures complex spatio-temporal
dependencies across the grid, while a joint discriminator enforces cycle
consistency to align the learned latent space with the true data distribution.
Anomalies are flagged in real-time using an adaptive score that combines
reconstruction error, latent space drift, and discriminator confidence.
Evaluated on a realistic hardware-in-the-loop PMU benchmark, T-BiGAN achieves
an ROC-AUC of 0.95 and an average precision of 0.996, significantly
outperforming leading supervised and unsupervised methods. It shows particular
strength in detecting subtle frequency and voltage deviations, demonstrating
its practical value for live, wide-area monitoring without relying on manually
labeled fault data.

### 8. [Policy Optimization in Robust Control: Weak Convexity and Subgradient Methods](http://arxiv.org/pdf/2509.25633v1)

Authors: Yuto Watanabe, Feng-Yi Liao, Yang Zheng

Robust control seeks stabilizing policies that perform reliably under
adversarial disturbances, with $\mathcal{H}_\infty$ control as a classical
formulation. It is known that policy optimization of robust
$\mathcal{H}_\infty$ control naturally lead to nonsmooth and nonconvex
problems. This paper builds on recent advances in nonsmooth optimization to
analyze discrete-time static output-feedback $\mathcal{H}_\infty$ control. We
show that the $\mathcal{H}_\infty$ cost is weakly convex over any convex subset
of a sublevel set. This structural property allows us to establish the first
non-asymptotic deterministic convergence rate for the subgradient method under
suitable assumptions. In addition, we prove a weak Polyak-{\L}ojasiewicz (PL)
inequality in the state-feedback case, implying that all stationary points are
globally optimal. We finally present a few numerical examples to validate the
theoretical results.

### 9. [Ingress Cryogenic Receivers Toward Scalable Quantum Information Processing: Theory and System Analysis](http://arxiv.org/pdf/2509.25768v1)

Authors: Malek Succar, Mohamed I. Ibrahim

Current control techniques for cryogenically cooled qubits are realized with
coaxial cables, posing multiple challenges in terms of cost, thermal load,
size, and long-term scalability. Emerging approaches to tackle this issue
include cryogenic CMOS electronics at 4 K, and photonic links for direct qubit
control. In this paper, we propose a multiplexed all-passive cryogenic high
frequency direct detection control platform (cryo-HFDD). The proposed classical
interface for direct qubit control utilizes optical or sub-THz bands. We
present the possible tradeoffs of this platform, and compare it with current
state-of-the-art cryogenic CMOS and conventional coaxial approaches. We assess
the feasibility of adopting these efficient links for a wide range of microwave
qubit power levels. Specifically, we estimate the heat load to achieve the
required signal-to-noise ratio SNR considering different noise sources,
component losses, as well as link density. We show that multiplexed photonic
receivers at 4 K can aggressively scale the control of thousands of qubits.
This opens the door for low cost scalable quantum computing systems.

### 10. [Intelligent Multi-link EDCA Optimization for Delay-Bounded QoS in Wi-Fi 7](http://arxiv.org/pdf/2509.25855v1)

Authors: Peini Yi, Wenchi Cheng, Jingqing Wang, Jinzhe Pan, Yuehui Ouyang, Wei Zhang

IEEE 802.11be (Wi-Fi 7) introduces Multi-Link Operation (MLO) as a While MLO
offers significant parallelism and capacity, realizing its full potential in
guaranteeing strict delay bounds and optimizing Quality of Service (QoS) for
diverse, heterogeneous traffic streams in complex multi-link scenarios remain a
significant challenge. This is largely due to the limitations of static
Enhanced Distributed Channel Access (EDCA) parameters and the complexity
inherent in cross-link traffic management. To address this, this paper
investigates the correlation between overall MLO QoS indicators and the
configuration of EDCA parameters and Acess Catagory (AC) traffic allocation
among links. Based on this analysis, we formulate a constrained optimization
problem aiming to minimize the sum of overall packet loss rates for all access
categories while satisfying their respective overall delay violation
probability constraints. A Genetic Algorithm (GA)-based MLO EDCA QoS
optimization algorithm is designed to efficiently search the complex
configuration space of AC assignments and EDCA parameters. Experimental results
demonstrate that the proposed approach's efficacy in generating adaptive MLO
configuration strategies that align with diverse service requirements. The
proposed solution significantly improves delay distribution characteristics,
and enhance QoS robustness and resource utilization efficiency in high-load MLO
environments.

### Machine Learning (Statistics Category)

### 1. [Transformers through the lens of support-preserving maps between measures](http://arxiv.org/pdf/2509.25611v1)

Authors: Takashi Furuya, Maarten V. de Hoop, Matti Lassas

Transformers are deep architectures that define ``in-context maps'' which
enable predicting new tokens based on a given set of tokens (such as a prompt
in NLP applications or a set of patches for a vision transformer). In previous
work, we studied the ability of these architectures to handle an arbitrarily
large number of context tokens. To mathematically, uniformly analyze their
expressivity, we considered the case that the mappings are conditioned on a
context represented by a probability distribution which becomes discrete for a
finite number of tokens. Modeling neural networks as maps on probability
measures has multiple applications, such as studying Wasserstein regularity,
proving generalization bounds and doing a mean-field limit analysis of the
dynamics of interacting particles as they go through the network. In this work,
we study the question what kind of maps between measures are transformers. We
fully characterize the properties of maps between measures that enable these to
be represented in terms of in-context maps via a push forward. On the one hand,
these include transformers; on the other hand, transformers universally
approximate representations with any continuous in-context map. These
properties are preserving the cardinality of support and that the regular part
of their Fr\'{e}chet derivative is uniformly continuous. Moreover, we show that
the solution map of the Vlasov equation, which is of nonlocal transport type,
for interacting particle systems in the mean-field regime for the Cauchy
problem satisfies the conditions on the one hand and, hence, can be
approximated by a transformer; on the other hand, we prove that the
measure-theoretic self-attention has the properties that ensure that the
infinite depth, mean-field measure-theoretic transformer can be identified with
a Vlasov flow.

### 2. [Test time training enhances in-context learning of nonlinear functions](http://arxiv.org/pdf/2509.25741v1)

Authors: Kento Kuwataka, Taiji Suzuki

Test-time training (TTT) enhances model performance by explicitly updating
designated parameters prior to each prediction to adapt to the test data. While
TTT has demonstrated considerable empirical success, its theoretical
underpinnings remain limited, particularly for nonlinear models. In this paper,
we investigate the combination of TTT with in-context learning (ICL), where the
model is given a few examples from the target distribution at inference time.
We analyze this framework in the setting of single-index models
$y=\sigma_*(\langle \beta, \mathbf{x} \rangle)$, where the feature vector
$\beta$ is drawn from a hidden low-dimensional subspace. For single-layer
transformers trained with gradient-based algorithms and adopting TTT, we
establish an upper bound on the prediction risk. Our theory reveals that TTT
enables the single-layer transformers to adapt to both the feature vector
$\beta$ and the link function $\sigma_*$, which vary across tasks. This creates
a sharp contrast with ICL alone, which is theoretically difficult to adapt to
shifts in the link function. Moreover, we provide the convergence rate with
respect to the data length, showing the predictive error can be driven
arbitrarily close to the noise level as the context size and the network width
grow.

### 3. [Online Decision Making with Generative Action Sets](http://arxiv.org/pdf/2509.25777v1)

Authors: Jianyu Xu, Vidhi Jain, Bryan Wilder, Aarti Singh

With advances in generative AI, decision-making agents can now dynamically
create new actions during online learning, but action generation typically
incurs costs that must be balanced against potential benefits. We study an
online learning problem where an agent can generate new actions at any time
step by paying a one-time cost, with these actions becoming permanently
available for future use. The challenge lies in learning the optimal sequence
of two-fold decisions: which action to take and when to generate new ones,
further complicated by the triangular tradeoffs among exploitation, exploration
and $\textit{creation}$. To solve this problem, we propose a doubly-optimistic
algorithm that employs Lower Confidence Bounds (LCB) for action selection and
Upper Confidence Bounds (UCB) for action generation. Empirical evaluation on
healthcare question-answering datasets demonstrates that our approach achieves
favorable generation-quality tradeoffs compared to baseline strategies. From
theoretical perspectives, we prove that our algorithm achieves the optimal
regret of $O(T^{\frac{d}{d+2}}d^{\frac{d}{d+2}} + d\sqrt{T\log T})$, providing
the first sublinear regret bound for online learning with expanding action
spaces.

### 4. [Graph Distribution-valued Signals: A Wasserstein Space Perspective](http://arxiv.org/pdf/2509.25802v1)

Authors: Yanan Zhao, Feng Ji, Xingchao Jian, Wee Peng Tay

We introduce a novel framework for graph signal processing (GSP) that models
signals as graph distribution-valued signals (GDSs), which are probability
distributions in the Wasserstein space. This approach overcomes key limitations
of classical vector-based GSP, including the assumption of synchronous
observations over vertices, the inability to capture uncertainty, and the
requirement for strict correspondence in graph filtering. By representing
signals as distributions, GDSs naturally encode uncertainty and stochasticity,
while strictly generalizing traditional graph signals. We establish a
systematic dictionary mapping core GSP concepts to their GDS counterparts,
demonstrating that classical definitions are recovered as special cases. The
effectiveness of the framework is validated through graph filter learning for
prediction tasks, supported by experimental results.

### 5. [Decentralized Asynchronous Multi-player Bandits](http://arxiv.org/pdf/2509.25824v1)

Authors: Jingqi Fan, Canzhe Zhao, Shuai Li, Siwei Wang

In recent years, multi-player multi-armed bandits (MP-MAB) have been
extensively studied due to their wide applications in cognitive radio networks
and Internet of Things systems. While most existing research on MP-MAB focuses
on synchronized settings, real-world systems are often decentralized and
asynchronous, where players may enter or leave the system at arbitrary times,
and do not have a global clock. This decentralized asynchronous setting
introduces two major challenges. First, without a global time, players cannot
implicitly coordinate their actions through time, making it difficult to avoid
collisions. Second, it is important to detect how many players are in the
system, but doing so may cost a lot. In this paper, we address the challenges
posed by such a fully asynchronous setting in a decentralized environment. We
develop a novel algorithm in which players adaptively change between
exploration and exploitation. During exploration, players uniformly pull their
arms, reducing the probability of collisions and effectively mitigating the
first challenge. Meanwhile, players continue pulling arms currently exploited
by others with a small probability, enabling them to detect when a player has
left, thereby addressing the second challenge. We prove that our algorithm
achieves a regret of $\mathcal{O}(\sqrt{T \log T} + {\log T}/{\Delta^2})$,
where $\Delta$ is the minimum expected reward gap between any two arms. To the
best of our knowledge, this is the first efficient MP-MAB algorithm in the
asynchronous and decentralized environment. Extensive experiments further
validate the effectiveness and robustness of our algorithm, demonstrating its
applicability to real-world scenarios.

### 6. [Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access](http://arxiv.org/pdf/2509.26000v1)

Authors: Daniel Ebi, Gaspard Lambrechts, Damien Ernst, Klemens Böhm

Reinforcement learning in partially observable environments requires agents
to act under uncertainty from noisy, incomplete observations. Asymmetric
actor-critic methods leverage privileged information during training to improve
learning under these conditions. However, existing approaches typically assume
full-state access during training. In this work, we challenge this assumption
by proposing a novel actor-critic framework, called informed asymmetric
actor-critic, that enables conditioning the critic on arbitrary privileged
signals without requiring access to the full state. We show that policy
gradients remain unbiased under this formulation, extending the theoretical
foundation of asymmetric methods to the more general case of privileged partial
information. To quantify the impact of such signals, we propose informativeness
measures based on kernel methods and return prediction error, providing
practical tools for evaluating training-time signals. We validate our approach
empirically on benchmark navigation tasks and synthetic partially observable
environments, showing that our informed asymmetric method improves learning
efficiency and value estimation when informative privileged inputs are
available. Our findings challenge the necessity of full-state access and open
new directions for designing asymmetric reinforcement learning methods that are
both practical and theoretically sound.

### 7. [BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields](http://arxiv.org/pdf/2509.26005v1)

Authors: Rui-Yang Zhang, Henry B. Moss, Lachlan Astfalck, Edward Cripps, David S. Leslie

We introduce a formal active learning methodology for guiding the placement
of Lagrangian observers to infer time-dependent vector fields -- a key task in
oceanography, marine science, and ocean engineering -- using a physics-informed
spatio-temporal Gaussian process surrogate model. The majority of existing
placement campaigns either follow standard `space-filling' designs or
relatively ad-hoc expert opinions. A key challenge to applying principled
active learning in this setting is that Lagrangian observers are continuously
advected through the vector field, so they make measurements at different
locations and times. It is, therefore, important to consider the likely future
trajectories of placed observers to account for the utility of candidate
placement locations. To this end, we present BALLAST: Bayesian Active Learning
with Look-ahead Amendment for Sea-drifter Trajectories. We observe noticeable
benefits of BALLAST-aided sequential observer placement strategies on both
synthetic and high-fidelity ocean current models.

### 8. [Non-Vacuous Generalization Bounds: Can Rescaling Invariances Help?](http://arxiv.org/pdf/2509.26149v1)

Authors: Damien Rouchouse, Antoine Gonon, Rémi Gribonval, Benjamin Guedj

A central challenge in understanding generalization is to obtain non-vacuous
guarantees that go beyond worst-case complexity over data or weight space.
Among existing approaches, PAC-Bayes bounds stand out as they can provide
tight, data-dependent guarantees even for large networks. However, in ReLU
networks, rescaling invariances mean that different weight distributions can
represent the same function while leading to arbitrarily different PAC-Bayes
complexities. We propose to study PAC-Bayes bounds in an invariant, lifted
representation that resolves this discrepancy. This paper explores both the
guarantees provided by this approach (invariance, tighter bounds via data
processing) and the algorithmic aspects of KL-based rescaling-invariant
PAC-Bayes bounds.

### 9. [Staged Event Trees for Transparent Treatment Effect Estimation](http://arxiv.org/pdf/2509.26265v1)

Authors: Gherardo Varando, Manuele Leonelli, Jordi Cerdà-Bautista, Vasileios Sitokonstantinou, Gustau Camps-Valls

Average and conditional treatment effects are fundamental causal quantities
used to evaluate the effectiveness of treatments in various critical
applications, including clinical settings and policy-making. Beyond the
gold-standard estimators from randomized trials, numerous methods have been
proposed to estimate treatment effects using observational data. In this paper,
we provide a novel characterization of widely used causal inference techniques
within the framework of staged event trees, demonstrating their capacity to
enhance treatment effect estimation. These models offer a distinct advantage
due to their interpretability, making them particularly valuable for practical
applications. We implement classical estimators within the framework of staged
event trees and illustrate their capabilities through both simulation studies
and real-world applications. Furthermore, we showcase how staged event trees
explicitly and visually describe when standard causal assumptions, such as
positivity, hold, further enhancing their practical utility.

### 10. [ACE: Adapting sampling for Counterfactual Explanations](http://arxiv.org/pdf/2509.26322v1)

Authors: Margarita A. Guerrero, Cristian R. Rojas

Counterfactual Explanations (CFEs) interpret machine learning models by
identifying the smallest change to input features needed to change the model's
prediction to a desired output. For classification tasks, CFEs determine how
close a given sample is to the decision boundary of a trained classifier.
Existing methods are often sample-inefficient, requiring numerous evaluations
of a black-box model -- an approach that is both costly and impractical when
access to the model is limited. We propose Adaptive sampling for Counterfactual
Explanations (ACE), a sample-efficient algorithm combining Bayesian estimation
and stochastic optimization to approximate the decision boundary with fewer
queries. By prioritizing informative points, ACE minimizes evaluations while
generating accurate and feasible CFEs. Extensive empirical results show that
ACE achieves superior evaluation efficiency compared to state-of-the-art
methods, while maintaining effectiveness in identifying minimal and actionable
changes.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-01 PST.

### 1. [AI is dreaming up millions of new materials. Are they any good?](https://www.nature.com/articles/d41586-025-03147-9)

Authors: Mark Peplow

### 2. [Fast, slow, and metacognitive thinking in AI](https://www.nature.com/articles/s44387-025-00027-5)

Authors: M. Bergamaschi Ganapini et al.

### 3. [A hybrid deep learning model for detection and mitigation of DDoS attacks in VANETs](https://www.nature.com/articles/s41598-025-15215-1)

Authors: Naramalli Jayakrishna et al.

### 4. [Inverse design of periodic cavities in anechoic coatings with gradient changes of radii and distances via a conditional generative adversarial network](https://www.nature.com/articles/s41598-025-15946-1)

Authors: Yiping Sun et al.

### 5. [A dynamic fractional generalized deterministic annealing for rapid convergence in deep learning optimization](https://www.nature.com/articles/s44387-025-00025-7)

Authors: Matthew Korban et al.

### 6. [Unifying machine learning and interpolation theory via interpolating neural networks](https://www.nature.com/articles/s41467-025-63790-8)

Authors: Chanwook Park et al.

### 7. [Quantum-Inspired gravitationally guided particle swarm optimization for feature selection and classification](https://www.nature.com/articles/s41598-025-14793-4)

Authors: Saleem Malik et al.

### 8. [Building adaptive knowledge bases for evolving continual learning models](https://www.nature.com/articles/s44387-025-00028-4)

Authors: Jack Julian et al.

### 9. [Children with and without reading difficulty value robot reading companions that are smart, supportive, and personalised](https://www.nature.com/articles/s41598-025-15341-w)

Authors: Ryssa Moffat et al.

### 10. [An intelligent fuzzy-neural framework for autism sensory assessment using hierarchical linguistic modeling and risk-based temporal decision-making](https://www.nature.com/articles/s41598-025-15730-1)

Authors: Nabilah Abughazalah et al.

### 11. [Pre-trained molecular language models with random functional group masking](https://www.nature.com/articles/s44387-025-00029-3)

Authors: Tianhao Peng et al.

### 12. [Data-driven uncertainty-aware forecasting of sea ice conditions in the gulf of Ob based on satellite radar imagery](https://www.nature.com/articles/s41598-025-16572-7)

Authors: Stefan Maria Ailuro et al.

### 13. [An interpretable hybrid deep learning framework for gastric cancer diagnosis using histopathological imaging](https://www.nature.com/articles/s41598-025-15702-5)

Authors: Tengfei Ren et al.

### 14. [Increasing the clock speed of a thermodynamic computer by adding noise](https://www.nature.com/articles/s44335-025-00038-0)

Authors: Stephen Whitelam

### 15. [Comparative analysis of algorithmic approaches in ensemble learning: bagging vs. boosting](https://www.nature.com/articles/s41598-025-15971-0)

Authors: Hongke Zhao et al.

