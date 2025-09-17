# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-16 17:00:26.691004 PST.

### Artificial Intelligence

### 1. [MedicalOS: An LLM Agent based Operating System for Digital Healthcare](http://arxiv.org/pdf/2509.11507v1)

Authors: Jared Zhu, Junde Wu

Decades' advances in digital health technologies, such as electronic health
records, have largely streamlined routine clinical processes. Yet, most these
systems are still hard to learn and use: Clinicians often face the burden of
managing multiple tools, repeating manual actions for each patient, navigating
complicated UI trees to locate functions, and spending significant time on
administration instead of caring for patients. The recent rise of large
language model (LLM) based agents demonstrates exceptional capability in coding
and computer operation, revealing the potential for humans to interact with
operating systems and software not by direct manipulation, but by instructing
agents through natural language. This shift highlights the need for an
abstraction layer, an agent-computer interface, that translates human language
into machine-executable commands. In digital healthcare, however, requires a
more domain-specific abstractions that strictly follow trusted clinical
guidelines and procedural standards to ensure safety, transparency, and
compliance. To address this need, we present \textbf{MedicalOS}, a unified
agent-based operational system designed as such a domain-specific abstract
layer for healthcare. It translates human instructions into pre-defined digital
healthcare commands, such as patient inquiry, history retrieval, exam
management, report generation, referrals, treatment planning, that we wrapped
as off-the-shelf tools using machine languages (e.g., Python, APIs, MCP,
Linux). We empirically validate MedicalOS on 214 patient cases across 22
specialties, demonstrating high diagnostic accuracy and confidence, clinically
sound examination requests, and consistent generation of structured reports and
medication recommendations. These results highlight MedicalOS as a trustworthy
and scalable foundation for advancing workflow automation in clinical practice.

### 2. [Task Decoding based on Eye Movements using Synthetic Data Augmentation](http://arxiv.org/pdf/2509.11547v1)

Authors: Shanmuka Sadhu, Arca Baran, Preeti Pandey, Ayush Kumar

Machine learning has been extensively used in various applications related to
eye-tracking research. Understanding eye movement is one of the most
significant subsets of eye-tracking research that reveals the scanning pattern
of an individual. Researchers have thoroughly analyzed eye movement data to
understand various eye-tracking applications, such as attention mechanisms,
navigational behavior, task understanding, etc. The outcome of traditional
machine learning algorithms used for decoding tasks based on eye movement data
has received a mixed reaction to Yarbus' claim that it is possible to decode
the observer's task from their eye movements. In this paper, to support the
hypothesis by Yarbus, we are decoding tasks categories while generating
synthetic data samples using well-known Synthetic Data Generators CTGAN and its
variations such as CopulaGAN and Gretel AI Synthetic Data generators on
available data from an in-person user study. Our results show that augmenting
more eye movement data combined with additional synthetically generated
improves classification accuracy even with traditional machine learning
algorithms. We see a significant improvement in task decoding accuracy from
28.1% using Random Forest to 82% using Inception Time when five times more data
is added in addition to the 320 real eye movement dataset sample. Our proposed
framework outperforms all the available studies on this dataset because of the
use of additional synthetic datasets. We validated our claim with various
algorithms and combinations of real and synthetic data to show how decoding
accuracy increases with the increase in the augmentation of generated data to
real data.

### 3. [A Survey of Reasoning and Agentic Systems in Time Series with Large Language Models](http://arxiv.org/pdf/2509.11575v1)

Authors: Ching Chang, Yidan Shi, Defu Cao, Wei Yang, Jeehyun Hwang, Haixin Wang, Jiacheng Pang, Wei Wang, Yan Liu, Wen-Chih Peng, Tien-Fu Chen

Time series reasoning treats time as a first-class axis and incorporates
intermediate evidence directly into the answer. This survey defines the problem
and organizes the literature by reasoning topology with three families: direct
reasoning in one step, linear chain reasoning with explicit intermediates, and
branch-structured reasoning that explores, revises, and aggregates. The
topology is crossed with the main objectives of the field, including
traditional time series analysis, explanation and understanding, causal
inference and decision making, and time series generation, while a compact tag
set spans these axes and captures decomposition and verification, ensembling,
tool use, knowledge access, multimodality, agent loops, and LLM alignment
regimes. Methods and systems are reviewed across domains, showing what each
topology enables and where it breaks down in faithfulness or robustness, along
with curated datasets, benchmarks, and resources that support study and
deployment (https://github.com/blacksnail789521/Time-Series-Reasoning-Survey).
Evaluation practices that keep evidence visible and temporally aligned are
highlighted, and guidance is distilled on matching topology to uncertainty,
grounding with observable artifacts, planning for shift and streaming, and
treating cost and latency as design budgets. We emphasize that reasoning
structures must balance capacity for grounding and self-correction against
computational cost and reproducibility, while future progress will likely
depend on benchmarks that tie reasoning quality to utility and on closed-loop
testbeds that trade off cost and risk under shift-aware, streaming, and
long-horizon settings. Taken together, these directions mark a shift from
narrow accuracy toward reliability at scale, enabling systems that not only
analyze but also understand, explain, and act on dynamic worlds with traceable
evidence and credible outcomes.

### 4. [Adapting and Evaluating Multimodal Large Language Models for Adolescent Idiopathic Scoliosis Self-Management: A Divide and Conquer Framework](http://arxiv.org/pdf/2509.11645v1)

Authors: Zhaolong Wu, Pu Luo, Jason Pui Yin Cheung, Teng Zhang

This study presents the first comprehensive evaluation of Multimodal Large
Language Models (MLLMs) for Adolescent Idiopathic Scoliosis (AIS)
self-management. We constructed a database of approximately 3,000
anteroposterior X-rays with diagnostic texts and evaluated five MLLMs through a
`Divide and Conquer' framework consisting of a visual question-answering task,
a domain knowledge assessment task, and a patient education counseling
assessment task. Our investigation revealed limitations of MLLMs' ability in
interpreting complex spinal radiographs and comprehending AIS care knowledge.
To address these, we pioneered enhancing MLLMs with spinal keypoint prompting
and compiled an AIS knowledge base for retrieval augmented generation (RAG),
respectively. Results showed varying effectiveness of visual prompting across
different architectures, while RAG substantially improved models' performances
on the knowledge assessment task. Our findings indicate current MLLMs are far
from capable in realizing personalized assistant in AIS care. The greatest
challenge lies in their abilities to obtain accurate detections of spinal
deformity locations (best accuracy: 0.55) and directions (best accuracy: 0.13).

### 5. [HeLoFusion: An Efficient and Scalable Encoder for Modeling Heterogeneous and Multi-Scale Interactions in Trajectory Prediction](http://arxiv.org/pdf/2509.11719v1)

Authors: Bingqing Wei, Lianmin Chen, Zhongyu Xia, Yongtao Wang

Multi-agent trajectory prediction in autonomous driving requires a
comprehensive understanding of complex social dynamics. Existing methods,
however, often struggle to capture the full richness of these dynamics,
particularly the co-existence of multi-scale interactions and the diverse
behaviors of heterogeneous agents. To address these challenges, this paper
introduces HeLoFusion, an efficient and scalable encoder for modeling
heterogeneous and multi-scale agent interactions. Instead of relying on global
context, HeLoFusion constructs local, multi-scale graphs centered on each
agent, allowing it to effectively model both direct pairwise dependencies and
complex group-wise interactions (\textit{e.g.}, platooning vehicles or
pedestrian crowds). Furthermore, HeLoFusion tackles the critical challenge of
agent heterogeneity through an aggregation-decomposition message-passing scheme
and type-specific feature networks, enabling it to learn nuanced,
type-dependent interaction patterns. This locality-focused approach enables a
principled representation of multi-level social context, yielding powerful and
expressive agent embeddings. On the challenging Waymo Open Motion Dataset,
HeLoFusion achieves state-of-the-art performance, setting new benchmarks for
key metrics including Soft mAP and minADE. Our work demonstrates that a
locality-grounded architecture, which explicitly models multi-scale and
heterogeneous interactions, is a highly effective strategy for advancing motion
forecasting.

### 6. [EgoMem: Lifelong Memory Agent for Full-duplex Omnimodal Models](http://arxiv.org/pdf/2509.11914v1)

Authors: Yiqun Yao, Naitong Yu, Xiang Li, Xin Jiang, Xuezhi Fang, Wenjia Ma, Xuying Meng, Jing Li, Aixin Sun, Yequan Wang

We introduce EgoMem, the first lifelong memory agent tailored for full-duplex
models that process real-time omnimodal streams. EgoMem enables real-time
models to recognize multiple users directly from raw audiovisual streams, to
provide personalized response, and to maintain long-term knowledge of users'
facts, preferences, and social relationships extracted from audiovisual
history. EgoMem operates with three asynchronous processes: (i) a retrieval
process that dynamically identifies user via face and voice, and gathers
relevant context from a long-term memory; (ii) an omnimodal dialog process that
generates personalized audio responses based on the retrieved context; and
(iii) a memory management process that automatically detects dialog boundaries
from omnimodal streams, and extracts necessary information to update the
long-term memory. Unlike existing memory agents for LLMs, EgoMem relies
entirely on raw audiovisual streams, making it especially suitable for
lifelong, real-time, and embodied scenarios. Experimental results demonstrate
that EgoMem's retrieval and memory management modules achieve over 95% accuracy
on the test set. When integrated with a fine-tuned RoboEgo omnimodal chatbot,
the system achieves fact-consistency scores above 87% in real-time personalized
dialogs, establishing a strong baseline for future research.

### 7. [BuildingGym: An open-source toolbox for AI-based building energy management using reinforcement learning](http://arxiv.org/pdf/2509.11922v1)

Authors: Xilei Dai, Ruotian Chen, Songze Guan, Wen-Tai Li, Chau Yuen

Reinforcement learning (RL) has proven effective for AI-based building energy
management. However, there is a lack of flexible framework to implement RL
across various control problems in building energy management. To address this
gap, we propose BuildingGym, an open-source tool designed as a
research-friendly and flexible framework for training RL control strategies for
common challenges in building energy management. BuildingGym integrates
EnergyPlus as its core simulator, making it suitable for both system-level and
room-level control. Additionally, BuildingGym is able to accept external
signals as control inputs instead of taking the building as a stand-alone
entity. This feature makes BuildingGym applicable for more flexible
environments, e.g. smart grid and EVs community. The tool provides several
built-in RL algorithms for control strategy training, simplifying the process
for building managers to obtain optimal control strategies. Users can achieve
this by following a few straightforward steps to configure BuildingGym for
optimization control for common problems in the building energy management
field. Moreover, AI specialists can easily implement and test state-of-the-art
control algorithms within the platform. BuildingGym bridges the gap between
building managers and AI specialists by allowing for the easy configuration and
replacement of RL algorithms, simulators, and control environments or problems.
With BuildingGym, we efficiently set up training tasks for cooling load
management, targeting both constant and dynamic cooling load management. The
built-in algorithms demonstrated strong performance across both tasks,
highlighting the effectiveness of BuildingGym in optimizing cooling strategies.

### 8. [Neuromorphic Intelligence](http://arxiv.org/pdf/2509.11940v1)

Authors: Marcel van Gerven

Neuromorphic computing seeks to replicate the remarkable efficiency,
flexibility, and adaptability of the human brain in artificial systems. Unlike
conventional digital approaches, which depend on massive computational and
energy resources, neuromorphic systems exploit brain-inspired principles of
computation to achieve orders of magnitude greater energy efficiency. By
drawing on insights from artificial intelligence, neuroscience, physics,
chemistry, and materials science, neuromorphic computing promises to deliver
intelligent systems that are sustainable, transparent, and widely accessible. A
central challenge, however, is to identify a unifying theoretical framework
capable of bridging these diverse disciplines. We argue that dynamical systems
theory provides such a foundation. Rooted in differential calculus, it offers a
principled language for modeling inference, learning, and control in both
natural and artificial substrates. Within this framework, noise can be
harnessed as a resource for learning, while differential genetic programming
enables the discovery of dynamical systems that implement adaptive behaviors.
Embracing this perspective paves the way toward emergent neuromorphic
intelligence, where intelligent behavior arises from the dynamics of physical
substrates, advancing both the science and sustainability of AI.

### 9. [Agentic Temporal Graph of Reasoning with Multimodal Language Models: A Potential AI Aid to Healthcare](http://arxiv.org/pdf/2509.11944v1)

Authors: Susanta Mitra

Healthcare and medicine are multimodal disciplines that deal with multimodal
data for reasoning and diagnosing multiple diseases. Although some multimodal
reasoning models have emerged for reasoning complex tasks in scientific
domains, their applications in the healthcare domain remain limited and fall
short in correct reasoning for diagnosis. To address the challenges of
multimodal medical reasoning for correct diagnosis and assist the healthcare
professionals, a novel temporal graph-based reasoning process modelled through
a directed graph has been proposed in the current work. It helps in
accommodating dynamic changes in reasons through backtracking, refining the
reasoning content, and creating new or deleting existing reasons to reach the
best recommendation or answer. Again, consideration of multimodal data at
different time points can enable tracking and analysis of patient health and
disease progression. Moreover, the proposed multi-agent temporal reasoning
framework provides task distributions and a cross-validation mechanism to
further enhance the accuracy of reasoning outputs. A few basic experiments and
analysis results justify the novelty and practical utility of the proposed
preliminary approach.

### 10. [Human-AI Use Patterns for Decision-Making in Disaster Scenarios: A Systematic Review](http://arxiv.org/pdf/2509.12034v1)

Authors: Emmanuel Adjei Domfeh, Christopher L. Dancy

In high-stakes disaster scenarios, timely and informed decision-making is
critical yet often challenged by uncertainty, dynamic environments, and limited
resources. This paper presents a systematic review of Human-AI collaboration
patterns that support decision-making across all disaster management phases.
Drawing from 51 peer-reviewed studies, we identify four major categories:
Human-AI Decision Support Systems, Task and Resource Coordination, Trust and
Transparency, and Simulation and Training. Within these, we analyze
sub-patterns such as cognitive-augmented intelligence, multi-agent
coordination, explainable AI, and virtual training environments. Our review
highlights how AI systems may enhance situational awareness, improves response
efficiency, and support complex decision-making, while also surfacing critical
limitations in scalability, interpretability, and system interoperability. We
conclude by outlining key challenges and future research directions,
emphasizing the need for adaptive, trustworthy, and context-aware Human-AI
systems to improve disaster resilience and equitable recovery outcomes.

### Hardware Architecture

### 1. [always_comm: An FPGA-based Hardware Accelerator for Audio/Video Compression and Transmission](http://arxiv.org/pdf/2509.11503v1)

Authors: Rishab Parthasarathy, Akshay Attaluri, Gilford Ting

We present a design for an extensible video conferencing stack implemented
entirely in hardware on a Nexys4 DDR FPGA, which uses the M-JPEG codec to
compress video and a UDP networking stack to communicate between the FPGA and
the receiving computer. This networking stack accepts real-time updates from
both the video codec and the audio controller, which means that video will be
able to be streamed at 30 FPS from the FPGA to a computer. On the computer
side, a Python script reads the Ethernet packets and decodes the packets into
the video and the audio for real time playback. We evaluate this architecture
using both functional, simulation-driven verification in Cocotb and by
synthesizing SystemVerilog RTL code using Vivado for deployment on our Nexys4
DDR FPGA, where we evaluate both end-to-end latency and throughput of video
transmission.

### 2. [SuperUROP: An FPGA-Based Spatial Accelerator for Sparse Matrix Operations](http://arxiv.org/pdf/2509.11529v1)

Authors: Rishab Parthasarathy

Solving sparse systems of linear equations is a fundamental problem in the
field of numerical methods, with applications spanning from circuit design to
urban planning. These problems can have millions of constraints, such as when
laying out transistors on a circuit, or trying to optimize traffic light
timings, making fast sparse solvers extremely important. However, existing
state-of-the-art software-level solutions for solving sparse linear systems,
termed iterative solvers, are extremely inefficient on current hardware. This
inefficiency can be attributed to two key reasons: (1) poor short-term data
reuse, which causes frequent, irregular memory accesses, and (2) complex data
dependencies, which limit parallelism. Hence, in this paper, we present an FPGA
implementation of the existing Azul accelerator, an SRAM-only hardware
accelerator that achieves both high memory bandwidth utilization and arithmetic
intensity. Azul features a grid of tiles, each of which is composed of a
processing element (PE) and a small independent SRAM memory, which are all
connected over a network on chip (NoC). We implement Azul on FPGA using simple
RISC-V CPU cores connected to a memory hierarchy of different FPGA memory
modules. We utilize custom RISC-V ISA augmentations to implement a task-based
programming model for the various PEs, allowing communication over the NoC.
Finally, we design simple distributed test cases so that we can functionally
verify the FPGA implementation, verifying equivalent performance to an
architectural simulation of the Azul framework.

### 3. [Vital Signs Monitoring with mmWave OFDM JCAS System](http://arxiv.org/pdf/2509.11767v1)

Authors: Jakub Dobosz, Maximilian Engelhardt, Diego Dupleich, Maciej Stapor, Pawel Kulakowski

Wireless techniques for monitoring human vital signs, such as heart and
breathing rates, offer a promising solution in the context of joint
communication and sensing (JCAS) with applications in medicine, sports, safety,
security, and even the military. This paper reports experimental results
obtained at the Fraunhofer Institute for Integrated Circuits in Ilmenau,
demonstrating the effectiveness of an indoor orthogonal frequency-division
multiplexing (OFDM) JCAS system for detecting human heart and breathing rates.
The system operated in a bistatic configuration at an FR2 frequency of 26.5 GHz
with a variable bandwidth of up to 1 GHz. Measurements were taken under various
scenarios, including a subject lying down, sitting, or walking, in both
line-of-sight and non-line-of-sight conditions, and with one or two subjects
present simultaneously. The results indicate that while vital sign detection is
generally feasible, its effectiveness is influenced by several factors, such as
the subjects clothing, activity, as well as the distance and angle relative to
the sensing system. In addition, no significant influence of bandwidth was
detected since the vital signs information is encoded in the phase of the
signal.

### 4. [LEGO: Spatial Accelerator Generation and Optimization for Tensor Applications](http://arxiv.org/pdf/2509.12053v1)

Authors: Yujun Lin, Zhekai Zhang, Song Han

Modern tensor applications, especially foundation models and generative AI
applications require multiple input modalities (both vision and language),
which increases the demand for flexible accelerator architecture. Existing
frameworks suffer from the trade-off between design flexibility and
productivity of RTL generation: either limited to very few hand-written
templates or cannot automatically generate the RTL. To address this challenge,
we propose the LEGO framework, which targets tensor applications and
automatically generates spatial architecture design and outputs synthesizable
RTL code without handwritten RTL design templates. Leveraging the
affine-transformation-based architecture representation, LEGO front end finds
interconnections between function units, synthesizes the memory system, and
fuses different spatial dataflow designs based on data reuse analysis. LEGO
back end then translates the hardware in a primitive-level graph to perform
lower-level optimizations, and applies a set of linear-programming algorithms
to optimally insert pipeline registers and reduce the overhead of unused logic
when switching spatial dataflows. Our evaluation demonstrates that LEGO can
achieve 3.2x speedup and 2.4x energy efficiency compared to previous work
Gemmini, and can generate one architecture for diverse modern foundation models
in generative AI applications.

### Computational Engineering

### 1. [Toward lean industry 5.0: a human-centered model for integrating lean and industry 4.0 in an automotive supplier](http://arxiv.org/pdf/2509.11658v1)

Authors: Peter Hines, Florian Magnani, Josefa Mula, Raquel Sanchis

This paper proposes a human-centered conceptual model integrating lean and
Industry 4.0 based on the literature review and validated it through a case
study in the context of an advanced automotive first-tier supplier. Addressing
a significant gap in existing research on lean Industry 4.0 implementations,
the study provides both theoretical insights and practical findings. It
emphasizes the importance of a human-centered approach, identifies key enablers
and barriers. In the implementation process of the case study, it is considered
at group level and model site level through operational, social and
technological perspectives in a five-phase multi-method approach. It shows what
effective human-centered lean Industry 4.0 implementation look like and how
advanced lean tools can be digitized. It highlights 26 positive and 10 negative
aspects of the case and their causal relation. With the appropriate internal
and external technological knowhow and people skills, it shows how successful
implementation can benefit the organization and employees based on the
conceptual model that serves as a first step toward lean Industry 5.0.

### 2. [Hetero-EUCLID: Interpretable model discovery for heterogeneous hyperelastic materials using stress-unsupervised learning](http://arxiv.org/pdf/2509.11784v1)

Authors: Kanhaiya Lal Chaurasiya, Saurav Dutta, Siddhant Kumar, Akshay Joshi

We propose a computational framework, Hetero-EUCLID, for segmentation and
parameter identification to characterize the full hyperelastic behavior of all
constituents of a heterogeneous material. In this work, we leverage the
Bayesian-EUCLID (Efficient Unsupervised Constitutive Law Identification and
Discovery) framework to efficiently solve the heterogenized formulation through
parsimonious model selection using sparsity-promoting priors and Monte Carlo
Markov Chain sampling. We utilize experimentally observable 3D surface
displacement and boundary-averaged force data generated from Finite Element
simulations of non-equi-biaxial tension tests on heterogeneous specimens. The
framework broadly consists of two steps -- residual force-based segmentation,
and constitutive parameter identification. We validate and demonstrate the
ability of the proposed framework to segment the domain, and characterize the
constituent materials on various types of thin square heterogeneous domains. We
validate of the framework's ability to segment and characterize materials with
various levels of displacement noises and non-native mesh discretizations, i.e,
using different meshes for the forward FE simulations and the inverse EUCLID
problem. This demonstrates Hetero-EUCLID framework's applicability in Digital
Image/Volume Correlation-based experimental scenarios. Furthermore, the
proposed framework performs successful segmentation and material
characterizations based on data from a single experiment, thereby making it
viable for rapid, interpretable model discovery in domains such as aerospace
and defense composites and for characterization of selective tissue stiffening
in medical conditions such as fibroatheroma, atherosclerosis, or cancer.

### 3. [Very-low-field MRI scanners: from the ideal to the real permanent magnet array](http://arxiv.org/pdf/2509.11762v1)

Authors: Umberto Zanovello, Alessandro Arduino, Vittorio Basso, Luca Zilberti, Alessandro Sola, Andrea Agosto, Luca Toso, Oriano Bottauscio

Very-low-field MRIs are becoming increasingly popular due to their
portability and adaptability to different environments. They are being
successfully used for various clinical applications, leading to a paradigm
shift in the way imaging care is typically performed. The development of
low-cost MRI scanner prototypes began a few years ago, with some interesting
and promising open-source projects emerging in both hardware and software
design. Using permanent magnets (PMs) to generate the static magnetic field B0
can substantially reduce the manufacturing cost of low-field scanners while
achieving satisfactory homogeneity. This article focuses on characterizing
magnet performance in terms of B0 spatial homogeneity. Specifically, it
investigates its sensitivity to various factors and explores the reasons for
discrepancies between numerical expectations and actual measurements on
fabricated magnets. The analysis also examines the consequences of using
different numerical model approximations, revisiting concepts most frequently
used in other design contexts. While these assumptions simplify the numerical
model and may improve its performance in terms of computational time, this
paper demonstrates that they also impact the reliability of the obtained
results.

### 4. [FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval](http://arxiv.org/pdf/2509.12042v1)

Authors: Ying Li, Mengyu Wang, Miguel de Carvalho, Sotirios Sabanis, Tiejun Ma

Financial disclosures such as 10-K filings present challenging retrieval
problems due to their length, regulatory section hierarchy, and domain-specific
language, which standard retrieval-augmented generation (RAG) models underuse.
We introduce FinGEAR (Financial Mapping-Guided Enhanced Answer Retrieval), a
retrieval framework tailored to financial documents. FinGEAR combines a finance
lexicon for Item-level guidance (FLAM), dual hierarchical indices for
within-Item search (Summary Tree and Question Tree), and a two-stage
cross-encoder reranker. This design aligns retrieval with disclosure structure
and terminology, enabling fine-grained, query-aware context selection.
Evaluated on full 10-Ks with queries aligned to the FinQA dataset, FinGEAR
delivers consistent gains in precision, recall, F1, and relevancy, improving F1
by up to 56.7% over flat RAG, 12.5% over graph-based RAGs, and 217.6% over
prior tree-based systems, while also increasing downstream answer accuracy with
a fixed reader. By jointly modeling section hierarchy and domain lexicon
signals, FinGEAR improves retrieval fidelity and provides a practical
foundation for high-stakes financial analysis.

### 5. [Numerical analysis of fluid estimation for source terms in neutral particles simulation](http://arxiv.org/pdf/2509.11883v1)

Authors: Zhirui Tang, Emil Løvbak, Julian Koellermeier, Giovanni Samaey

In plasma edge simulations, kinetic Monte Carlo (MC) is often used to
simulate neutral particles and estimate source terms. For large-sized reactors,
like ITER and DEMO, high particle collision rates lead to a substantial
computational cost for such schemes. To address this challenge, an
asymptotic-preserving kinetic-diffusion Monte Carlo (KDMC) simulation method
and a corresponding fluid estimation technique have been proposed in the
literature. In this work, we perform numerical analysis on the convergence of
KDMC with the fluid estimation. To do so, we compare the accuracy of the
analyzed algorithm with the accuracy of an approximate fluid method using the
kinetic MC method as a reference. In a one-dimensional test case, KDMC with the
fluid estimation achieves at least one order of magnitude lower errors than the
fluid method for both high- and low-collisional regimes. Moreover, KDMC with
the fluid estimation outperforms the kinetic MC method with a clear speed-up.
Overall, our analysis confirms the effectiveness of the discussed algorithm.

### 6. [AMLNet: A Knowledge-Based Multi-Agent Framework to Generate and Detect Realistic Money Laundering Transactions](http://arxiv.org/pdf/2509.11595v1)

Authors: Sabin Huda, Ernest Foo, Zahra Jadidi, MA Hakim Newton, Abdul Sattar

Anti-money laundering (AML) research is constrained by the lack of publicly
shareable, regulation-aligned transaction datasets. We present AMLNet, a
knowledge-based multi-agent framework with two coordinated units: a
regulation-aware transaction generator and an ensemble detection pipeline. The
generator produces 1,090,173 synthetic transactions (approximately 0.16\%
laundering-positive) spanning core laundering phases (placement, layering,
integration) and advanced typologies (e.g., structuring, adaptive threshold
behavior). Regulatory alignment reaches 75\% based on AUSTRAC rule coverage
(Section 4.2), while a composite technical fidelity score of 0.75 summarizes
temporal, structural, and behavioral realism components (Section 4.4). The
detection ensemble achieves F1 0.90 (precision 0.84, recall 0.97) on the
internal test partitions of AMLNet and adapts to the external SynthAML dataset,
indicating architectural generalizability across different synthetic generation
paradigms. We provide multi-dimensional evaluation (regulatory, temporal,
network, behavioral) and release the dataset (Version 1.0,
https://doi.org/10.5281/zenodo.16736515), to advance reproducible and
regulation-conscious AML experimentation.

### Computation and Language

### 1. [AKCIT-FN at CheckThat! 2025: Switching Fine-Tuned SLMs and LLM Prompting for Multilingual Claim Normalization](http://arxiv.org/pdf/2509.11496v1)

Authors: Fabrycio Leite Nakano Almada, Kauan Divino Pouso Mariano, Maykon Adriell Dutra, Victor Emanuel da Silva Monteiro, Juliana Resplande Sant'Anna Gomes, Arlindo Rodrigues Galvão Filho, Anderson da Silva Soares

Claim normalization, the transformation of informal social media posts into
concise, self-contained statements, is a crucial step in automated
fact-checking pipelines. This paper details our submission to the CLEF-2025
CheckThat! Task~2, which challenges systems to perform claim normalization
across twenty languages, divided into thirteen supervised (high-resource) and
seven zero-shot (no training data) tracks.
  Our approach, leveraging fine-tuned Small Language Models (SLMs) for
supervised languages and Large Language Model (LLM) prompting for zero-shot
scenarios, achieved podium positions (top three) in fifteen of the twenty
languages. Notably, this included second-place rankings in eight languages,
five of which were among the seven designated zero-shot languages, underscoring
the effectiveness of our LLM-based zero-shot strategy. For Portuguese, our
initial development language, our system achieved an average METEOR score of
0.5290, ranking third. All implementation artifacts, including inference,
training, evaluation scripts, and prompt configurations, are publicly available
at https://github.com/ju-resplande/checkthat2025_normalization.

### 2. [DeDisCo at the DISRPT 2025 Shared Task: A System for Discourse Relation Classification](http://arxiv.org/pdf/2509.11498v1)

Authors: Zhuoxuan Ju, Jingni Wu, Abhishek Purushothama, Amir Zeldes

This paper presents DeDisCo, Georgetown University's entry in the DISRPT 2025
shared task on discourse relation classification. We test two approaches, using
an mt5-based encoder and a decoder based approach using the openly available
Qwen model. We also experiment on training with augmented dataset for
low-resource languages using matched data translated automatically from
English, as well as using some additional linguistic features inspired by
entries in previous editions of the Shared Task. Our system achieves a
macro-accuracy score of 71.28, and we provide some interpretation and error
analysis for our results.

### 3. [LVLMs are Bad at Overhearing Human Referential Communication](http://arxiv.org/pdf/2509.11514v1)

Authors: Zhengxiang Wang, Weiling Li, Panagiotis Kaliosis, Owen Rambow, Susan E. Brennan

During spontaneous conversations, speakers collaborate on novel referring
expressions, which they can then re-use in subsequent conversations.
Understanding such referring expressions is an important ability for an
embodied agent, so that it can carry out tasks in the real world. This requires
integrating and understanding language, vision, and conversational interaction.
We study the capabilities of seven state-of-the-art Large Vision Language
Models (LVLMs) as overhearers to a corpus of spontaneous conversations between
pairs of human discourse participants engaged in a collaborative
object-matching task. We find that such a task remains challenging for current
LVLMs and they all fail to show a consistent performance improvement as they
overhear more conversations from the same discourse participants repeating the
same task for multiple rounds. We release our corpus and code for
reproducibility and to facilitate future research.

### 4. [On the Distinctive Co-occurrence Characteristics of Antonymy](http://arxiv.org/pdf/2509.11534v1)

Authors: Zhihan Cao, Hiroaki Yamada, Takenobu Tokunaga

Antonymy has long received particular attention in lexical semantics.
Previous studies have shown that antonym pairs frequently co-occur in text,
across genres and parts of speech, more often than would be expected by chance.
However, whether this co-occurrence pattern is distinctive of antonymy remains
unclear, due to a lack of comparison with other semantic relations. This work
fills the gap by comparing antonymy with three other relations across parts of
speech using robust co-occurrence metrics. We find that antonymy is distinctive
in three respects: antonym pairs co-occur with high strength, in a preferred
linear order, and within short spans. All results are available online.

### 5. [D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs](http://arxiv.org/pdf/2509.11569v1)

Authors: Yue Ding, Xiaofang Zhu, Tianze Xia, Junfei Wu, Xinlong Chen, Qiang Liu, Liang Wang

Although large Language Models (LLMs) have achieved remarkable success, their
practical application is often hindered by the generation of non-factual
content, which is called "hallucination". Ensuring the reliability of LLMs'
outputs is a critical challenge, particularly in high-stakes domains such as
finance, security, and healthcare. In this work, we revisit hallucination
detection from the perspective of model architecture and generation dynamics.
Leveraging the multi-layer structure and autoregressive decoding process of
LLMs, we decompose hallucination signals into two complementary dimensions: the
semantic breadth of token representations within each layer, and the semantic
depth of core concepts as they evolve across layers. Based on this insight, we
propose \textbf{D$^2$HScore (Dispersion and Drift-based Hallucination Score)},
a training-free and label-free framework that jointly measures: (1)
\textbf{Intra-Layer Dispersion}, which quantifies the semantic diversity of
token representations within each layer; and (2) \textbf{Inter-Layer Drift},
which tracks the progressive transformation of key token representations across
layers. To ensure drift reflects the evolution of meaningful semantics rather
than noisy or redundant tokens, we guide token selection using attention
signals. By capturing both the horizontal and vertical dynamics of
representation during inference, D$^2$HScore provides an interpretable and
lightweight proxy for hallucination detection. Extensive experiments across
five open-source LLMs and five widely used benchmarks demonstrate that
D$^2$HScore consistently outperforms existing training-free baselines.

### 6. [Bhaasha, Bhasa, Zaban: A Survey for Low-Resourced Languages in South Asia -- Current Stage and Challenges](http://arxiv.org/pdf/2509.11570v1)

Authors: Sampoorna Poria, Xiaolei Huang

Rapid developments of large language models have revolutionized many NLP
tasks for English data. Unfortunately, the models and their evaluations for
low-resource languages are being overlooked, especially for languages in South
Asia. Although there are more than 650 languages in South Asia, many of them
either have very limited computational resources or are missing from existing
language models. Thus, a concrete question to be answered is: Can we assess the
current stage and challenges to inform our NLP community and facilitate model
developments for South Asian languages? In this survey, we have comprehensively
examined current efforts and challenges of NLP models for South Asian languages
by retrieving studies since 2020, with a focus on transformer-based models,
such as BERT, T5, & GPT. We present advances and gaps across 3 essential
aspects: data, models, & tasks, such as available data sources, fine-tuning
strategies, & domain applications. Our findings highlight substantial issues,
including missing data in critical domains (e.g., health), code-mixing, and
lack of standardized evaluation benchmarks. Our survey aims to raise awareness
within the NLP community for more targeted data curation, unify benchmarks
tailored to cultural and linguistic nuances of South Asia, and encourage an
equitable representation of South Asian languages. The complete list of
resources is available at: https://github.com/trust-nlp/LM4SouthAsia-Survey.

### 7. [Analyzing Information-Seeking Behaviors in a Hakka AI Chatbot: A Cognitive-Pragmatic Study](http://arxiv.org/pdf/2509.11591v1)

Authors: Chu-Hsuan Lee, Chen-Chi Chang, Hung-Shin Lee, Yun-Hsiang Hsu, Ching-Yuan Chen

With many endangered languages at risk of disappearing, efforts to preserve
them now rely more than ever on using technology alongside culturally informed
teaching strategies. This study examines user behaviors in TALKA, a generative
AI-powered chatbot designed for Hakka language engagement, by employing a
dual-layered analytical framework grounded in Bloom's Taxonomy of cognitive
processes and dialogue act categorization. We analyzed 7,077 user utterances,
each carefully annotated according to six cognitive levels and eleven dialogue
act types. These included a variety of functions, such as asking for
information, requesting translations, making cultural inquiries, and using
language creatively. Pragmatic classifications further highlight how different
types of dialogue acts--such as feedback, control commands, and social
greetings--align with specific cognitive intentions. The results suggest that
generative AI chatbots can support language learning in meaningful
ways--especially when they are designed with an understanding of how users
think and communicate. They may also help learners express themselves more
confidently and connect with their cultural identity. The TALKA case provides
empirical insights into how AI-mediated dialogue facilitates cognitive
development in low-resource language learners, as well as pragmatic negotiation
and socio-cultural affiliation. By focusing on AI-assisted language learning,
this study offers new insights into how technology can support language
preservation and educational practice.

### 8. [Dynamic Span Interaction and Graph-Aware Memory for Entity-Level Sentiment Classification](http://arxiv.org/pdf/2509.11604v1)

Authors: Md. Mithun Hossain, Sanjara, Md. Shakil Hossain, Sudipto Chaki

Entity-level sentiment classification involves identifying the sentiment
polarity linked to specific entities within text. This task poses several
challenges: effectively modeling the subtle and complex interactions between
entities and their surrounding sentiment expressions; capturing dependencies
that may span across sentences; and ensuring consistent sentiment predictions
for multiple mentions of the same entity through coreference resolution.
Additionally, linguistic phenomena such as negation, ambiguity, and overlapping
opinions further complicate the analysis. These complexities make entity-level
sentiment classification a difficult problem, especially in real-world, noisy
textual data. To address these issues, we propose SpanEIT, a novel framework
integrating dynamic span interaction and graph-aware memory mechanisms for
enhanced entity-sentiment relational modeling. SpanEIT builds span-based
representations for entities and candidate sentiment phrases, employs
bidirectional attention for fine-grained interactions, and uses a graph
attention network to capture syntactic and co-occurrence relations. A
coreference-aware memory module ensures entity-level consistency across
documents. Experiments on FSAD, BARU, and IMDB datasets show SpanEIT
outperforms state-of-the-art transformer and hybrid baselines in accuracy and
F1 scores. Ablation and interpretability analyses validate the effectiveness of
our approach, underscoring its potential for fine-grained sentiment analysis in
applications like social media monitoring and customer feedback analysis.

### 9. [HalluDetect: Detecting, Mitigating, and Benchmarking Hallucinations in Conversational Systems](http://arxiv.org/pdf/2509.11619v1)

Authors: Spandan Anaokar, Shrey Ganatra, Harshvivek Kashid, Swapnil Bhattacharyya, Shruti Nair, Reshma Sekhar, Siddharth Manohar, Rahul Hemrajani, Pushpak Bhattacharyya

Large Language Models (LLMs) are widely used in industry but remain prone to
hallucinations, limiting their reliability in critical applications. This work
addresses hallucination reduction in consumer grievance chatbots built using
LLaMA 3.1 8B Instruct, a compact model frequently used in industry. We develop
HalluDetect, an LLM-based hallucination detection system that achieves an F1
score of 69% outperforming baseline detectors by 25.44%. Benchmarking five
chatbot architectures, we find that out of them, AgentBot minimizes
hallucinations to 0.4159 per turn while maintaining the highest token accuracy
(96.13%), making it the most effective mitigation strategy. Our findings
provide a scalable framework for hallucination mitigation, demonstrating that
optimized inference strategies can significantly improve factual accuracy.
While applied to consumer law, our approach generalizes to other high-risk
domains, enhancing trust in LLM-driven assistants. We will release the code and
dataset

### 10. [A Dynamic Knowledge Update-Driven Model with Large Language Models for Fake News Detection](http://arxiv.org/pdf/2509.11687v1)

Authors: Di Jin, Jun Yang, Xiaobao Wang, Junwei Zhang, Shuqi Li, Dongxiao He

As the Internet and social media evolve rapidly, distinguishing credible news
from a vast amount of complex information poses a significant challenge. Due to
the suddenness and instability of news events, the authenticity labels of news
can potentially shift as events develop, making it crucial for fake news
detection to obtain the latest event updates. Existing methods employ
retrieval-augmented generation to fill knowledge gaps, but they suffer from
issues such as insufficient credibility of retrieved content and interference
from noisy information. We propose a dynamic knowledge update-driven model for
fake news detection (DYNAMO), which leverages knowledge graphs to achieve
continuous updating of new knowledge and integrates with large language models
to fulfill dual functions: news authenticity detection and verification of new
knowledge correctness, solving the two key problems of ensuring the
authenticity of new knowledge and deeply mining news semantics. Specifically,
we first construct a news-domain-specific knowledge graph. Then, we use Monte
Carlo Tree Search to decompose complex news and verify them step by step.
Finally, we extract and update new knowledge from verified real news texts and
reasoning paths. Experimental results demonstrate that DYNAMO achieves the best
performance on two real-world datasets.

### Cryptography and Security

### 1. [Cyber Threat Hunting: Non-Parametric Mining of Attack Patterns from Cyber Threat Intelligence for Precise Threats Attribution](http://arxiv.org/pdf/2509.11615v1)

Authors: Rimsha Kanwal, Umara Noor, Zafar Iqbal, Zahid Rashid

With the ever-changing landscape of cyber threats, identifying their origin
has become paramount, surpassing the simple task of attack classification.
Cyber threat attribution gives security analysts the insights they need to
device effective threat mitigation strategies. Such strategies empower
enterprises to proactively detect and defend against future cyber-attacks.
However, existing approaches exhibit limitations in accurately identifying
threat actors, leading to low precision and a significant occurrence of false
positives. Machine learning offers the potential to automate certain aspects of
cyber threat attribution. The distributed nature of information regarding cyber
threat actors and their intricate attack methodologies has hindered substantial
progress in this domain. Cybersecurity analysts deal with an ever-expanding
collection of cyber threat intelligence documents. While these documents hold
valuable insights, their sheer volume challenges efficient organization and
retrieval of pertinent information. To assist the cybersecurity analyst
activities, we propose a machine learning based approach featuring visually
interactive analytics tool named the Cyber-Attack Pattern Explorer (CAPE),
designed to facilitate efficient information discovery by employing interactive
visualization and mining techniques. In the proposed system, a non-parametric
mining technique is proposed to create a dataset for identifying the attack
patterns within cyber threat intelligence documents. These attack patterns
align semantically with commonly employed themes ensuring ease of
interpretation. The extracted dataset is used for training of proposed machine
learning algorithms that enables the attribution of cyber threats with
respective to the actors.

### 2. [Cyber Attack Mitigation Framework for Denial of Service (DoS) Attacks in Fog Computing](http://arxiv.org/pdf/2509.11668v1)

Authors: Fizza Khurshid, Umara Noor, Zahid Rashid

Innovative solutions to cyber security issues are shaped by the ever-changing
landscape of cyber threats. Automating the mitigation of these threats can be
achieved through a new methodology that addresses the domain of mitigation
automation, which is often overlooked. This literature overview emphasizes the
lack of scholarly work focusing specifically on automated cyber threat
mitigation, particularly in addressing challenges beyond detection. The
proposed methodology comprise of the development of an automatic cyber threat
mitigation framework tailored for Distributed Denial-of-Service (DDoS) attacks.
This framework adopts a multi-layer security approach, utilizing smart devices
at the device layer, and leveraging fog network and cloud computing layers for
deeper understanding and technological adaptability. Initially, firewall
rule-based packet inspection is conducted on simulated attack traffic to filter
out DoS packets, forwarding legitimate packets to the fog. The methodology
emphasizes the integration of fog detection through statistical and behavioral
analysis, specification-based detection, and deep packet inspection, resulting
in a comprehensive cyber protection system. Furthermore, cloud-level inspection
is performed to confirm and mitigate attacks using firewalls, enhancing
strategic defense and increasing robustness against cyber threats. These
enhancements enhance understanding of the research framework's practical
implementation and assessment strategies, substantiating its importance in
addressing current cyber security challenges and shaping future automation
mitigation approaches.

### 3. [An Unsupervised Learning Approach For A Reliable Profiling Of Cyber Threat Actors Reported Globally Based On Complete Contextual Information Of Cyber Attacks](http://arxiv.org/pdf/2509.11683v1)

Authors: Sawera Shahid, Umara Noor, Zahid Rashid

Cyber attacks are rapidly increasing with the advancement of technology and
there is no protection for our information. To prevent future cyberattacks it
is critical to promptly recognize cyberattacks and establish strong defense
mechanisms against them. To respond to cybersecurity threats immediately, it is
essential to examine the attackers skills, knowledge, and behaviors with the
goal of evaluating their impact on the system and comprehending the traits
associated with these attacks. Creating a profile of cyber threat actors based
on their traits or patterns of behavior can help to create effective defenses
against cyberattacks in advance. In the current literature, multiple supervised
machine learning based approaches considered a smaller number of features for
attacker profiling that are reported in textual cyber threat incident documents
although these profiles have been developed based on the security experts own
perception, we cannot rely on them. Supervised machine learning approaches
strictly depend upon the structure data set. This usually leads to a two step
process where we first have to establish a structured data set before we can
analyze it and then employ it to construct defense mechanisms, which takes
time. In this paper, an unsupervised efficient agglomerative hierarchal
clustering technique is proposed for profiling cybercriminal groups based on
their comprehensive contextual threat information in order to address the
aforementioned issues. The main objective of this report is to identify the
relationship between cyber threat actors based on their common features,
aggregate them, and also profile cyber criminal groups.

### 4. [Time-Based State-Management of Hash-Based Signature CAs for VPN-Authentication](http://arxiv.org/pdf/2509.11695v1)

Authors: Daniel Herzinger, Linus Heise, Daniel Loebenberger, Matthias Söllner

Advances in quantum computing necessitate migrating the entire technology
stack to post-quantum cryptography. This includes IPsec-based VPN connection
authentication. Although there is an RFC draft for post-quantum authentication
in this setting, the draft does not consider (stateful) hash-based signatures
despite their small signature size and trusted long-term security.
  We propose a design with time-based state-management that assigns VPN devices
a certificate authority (CA) based on the hash-based signature scheme XMSS. The
CA then issues leaf certificates which are based on classical cryptography but
have a short validity time, e. g., four hours. It is to be expected that even
large quantum computers will take significantly longer to break the
cryptography, making the design quantum-secure. We propose strategies to make
the timekeeping more resilient to faults and tampering, as well as strategies
to recognize a wrong system time, minimize its potential damage, and quickly
recover.
  The result is an OpenBSD implementation of a quantum-safe and, regarding the
leaf certificates, highly flexible VPN authentication design that requires
significantly less bandwidth and computational resources compared to existing
alternatives.

### 5. [Removal Attack and Defense on AI-generated Content Latent-based Watermarking](http://arxiv.org/pdf/2509.11745v1)

Authors: De Zhang Lee, Han Fang, Hanyi Wang, Ee-Chien Chang

Digital watermarks can be embedded into AI-generated content (AIGC) by
initializing the generation process with starting points sampled from a secret
distribution. When combined with pseudorandom error-correcting codes, such
watermarked outputs can remain indistinguishable from unwatermarked objects,
while maintaining robustness under whitenoise. In this paper, we go beyond
indistinguishability and investigate security under removal attacks. We
demonstrate that indistinguishability alone does not necessarily guarantee
resistance to adversarial removal. Specifically, we propose a novel attack that
exploits boundary information leaked by the locations of watermarked objects.
This attack significantly reduces the distortion required to remove watermarks
-- by up to a factor of $15 \times$ compared to a baseline whitenoise attack
under certain settings. To mitigate such attacks, we introduce a defense
mechanism that applies a secret transformation to hide the boundary, and prove
that the secret transformation effectively rendering any attacker's
perturbations equivalent to those of a naive whitenoise adversary. Our
empirical evaluations, conducted on multiple versions of Stable Diffusion,
validate the effectiveness of both the attack and the proposed defense,
highlighting the importance of addressing boundary leakage in latent-based
watermarking schemes.

### 6. [Anomaly Detection in Industrial Control Systems Based on Cross-Domain Representation Learning](http://arxiv.org/pdf/2509.11786v1)

Authors: Dongyang Zhan, Wenqi Zhang, Lin Ye, Xiangzhan Yu, Hongli Zhang, Zheng He

Industrial control systems (ICSs) are widely used in industry, and their
security and stability are very important. Once the ICS is attacked, it may
cause serious damage. Therefore, it is very important to detect anomalies in
ICSs. ICS can monitor and manage physical devices remotely using communication
networks. The existing anomaly detection approaches mainly focus on analyzing
the security of network traffic or sensor data. However, the behaviors of
different domains (e.g., network traffic and sensor physical status) of ICSs
are correlated, so it is difficult to comprehensively identify anomalies by
analyzing only a single domain. In this paper, an anomaly detection approach
based on cross-domain representation learning in ICSs is proposed, which can
learn the joint features of multi-domain behaviors and detect anomalies within
different domains. After constructing a cross-domain graph that can represent
the behaviors of multiple domains in ICSs, our approach can learn the joint
features of them by leveraging graph neural networks. Since anomalies behave
differently in different domains, we leverage a multi-task learning approach to
identify anomalies in different domains separately and perform joint training.
The experimental results show that the performance of our approach is better
than existing approaches for identifying anomalies in ICSs.

### 7. [Off-Path TCP Exploits: PMTUD Breaks TCP Connection Isolation in IP Address Sharing Scenarios](http://arxiv.org/pdf/2509.11833v1)

Authors: Xuewei Feng, Zhaoxi Li, Qi Li, Ziqiang Wang, Kun Sun, Ke Xu

Path MTU Discovery (PMTUD) and IP address sharing are integral aspects of
modern Internet infrastructure. In this paper, we investigate the security
vulnerabilities associated with PMTUD within the context of prevalent IP
address sharing practices. We reveal that PMTUD is inadequately designed to
handle IP address sharing, creating vulnerabilities that attackers can exploit
to perform off-path TCP hijacking attacks. We demonstrate that by observing the
path MTU value determined by a server for a public IP address (shared among
multiple devices), an off-path attacker on the Internet, in collaboration with
a malicious device, can infer the sequence numbers of TCP connections
established by other legitimate devices sharing the same IP address. This
vulnerability enables the attacker to perform off-path TCP hijacking attacks,
significantly compromising the security of the affected TCP connections. Our
attack involves first identifying a target TCP connection originating from the
shared IP address, followed by inferring the sequence numbers of the identified
connection. We thoroughly assess the impacts of our attack under various
network configurations. Experimental results reveal that the attack can be
executed within an average time of 220 seconds, achieving a success rate of
70%.Case studies, including SSH DoS, FTP traffic poisoning, and HTTP injection,
highlight the threat it poses to various applications. Additionally, we
evaluate our attack across 50 real-world networks with IP address
sharing--including public Wi-Fi, VPNs, and 5G--and find 38 vulnerable. Finally,
we responsibly disclose the vulnerabilities, receive recognition from
organizations such as IETF, Linux, and Cisco, and propose our countermeasures.

### 8. [NeuroStrike: Neuron-Level Attacks on Aligned LLMs](http://arxiv.org/pdf/2509.11864v1)

Authors: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

Safety alignment is critical for the ethical deployment of large language
models (LLMs), guiding them to avoid generating harmful or unethical content.
Current alignment techniques, such as supervised fine-tuning and reinforcement
learning from human feedback, remain fragile and can be bypassed by carefully
crafted adversarial prompts. Unfortunately, such attacks rely on trial and
error, lack generalizability across models, and are constrained by scalability
and reliability.
  This paper presents NeuroStrike, a novel and generalizable attack framework
that exploits a fundamental vulnerability introduced by alignment techniques:
the reliance on sparse, specialized safety neurons responsible for detecting
and suppressing harmful inputs. We apply NeuroStrike to both white-box and
black-box settings: In the white-box setting, NeuroStrike identifies safety
neurons through feedforward activation analysis and prunes them during
inference to disable safety mechanisms. In the black-box setting, we propose
the first LLM profiling attack, which leverages safety neuron transferability
by training adversarial prompt generators on open-weight surrogate models and
then deploying them against black-box and proprietary targets. We evaluate
NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing
less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average
attack success rate (ASR) of 76.9% using only vanilla malicious prompts.
Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on
unsafe image inputs. Safety neurons transfer effectively across architectures,
raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled
models. The black-box LLM profiling attack achieves an average ASR of 63.7%
across five black-box models, including the Google Gemini family.

### 9. [Efficient Byzantine-Robust Privacy-Preserving Federated Learning via Dimension Compression](http://arxiv.org/pdf/2509.11870v1)

Authors: Xian Qin, Xue Yang, Xiaohu Tang

Federated Learning (FL) allows collaborative model training across
distributed clients without sharing raw data, thus preserving privacy. However,
the system remains vulnerable to privacy leakage from gradient updates and
Byzantine attacks from malicious clients. Existing solutions face a critical
trade-off among privacy preservation, Byzantine robustness, and computational
efficiency. We propose a novel scheme that effectively balances these competing
objectives by integrating homomorphic encryption with dimension compression
based on the Johnson-Lindenstrauss transformation. Our approach employs a
dual-server architecture that enables secure Byzantine defense in the
ciphertext domain while dramatically reducing computational overhead through
gradient compression. The dimension compression technique preserves the
geometric relationships necessary for Byzantine defence while reducing
computation complexity from $O(dn)$ to $O(kn)$ cryptographic operations, where
$k \ll d$. Extensive experiments across diverse datasets demonstrate that our
approach maintains model accuracy comparable to non-private FL while
effectively defending against Byzantine clients comprising up to $40\%$ of the
network.

### 10. [zkToken: Empowering Holders to Limit Revocation Checks for Verifiable Credentials](http://arxiv.org/pdf/2509.11934v1)

Authors: Praveensankar Manimaran, Mayank Raikwar, Thiago Garrett, Arlindo F. da Conceição, Leander Jehl, Roman Vitenberg

Systems managing Verifiable Credentials are becoming increasingly popular.
Unfortunately, their support for revoking previously issued credentials allows
verifiers to effectively monitor the validity of the credentials, which is
sensitive information. While the issue started to gain recognition, no adequate
solution has been proposed so far.
  In this work, we propose a novel framework for time-limited continuous
verification. The holder is able to individually configure the verification
period when sharing information with the verifier, and the system guarantees
proven untraceability of the revocation status after the verification period
expires. Different from existing systems, the implementation adopts a more
scalable blacklist approach where tokens corresponding to revoked credentials
are stored in the registry. The approach employs ZK proofs that allow holders
to prove non-membership in the blacklist. In addition to theoretically proving
security, we evaluate the approach analytically and experimentally and show
that it significantly improves bandwidth consumption on the holder while being
on par with state-of-the-art solutions with respect to the other performance
metrics.

### Computer Vision and Pattern Recognition

### 1. [Multiple Instance Learning Framework with Masked Hard Instance Mining for Gigapixel Histopathology Image Analysis](http://arxiv.org/pdf/2509.11526v1)

Authors: Wenhao Tang, Sheng Huang, Heng Fang, Fengtao Zhou, Bo Liu, Qingshan Liu

Digitizing pathological images into gigapixel Whole Slide Images (WSIs) has
opened new avenues for Computational Pathology (CPath). As positive tissue
comprises only a small fraction of gigapixel WSIs, existing Multiple Instance
Learning (MIL) methods typically focus on identifying salient instances via
attention mechanisms. However, this leads to a bias towards easy-to-classify
instances while neglecting challenging ones. Recent studies have shown that
hard examples are crucial for accurately modeling discriminative boundaries.
Applying such an idea at the instance level, we elaborate a novel MIL framework
with masked hard instance mining (MHIM-MIL), which utilizes a Siamese structure
with a consistency constraint to explore the hard instances. Using a
class-aware instance probability, MHIM-MIL employs a momentum teacher to mask
salient instances and implicitly mine hard instances for training the student
model. To obtain diverse, non-redundant hard instances, we adopt large-scale
random masking while utilizing a global recycle network to mitigate the risk of
losing key features. Furthermore, the student updates the teacher using an
exponential moving average, which identifies new hard instances for subsequent
training iterations and stabilizes optimization. Experimental results on cancer
diagnosis, subtyping, survival analysis tasks, and 12 benchmarks demonstrate
that MHIM-MIL outperforms the latest methods in both performance and
efficiency. The code is available at: https://github.com/DearCaat/MHIM-MIL.

### 2. [SFGNet: Semantic and Frequency Guided Network for Camouflaged Object Detection](http://arxiv.org/pdf/2509.11539v1)

Authors: Dezhen Wang, Haixiang Zhao, Xiang Shen, Sheng Miao

Camouflaged object detection (COD) aims to segment objects that blend into
their surroundings. However, most existing studies overlook the semantic
differences among textual prompts of different targets as well as fine-grained
frequency features. In this work, we propose a novel Semantic and Frequency
Guided Network (SFGNet), which incorporates semantic prompts and
frequency-domain features to capture camouflaged objects and improve boundary
perception. We further design Multi-Band Fourier Module(MBFM) to enhance the
ability of the network in handling complex backgrounds and blurred boundaries.
In addition, we design an Interactive Structure Enhancement Block (ISEB) to
ensure structural integrity and boundary details in the predictions. Extensive
experiments conducted on three COD benchmark datasets demonstrate that our
method significantly outperforms state-of-the-art approaches. The core code of
the model is available at the following link:
https://github.com/winter794444/SFGNetICASSP2026.

### 3. [How Auxiliary Reasoning Unleashes GUI Grounding in VLMs](http://arxiv.org/pdf/2509.11548v1)

Authors: Weiming Li, Yan Shao, Jing Yang, Yujing Lu, Ling Zhong, Yuhan Wang, Manni Duan

Graphical user interface (GUI) grounding is a fundamental task for building
GUI agents. However, general vision-language models (VLMs) struggle with this
task due to a lack of specific optimization. We identify a key gap in this
paper: while VLMs exhibit significant latent grounding potential, as
demonstrated by their performance measured by Pointing Game, they underperform
when tasked with outputting explicit coordinates. To address this discrepancy,
and bypass the high data and annotation costs of current fine-tuning
approaches, we propose three zero-shot auxiliary reasoning methods. By
providing explicit spatial cues such as axes, grids and labeled intersections
as part of the input image, these methods enable VLMs to articulate their
implicit spatial understanding capabilities. We evaluate these methods on four
GUI grounding benchmarks across seven open-source and proprietary VLMs. The
evaluation results demonstrate that the proposed methods substantially improve
the performance of GUI grounding.

### 4. [Gaussian-Plus-SDF SLAM: High-fidelity 3D Reconstruction at 150+ fps](http://arxiv.org/pdf/2509.11574v1)

Authors: Zhexi Peng, Kun Zhou, Tianjia Shao

While recent Gaussian-based SLAM methods achieve photorealistic
reconstruction from RGB-D data, their computational performance remains a
critical bottleneck. State-of-the-art techniques operate at less than 20 fps,
significantly lagging behind geometry-centric approaches like KinectFusion
(hundreds of fps). This limitation stems from the heavy computational burden:
modeling scenes requires numerous Gaussians and complex iterative optimization
to fit RGB-D data, where insufficient Gaussian counts or optimization
iterations cause severe quality degradation. To address this, we propose a
Gaussian-SDF hybrid representation, combining a colorized Signed Distance Field
(SDF) for smooth geometry and appearance with 3D Gaussians to capture
underrepresented details. The SDF is efficiently constructed via RGB-D fusion
(as in geometry-centric methods), while Gaussians undergo iterative
optimization. Our representation enables drastic Gaussian reduction (50% fewer)
by avoiding full-scene Gaussian modeling, and efficient Gaussian optimization
(75% fewer iterations) through targeted appearance refinement. Building upon
this representation, we develop GPS-SLAM (Gaussian-Plus-SDF SLAM), a real-time
3D reconstruction system achieving over 150 fps on real-world Azure Kinect
sequences -- delivering an order-of-magnitude speedup over state-of-the-art
techniques while maintaining comparable reconstruction quality. We will release
the source code and data to facilitate future research.

### 5. [Optimizing Class Distributions for Bias-Aware Multi-Class Learning](http://arxiv.org/pdf/2509.11588v1)

Authors: Mirco Felske, Stefan Stiene

We propose BiCDO (Bias-Controlled Class Distribution Optimizer), an
iterative, data-centric framework that identifies Pareto optimized class
distributions for multi-class image classification. BiCDO enables performance
prioritization for specific classes, which is useful in safety-critical
scenarios (e.g. prioritizing 'Human' over 'Dog'). Unlike uniform distributions,
BiCDO determines the optimal number of images per class to enhance reliability
and minimize bias and variance in the objective function. BiCDO can be
incorporated into existing training pipelines with minimal code changes and
supports any labelled multi-class dataset. We have validated BiCDO using
EfficientNet, ResNet and ConvNeXt on CIFAR-10 and iNaturalist21 datasets,
demonstrating improved, balanced model performance through optimized data
distribution.

### 6. [MVQA-68K: A Multi-dimensional and Causally-annotated Dataset with Quality Interpretability for Video Assessment](http://arxiv.org/pdf/2509.11589v1)

Authors: Yanyun Pu, Kehan Li, Zeyi Huang, Zhijie Zhong, Kaixiang Yang

With the rapid advancement of video generation models such as Sora, video
quality assessment (VQA) is becoming increasingly crucial for selecting
high-quality videos from large-scale datasets used in pre-training. Traditional
VQA methods, typically producing single numerical scores, often lack
comprehensiveness and interpretability. To address these challenges, we
introduce MVQA-68K, a novel multi-dimensional VQA dataset comprising over
68,000 carefully annotated videos, covering seven essential quality dimensions:
overall aesthetics, camera movement, dynamic degree, texture detail,
composition, visual quality, and factual consistency. Each annotation includes
detailed chain-of-thought reasoning to facilitate interpretability and
comprehensive understanding. Extensive experiments demonstrate that MVQA-68K
significantly enhances the performance of various multimodal large language
models (MLLMs) on the VQA task, achieving state-of-the-art results not only on
our internal test set (Fig.1) but also on public benchmarks including
LSVQ-test, LSVQ-1080p, and LIVE-VQC. Meantime, incorporating explicit reasoning
process during VQA training substantially boosts the zero-shot generalization.
Code and dataset will be available at github:
https://github.com/Controller01-ai/MVQA-68K

### 7. [DUAL-VAD: Dual Benchmarks and Anomaly-Focused Sampling for Video Anomaly Detection](http://arxiv.org/pdf/2509.11605v1)

Authors: Seoik Jung, Taekyung Song, Joshua Jordan Daniel, JinYoung Lee, SungJun Lee

Video Anomaly Detection (VAD) is critical for surveillance and public safety.
However, existing benchmarks are limited to either frame-level or video-level
tasks, restricting a holistic view of model generalization. This work first
introduces a softmax-based frame allocation strategy that prioritizes
anomaly-dense segments while maintaining full-video coverage, enabling balanced
sampling across temporal scales. Building on this process, we construct two
complementary benchmarks. The image-based benchmark evaluates frame-level
reasoning with representative frames, while the video-based benchmark extends
to temporally localized segments and incorporates an abnormality scoring
task.Experiments on UCF-Crime demonstrate improvements at both the frame and
video levels, and ablation studies confirm clear advantages of anomaly-focused
sampling over uniform and random baselines.

### 8. [IS-Diff: Improving Diffusion-Based Inpainting with Better Initial Seed](http://arxiv.org/pdf/2509.11638v1)

Authors: Yongzhe Lyu, Yu Wu, Yutian Lin, Bo Du

Diffusion models have shown promising results in free-form inpainting. Recent
studies based on refined diffusion samplers or novel architectural designs led
to realistic results and high data consistency. However, random initialization
seed (noise) adopted in vanilla diffusion process may introduce mismatched
semantic information in masked regions, leading to biased inpainting results,
e.g., low consistency and low coherence with the other unmasked area. To
address this issue, we propose the Initial Seed refined Diffusion Model
(IS-Diff), a completely training-free approach incorporating distributional
harmonious seeds to produce harmonious results. Specifically, IS-Diff employs
initial seeds sampled from unmasked areas to imitate the masked data
distribution, thereby setting a promising direction for the diffusion
procedure. Moreover, a dynamic selective refinement mechanism is proposed to
detect severe unharmonious inpaintings in intermediate latent and adjust the
strength of our initialization prior dynamically. We validate our method on
both standard and large-mask inpainting tasks using the CelebA-HQ, ImageNet,
and Places2 datasets, demonstrating its effectiveness across all metrics
compared to state-of-the-art inpainting methods.

### 9. [WeatherBench: A Real-World Benchmark Dataset for All-in-One Adverse Weather Image Restoration](http://arxiv.org/pdf/2509.11642v1)

Authors: Qiyuan Guan, Qianfeng Yang, Xiang Chen, Tianyu Song, Guiyue Jin, Jiyu Jin

Existing all-in-one image restoration approaches, which aim to handle
multiple weather degradations within a single framework, are predominantly
trained and evaluated using mixed single-weather synthetic datasets. However,
these datasets often differ significantly in resolution, style, and domain
characteristics, leading to substantial domain gaps that hinder the development
and fair evaluation of unified models. Furthermore, the lack of a large-scale,
real-world all-in-one weather restoration dataset remains a critical bottleneck
in advancing this field. To address these limitations, we present a real-world
all-in-one adverse weather image restoration benchmark dataset, which contains
image pairs captured under various weather conditions, including rain, snow,
and haze, as well as diverse outdoor scenes and illumination settings. The
resulting dataset provides precisely aligned degraded and clean images,
enabling supervised learning and rigorous evaluation. We conduct comprehensive
experiments by benchmarking a variety of task-specific, task-general, and
all-in-one restoration methods on our dataset. Our dataset offers a valuable
foundation for advancing robust and practical all-in-one image restoration in
real-world scenarios. The dataset has been publicly released and is available
at https://github.com/guanqiyuan/WeatherBench.

### 10. [Joint-octamamba:an octa joint segmentation network based on feature enhanced mamba](http://arxiv.org/pdf/2509.11649v1)

Authors: Chuang Liu, Nan Guo

OCTA is a crucial non-invasive imaging technique for diagnosing and
monitoring retinal diseases like diabetic retinopathy, age-related macular
degeneration, and glaucoma. Current 2D-based methods for retinal vessel (RV)
segmentation offer insufficient accuracy. To address this, we propose RVMamba,
a novel architecture integrating multiple feature extraction modules with the
Mamba state-space model. Moreover, existing joint segmentation models for OCTA
data exhibit performance imbalance between different tasks. To simultaneously
improve the segmentation of the foveal avascular zone (FAZ) and mitigate this
imbalance, we introduce FAZMamba and a unified Joint-OCTAMamba framework.
Experimental results on the OCTA-500 dataset demonstrate that Joint-OCTAMamba
outperforms existing models across evaluation metrics.The code is available at
https://github.com/lc-sfis/Joint-OCTAMamba.

### Computers and Society

### 1. [Making Judicial Reasoning Visible: Structured Annotation of Holding, Evidentiary Considerations, and Subsumption in Criminal Judgments](http://arxiv.org/pdf/2509.11732v1)

Authors: Yu-Cheng Chih, Yong-Hao Hou

Judicial reasoning in criminal judgments typically consists of three
elements: Holding , evidentiary considerations, and subsumption. These elements
form the logical foundation of judicial decision-making but remain unstructured
in court documents, limiting large-scale empirical analysis. In this study, we
design annotation guidelines to define and distinguish these reasoning
components and construct the first dedicated datasets from Taiwanese High Court
and Supreme Court criminal judgments. Using the bilingual large language model
ChatGLM2, we fine-tune classifiers for each category. Preliminary experiments
demonstrate that the model achieves approximately 80% accuracy, showing that
judicial reasoning patterns can be systematically identified by large language
models even with relatively small annotated corpora. Our contributions are
twofold: (1) the creation of structured annotation rules and datasets for
Holding, evidentiary considerations, and subsumption; and (2) the demonstration
that such reasoning can be computationally learned. This work lays the
foundation for large-scale empirical legal studies and legal sociology,
providing new tools to analyze judicial fairness, consistency, and
transparency.

### 2. [Collective Recourse for Generative Urban Visualizations](http://arxiv.org/pdf/2509.11487v1)

Authors: Rashid Mushkani

Text-to-image diffusion models help visualize urban futures but can amplify
group-level harms. We propose collective recourse: structured community "visual
bug reports" that trigger fixes to models and planning workflows. We (1)
formalize collective recourse and a practical pipeline (report, triage, fix,
verify, closure); (2) situate four recourse primitives within the diffusion
stack: counter-prompts, negative prompts, dataset edits, and reward-model
tweaks; (3) define mandate thresholds via a mandate score combining severity,
volume saturation, representativeness, and evidence; and (4) evaluate a
synthetic program of 240 reports. Prompt-level fixes were fastest (median
2.1-3.4 days) but less durable (21-38% recurrence); dataset edits and reward
tweaks were slower (13.5 and 21.9 days) yet more durable (12-18% recurrence)
with higher planner uptake (30-36%). A threshold of 0.12 yielded 93% precision
and 75% recall; increasing representativeness raised recall to 81% with little
precision loss. We discuss integration with participatory governance, risks
(e.g., overfitting to vocal groups), and safeguards (dashboards, rotating
juries).

### 3. [AesBiasBench: Evaluating Bias and Alignment in Multimodal Language Models for Personalized Image Aesthetic Assessment](http://arxiv.org/pdf/2509.11620v1)

Authors: Kun Li, Lai-Man Po, Hongzheng Yang, Xuyuan Xu, Kangcheng Liu, Yuzhi Zhao

Multimodal Large Language Models (MLLMs) are increasingly applied in
Personalized Image Aesthetic Assessment (PIAA) as a scalable alternative to
expert evaluations. However, their predictions may reflect subtle biases
influenced by demographic factors such as gender, age, and education. In this
work, we propose AesBiasBench, a benchmark designed to evaluate MLLMs along two
complementary dimensions: (1) stereotype bias, quantified by measuring
variations in aesthetic evaluations across demographic groups; and (2)
alignment between model outputs and genuine human aesthetic preferences. Our
benchmark covers three subtasks (Aesthetic Perception, Assessment, Empathy) and
introduces structured metrics (IFD, NRD, AAS) to assess both bias and
alignment. We evaluate 19 MLLMs, including proprietary models (e.g., GPT-4o,
Claude-3.5-Sonnet) and open-source models (e.g., InternVL-2.5, Qwen2.5-VL).
Results indicate that smaller models exhibit stronger stereotype biases,
whereas larger models align more closely with human preferences. Incorporating
identity information often exacerbates bias, particularly in emotional
judgments. These findings underscore the importance of identity-aware
evaluation frameworks in subjective vision-language tasks.

### 4. [Regulating Ride-Sourcing Markets: Can Minimum Wage Regulation Protect Drivers Without Disrupting the Market?](http://arxiv.org/pdf/2509.11845v1)

Authors: Farnoud Ghasemi, Arjan de Ruijter, Rafal Kucharski, Oded Cats

Ride-sourcing platforms such as Uber and Lyft are prime examples of the gig
economy, recruiting drivers as independent contractors, thereby avoiding legal
and fiscal obligations. Although platforms offer flexibility in choosing work
shifts and areas, many drivers experience low income and poor working
conditions, leading to widespread strikes and protests. Minimum wage regulation
is adopted to improve drivers welfare. However, the impacts of this regulation
on drivers as well as on travelers and platforms, remain largely unknown. While
ride-sourcing platforms do not disclose the relevant data, state-of-the-art
models fail to explain the effects of minimum wage regulation on market
dynamics. In this study, we assess the effectiveness and implications of
minimum wage regulation in ride-sourcing markets while simulating the detailed
dynamics of ride-sourcing markets under varying regulation intensities, both
with and without the so-called platform lockout strategy. Our findings reveal
that minimum wage regulation impacts substantially drivers income, and may lead
to higher fares for travelers and threaten platforms survival. When platforms
adopt a lockout strategy, their profitability significantly improves and
drivers earn more, although many others lose their jobs, and service level for
travelers consequently declines.

### 5. [Transparent and Fair Profiling in Employment Services: Evidence from Switzerland](http://arxiv.org/pdf/2509.11847v1)

Authors: Tim Räz

Long-term unemployment (LTU) is a challenge for both jobseekers and public
employment services. Statistical profiling tools are increasingly used to
predict LTU risk. Some profiling tools are opaque, black-box machine learning
models, which raise issues of transparency and fairness. This paper
investigates whether interpretable models could serve as an alternative, using
administrative data from Switzerland. Traditional statistical, interpretable,
and black-box models are compared in terms of predictive performance,
interpretability, and fairness. It is shown that explainable boosting machines,
a recent interpretable model, perform nearly as well as the best black-box
models. It is also shown how model sparsity, feature smoothing, and fairness
mitigation can enhance transparency and fairness with only minor losses in
performance. These findings suggest that interpretable profiling provides an
accountable and trustworthy alternative to black-box models without
compromising performance.

### 6. [The dimensions of accessibility: proximity, opportunities, values](http://arxiv.org/pdf/2509.11875v1)

Authors: Matteo Bruno, Bruno Campanelli, Hygor Piaget Monteiro Melo, Lavinia Rossi Mori, Vittorio Loreto

Accessibility is essential for designing inclusive urban systems. However,
the attempt to capture the complexity of accessibility in a single universal
metric has often limited its effective use in design, measurement, and
governance across various fields. Building on the work of Levinson and Wu, we
emphasise that accessibility consists of several key dimensions. Specifically,
we introduce a conceptual framework that defines accessibility through three
main dimensions: Proximity (which pertains to active, short-range accessibility
to local services and amenities), Opportunity (which refers to quick access to
relevant non-local resources, such as jobs or major cultural venues), and Value
(which encompasses the overall quality and personal significance assigned to
specific points of interest). While it is generally beneficial to improve
accessibility, different users and contexts present unique trade-offs that make
a one-size-fits-all solution neither practical nor desirable. Our framework
establishes a foundation for a quantitative and integrative approach to
modelling accessibility. It considers the complex interactions among its
various dimensions and facilitates more systematic analysis, comparison, and
decision-making across diverse contexts.

### 7. [A GPU-Accelerated RAG-Based Telegram Assistant for Supporting Parallel Processing Students](http://arxiv.org/pdf/2509.11947v1)

Authors: Guy Tel-Zur

This project addresses a critical pedagogical need: offering students
continuous, on-demand academic assistance beyond conventional reception hours.
I present a domain-specific Retrieval-Augmented Generation (RAG) system powered
by a quantized Mistral-7B Instruct model and deployed as a Telegram bot. The
assistant enhances learning by delivering real-time, personalized responses
aligned with the "Introduction to Parallel Processing" course materials. GPU
acceleration significantly improves inference latency, enabling practical
deployment on consumer hardware. This approach demonstrates how consumer GPUs
can enable affordable, private, and effective AI tutoring for HPC education.

### 8. [Examining the Relationship between Scientific Publishing Activity and Hype-Driven Financial Bubbles: A Comparison of the Dot-Com and AI Eras](http://arxiv.org/pdf/2509.11982v1)

Authors: Aksheytha Chelikavada, Casey C. Bennett

Financial bubbles often arrive without much warning, but create long-lasting
economic effects. For example, during the dot-com bubble, innovative
technologies created market disruptions through excitement for a promised
bright future. Such technologies originated from research where scientists had
developed them for years prior to their entry into the markets. That raises a
question on the possibility of analyzing scientific publishing data (e.g.
citation networks) leading up to a bubble for signals that may forecast the
rise and fall of similar future bubbles. To that end, we utilized temporal SNAs
to detect possible relationships between the publication citation networks of
scientists and financial market data during two modern eras of rapidly shifting
technology: 1) dot-com era from 1994 to 2001 and 2) AI era from 2017 to 2024.
Results showed that the patterns from the dot-com era (which did end in a
bubble) did not definitively predict the rise and fall of an AI bubble. While
yearly citation networks reflected possible changes in publishing behavior of
scientists between the two eras, there was a subset of AI era scientists whose
publication influence patterns mirrored those during the dot-com era. Upon
further analysis using multiple analysis techniques (LSTM, KNN, AR X/GARCH),
the data seems to suggest two possibilities for the AI era: unprecedented form
of financial bubble unseen or that no bubble exists. In conclusion, our
findings imply that the patterns present in the dot-com era do not effectively
translate in such a manner to apply them to the AI market.

### 9. [Worker Discretion Advised: Co-designing Risk Disclosure in Crowdsourced Responsible AI (RAI) Content Work](http://arxiv.org/pdf/2509.12140v1)

Authors: Alice Qian, Ziqi Yang, Ryland Shaw, Jina Suh, Laura Dabbish, Hong Shen

Responsible AI (RAI) content work, such as annotation, moderation, or red
teaming for AI safety, often exposes crowd workers to potentially harmful
content. While prior work has underscored the importance of communicating
well-being risk to employed content moderators, designing effective disclosure
mechanisms for crowd workers while balancing worker protection with the needs
of task designers and platforms remains largely unexamined. To address this
gap, we conducted co-design sessions with 29 task designers, workers, and
platform representatives. We investigated task designer preferences for support
in disclosing tasks, worker preferences for receiving risk disclosure warnings,
and how platform stakeholders envision their role in shaping risk disclosure
practices. We identify design tensions and map the sociotechnical tradeoffs
that shape disclosure practices. We contribute design recommendations and
feature concepts for risk disclosure mechanisms in the context of RAI content
work.

### 10. [EthicsMH: A Pilot Benchmark for Ethical Reasoning in Mental Health AI](http://arxiv.org/pdf/2509.11648v1)

Authors: Sai Kartheek Reddy Kasu

The deployment of large language models (LLMs) in mental health and other
sensitive domains raises urgent questions about ethical reasoning, fairness,
and responsible alignment. Yet, existing benchmarks for moral and clinical
decision-making do not adequately capture the unique ethical dilemmas
encountered in mental health practice, where confidentiality, autonomy,
beneficence, and bias frequently intersect. To address this gap, we introduce
Ethical Reasoning in Mental Health (EthicsMH), a pilot dataset of 125 scenarios
designed to evaluate how AI systems navigate ethically charged situations in
therapeutic and psychiatric contexts. Each scenario is enriched with structured
fields, including multiple decision options, expert-aligned reasoning, expected
model behavior, real-world impact, and multi-stakeholder viewpoints. This
structure enables evaluation not only of decision accuracy but also of
explanation quality and alignment with professional norms. Although modest in
scale and developed with model-assisted generation, EthicsMH establishes a task
framework that bridges AI ethics and mental health decision-making. By
releasing this dataset, we aim to provide a seed resource that can be expanded
through community and expert contributions, fostering the development of AI
systems capable of responsibly handling some of society's most delicate
decisions.

### Databases

### 1. [The Space-Time Complexity of Sum-Product Queries](http://arxiv.org/pdf/2509.11920v1)

Authors: Kyle Deeds, Timo Camillo Merkl, Reinhard Pichler, Dan Suciu

While extensive research on query evaluation has achieved consistent
improvements in the time complexity of algorithms, the space complexity of
query evaluation has been largely ignored. This is a particular challenge in
settings with strict pre-defined space constraints. In this paper, we examine
the combined space-time complexity of conjunctive queries (CQs) and, more
generally, of sum-product queries (SPQs). We propose several classes of
space-efficient algorithms for evaluating SPQs, and we show that the optimal
time complexity is almost always achievable with asymptotically lower space
complexity than traditional approaches.

### 2. [Query Answering under Volume-Based Diversity Functions](http://arxiv.org/pdf/2509.11929v1)

Authors: Marcelo Arenas, Timo Camillo Merkl, Reinhard Pichler, Cristian Riveros

When query evaluation produces too many tuples, a new approach in query
answering is to retrieve a diverse subset of them. The standard approach for
measuring the diversity of a set of tuples is to use a distance function
between tuples, which measures the dissimilarity between them, to then
aggregate the pairwise distances of the set into a score (e.g., by using sum or
min aggregation). However, as we will point out in this work, the resulting
diversity measures may display some unintuitive behavior. Moreover, even in
very simple settings, finding a maximally diverse subset of the answers of
fixed size is, in general, intractable and little is known about approximations
apart from some hand-picked distance-aggregator pairs.
  In this work, we introduce a novel approach for computing the diversity of
tuples based on volume instead of distance. We present a framework for defining
volume-based diversity functions and provide several examples of these measures
applied to relational data. Although query answering of conjunctive queries
(CQ) under this setting is intractable in general, we show that one can always
compute a (1-1/e)-approximation for any volume-based diversity function.
Furthermore, in terms of combined complexity, we connect the evaluation of CQs
under volume-based diversity functions with the ranked enumeration of
solutions, finding general conditions under which a (1-1/e)-approximation can
be computed in polynomial time.

### 3. [Towards a Standard for JSON Document Databases](http://arxiv.org/pdf/2509.12189v1)

Authors: Elena Botoeva, Julien Corman

In this technical report, we present a formalisation of the MongoDB
aggregation framework. Our aim is to identify a fragment that could serve as
the starting point for an industry-wide standard for querying JSON document
databases. We provide a syntax and formal semantics for a set of selected
operators, We show how this fragment relates to known relational query
languages. We explain how our semantics differs from the current implementation
of MongoDB, and justify our choices. We provide a set of algebraic
transformations that can be used for query optimisation.

### 4. [SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation](http://arxiv.org/pdf/2509.12086v1)

Authors: Hui Li, Shiyuan Deng, Xiao Yan, Xiangyu Zhi, James Cheng

Approximate Nearest Neighbor Search (ANNS) plays a critical role in
applications such as search engines, recommender systems, and RAG for LLMs.
Vector quantization (VQ), a crucial technique for ANNS, is commonly used to
reduce space overhead and accelerate distance computations. However, despite
significant research advances, state-of-the-art VQ methods still face
challenges in balancing encoding efficiency and quantization accuracy. To
address these limitations, we propose a novel VQ method called SAQ. To improve
accuracy, SAQ employs a new dimension segmentation technique to strategically
partition PCA-projected vectors into segments along their dimensions. By
prioritizing leading dimension segments with larger magnitudes, SAQ allocates
more bits to high-impact segments, optimizing the use of the available space
quota. An efficient dynamic programming algorithm is developed to optimize
dimension segmentation and bit allocation, ensuring minimal quantization error.
To speed up vector encoding, SAQ devises a code adjustment technique to first
quantize each dimension independently and then progressively refine quantized
vectors using a coordinate-descent-like approach to avoid exhaustive
enumeration. Extensive experiments demonstrate SAQ's superiority over classical
methods (e.g., PQ, PCA) and recent state-of-the-art approaches (e.g., LVQ,
Extended RabitQ). SAQ achieves up to 80% reduction in quantization error and
accelerates encoding speed by over 80x compared to Extended RabitQ.

### Distributed, Parallel, and Cluster Computing

### 1. [Towards the Distributed Large-scale k-NN Graph Construction by Graph Merge](http://arxiv.org/pdf/2509.11697v1)

Authors: Cheng Zhang, Wan-Lei Zhao, Shihai Xiao, Jiajie Yao, Xuecang Zhang

In order to support the real-time interaction with LLMs and the instant
search or the instant recommendation on social media, it becomes an imminent
problem to build k-NN graph or indexing graph for the massive number of
vectorized multimedia data. In such scenarios, the scale of the data or the
scale of the graph may exceed the processing capacity of a single machine. This
paper aims to address the graph construction problem of such scale via
efficient graph merge. For the graph construction on a single node, two generic
and highly parallelizable algorithms, namely Two-way Merge and Multi-way Merge
are proposed to merge subgraphs into one. For the graph construction across
multiple nodes, a multi-node procedure based on Two-way Merge is presented. The
procedure makes it feasible to construct a large-scale k-NN graph/indexing
graph on either a single node or multiple nodes when the data size exceeds the
memory capacity of one node. Extensive experiments are conducted on both
large-scale k-NN graph and indexing graph construction. For the k-NN graph
construction, the large-scale and high-quality k-NN graphs are constructed by
graph merge in parallel. Typically, a billion-scale k-NN graph can be built in
approximately 17h when only three nodes are employed. For the indexing graph
construction, similar NN search performance as the original indexing graph is
achieved with the merged indexing graphs while requiring much less time of
construction.

### 2. [LASLiN: A Learning-Augmented Peer-to-Peer Network](http://arxiv.org/pdf/2509.11904v1)

Authors: Julien Dallot, Caio Caldeira, Arash Pourdamghani, Olga Goussevskaia, Stefan Schmid

We introduce a learning-augmented peer-to-peer (P2P) network design that
leverages the predictions of traffic patterns to optimize the network's
topology. While keeping formal guarantees on the standard P2P metrics (routing
path length, maximum degree), we optimize the network in a demand-aware manner
and minimize the path lengths weighted by the peer-to-peer communication
demands. Our protocol is learning-augmented, meaning that each node receives an
individual, possibly inaccurate prediction about the future traffic patterns,
with the goal of improving the network's performances. We strike a trade-off
between significantly improved performances when the predictions are correct
(consistency) and polylogarithmic performances when the predictions are
arbitrary (robustness).
  We have two main contributions. First, we consider the centralized setting
and show that the problem of constructing an optimum static skip list network
(SLN) is solvable in polynomial time and can be computed via dynamic
programming. This problem is the natural demand-aware extension of the optimal
skip list problem.
  Second, we introduce the Uniform P2P protocol which generalizes skip list
networks (SLN) by relaxing the node's heights from discrete to continuous. We
show that Uniform achieves state-of-the-art performances: logarithmic routing
and maximum degree, both with high probability. We then use Uniform to build a
learning-augmented P2P protocol in order to incorporate demand-awareness,
leading to our main contribution, LASLiN. We prove that the performances of
LASLiN are consistent with those of an optimum static SLN with correct
predictions (given via our dynamic programming approach), and are at most a
logarithmic factor off the state-of-the-art P2P protocols if the predictions
are arbitrary wrong. For the special case of highly sparse demands, we show
that LASLiN achieves improved performances.

### 3. [UniPar: A Unified LLM-Based Framework for Parallel and Accelerated Code Translation in HPC](http://arxiv.org/pdf/2509.12136v1)

Authors: Tomer Bitan, Tal Kadosh, Erel Kaplan, Shira Meiri, Le Chen, Peter Morales, Niranjan Hasabnis, Gal Oren

Translating programs between various parallel programming languages is an
important problem in the high-performance computing (HPC) community. Existing
tools for this problem are either too narrow in scope and/or outdated. Recent
explosive growth in the popularity of large language models (LLMs) and their
ability to generate and translate code offers a potential alternative approach.
Toward that end, we first need to systematically evaluate the ability of LLMs
to translate between parallel languages.
  In this work, we introduce UniPar, a systematic evaluation framework for
LLM-based parallel code translation. Specifically, in this work, we target
translations between serial code, CUDA, and OpenMP. Our goal is to assess how
well current instruction-tuned LLMs -- specifically GPT-4o-mini and
LLaMA-3.3-70B-Instruct -- can be used out of the box or enhanced through known
strategies. We evaluated four major usage modes: hyperparameter optimization
for decoding, zero- and few-shot prompting, supervised fine-tuning, and
iterative feedback through compiler-based repair. As a part of the evaluation,
we construct a new dataset called PARATRANS, covering both serial-to-parallel
translation and cross-paradigm transformations.
  Our findings reveal that while off-the-shelf models struggle under the
default settings (e.g., GPT-4o-mini achieves only 46% compilation and 15%
functional correctness), our UniPar methodology -- combining fine-tuning,
hyperparameter tuning, and compiler-guided repair -- improves performance by up
to 2X (69% compilation and 33% correctness). We believe that our findings will
provide useful insights for researchers to further improve LLMs for the
parallel language translation problem.
  UniPar source code and PARATRANS dataset are available at our GitHub
repository https://github.com/Scientific-Computing-Lab/UniPar_AI.

### 4. [Distributed 3D Gaussian Splatting for High-Resolution Isosurface Visualization](http://arxiv.org/pdf/2509.12138v1)

Authors: Mengjiao Han, Andres Sewell, Joseph Insley, Janet Knowles, Victor A. Mateevitsi, Michael E. Papka, Steve Petruzza, Silvio Rizzi

3D Gaussian Splatting (3D-GS) has recently emerged as a powerful technique
for real-time, photorealistic rendering by optimizing anisotropic Gaussian
primitives from view-dependent images. While 3D-GS has been extended to
scientific visualization, prior work remains limited to single-GPU settings,
restricting scalability for large datasets on high-performance computing (HPC)
systems. We present a distributed 3D-GS pipeline tailored for HPC. Our approach
partitions data across nodes, trains Gaussian splats in parallel using
multi-nodes and multi-GPUs, and merges splats for global rendering. To
eliminate artifacts, we add ghost cells at partition boundaries and apply
background masks to remove irrelevant pixels. Benchmarks on the
Richtmyer-Meshkov datasets (about 106.7M Gaussians) show up to 3X speedup
across 8 nodes on Polaris while preserving image quality. These results
demonstrate that distributed 3D-GS enables scalable visualization of
large-scale scientific data and provide a foundation for future in situ
applications.

### 5. [When MoE Meets Blockchain: A Trustworthy Distributed Framework of Large Models](http://arxiv.org/pdf/2509.12141v1)

Authors: Weihao Zhu, Long Shi, Kang Wei, Zhen Mei, Zhe Wang, Jiaheng Wang, Jun Li

As an enabling architecture of Large Models (LMs), Mixture of Experts (MoE)
has become prevalent thanks to its sparsely-gated mechanism, which lowers
computational overhead while maintaining learning performance comparable to
dense LMs. The essence of MoE lies in utilizing a group of neural networks
(called experts) with each specializing in different types of tasks, along with
a trainable gating network that selectively activates a subset of these experts
to handle specific tasks. Traditional cloud-based MoE encounters challenges
such as prolonged response latency, high bandwidth consumption, and data
privacy leakage. To address these issues, researchers have proposed to deploy
MoE over distributed edge networks. However, a key concern of distributed MoE
frameworks is the lack of trust in data interactions among distributed experts
without the surveillance of any trusted authority, and thereby prone to
potential attacks such as data manipulation. In response to the security issues
of traditional distributed MoE, we propose a blockchain-aided trustworthy MoE
(B-MoE) framework that consists of three layers: the edge layer, the blockchain
layer, and the storage layer. In this framework, the edge layer employs the
activated experts downloaded from the storage layer to process the learning
tasks, while the blockchain layer functions as a decentralized trustworthy
network to trace, verify, and record the computational results of the experts
from the edge layer. The experimental results demonstrate that B-MoE is more
robust to data manipulation attacks than traditional distributed MoE during
both the training and inference processes.

### 6. [A Uniqueness Theorem for Distributed Computation under Physical Constraint](http://arxiv.org/pdf/2509.11754v1)

Authors: Zhiyuan Ren, Mingxuan Lu, Wenchi Cheng

Foundational models of computation often abstract away physical hardware
limitations. However, in extreme environments like In-Network Computing (INC),
these limitations become inviolable laws, creating an acute trilemma among
communication efficiency, bounded memory, and robust scalability. Prevailing
distributed paradigms, while powerful in their intended domains, were not
designed for this stringent regime and thus face fundamental challenges. This
paper demonstrates that resolving this trilemma requires a shift in perspective
- from seeking engineering trade-offs to deriving solutions from logical
necessity. We establish a rigorous axiomatic system that formalizes these
physical constraints and prove that for the broad class of computations
admitting an idempotent merge operator, there exists a unique, optimal
paradigm. Any system satisfying these axioms must converge to a single normal
form: Self-Describing Parallel Flows (SDPF), a purely data-centric model where
stateless executors process flows that carry their own control logic. We
further prove this unique paradigm is convergent, Turing-complete, and minimal.
In the same way that the CAP theorem established a boundary for what is
impossible in distributed state management, our work provides a constructive
dual: a uniqueness theorem that reveals what is \textit{inevitable} for
distributed computation flows under physical law.

### 7. [Machine Learning-Driven Predictive Resource Management in Complex Science Workflows](http://arxiv.org/pdf/2509.11512v1)

Authors: Tasnuva Chowdhury, Tadashi Maeno, Fatih Furkan Akman, Joseph Boudreau, Sankha Dutta, Shengyu Feng, Adolfy Hoisie, Kuan-Chieh Hsu, Raees Khan, Jaehyung Kim, Ozgur O. Kilic, Scott Klasky, Alexei Klimentov, Tatiana Korchuganova, Verena Ingrid Martinez Outschoorn, Paul Nilsson, David K. Park, Norbert Podhorszki, Yihui Ren, John Rembrandt Steele, Frédéric Suter, Sairam Sri Vatsavai, Torre Wenaus, Wei Yang, Yiming Yang, Shinjae Yoo

The collaborative efforts of large communities in science experiments, often
comprising thousands of global members, reflect a monumental commitment to
exploration and discovery. Recently, advanced and complex data processing has
gained increasing importance in science experiments. Data processing workflows
typically consist of multiple intricate steps, and the precise specification of
resource requirements is crucial for each step to allocate optimal resources
for effective processing. Estimating resource requirements in advance is
challenging due to a wide range of analysis scenarios, varying skill levels
among community members, and the continuously increasing spectrum of computing
options. One practical approach to mitigate these challenges involves initially
processing a subset of each step to measure precise resource utilization from
actual processing profiles before completing the entire step. While this
two-staged approach enables processing on optimal resources for most of the
workflow, it has drawbacks such as initial inaccuracies leading to potential
failures and suboptimal resource usage, along with overhead from waiting for
initial processing completion, which is critical for fast-turnaround analyses.
In this context, our study introduces a novel pipeline of machine learning
models within a comprehensive workflow management system, the Production and
Distributed Analysis (PanDA) system. These models employ advanced machine
learning techniques to predict key resource requirements, overcoming challenges
posed by limited upfront knowledge of characteristics at each step. Accurate
forecasts of resource requirements enable informed and proactive
decision-making in workflow management, enhancing the efficiency of handling
diverse, complex workflows across heterogeneous resources.

### Digital Libraries

### 1. [Updating the Complex Systems Keyword Diagram Using Collective Feedback and Latest Literature Data](http://arxiv.org/pdf/2509.11997v1)

Authors: Hiroki Sayama

The complex systems keyword diagram generated by the author in 2010 has been
used widely in a variety of educational and outreach purposes, but it
definitely needs a major update and reorganization. This short paper reports
our recent attempt to update the keyword diagram using information collected
from the following multiple sources: (a) collective feedback posted on social
media, (b) recent reference books on complex systems and network science, (c)
online resources on complex systems, and (d) keyword search hits obtained using
OpenAlex, an open-access bibliographic catalogue of scientific publications.
The data (a), (b) and (c) were used to incorporate the research community's
internal perceptions of the relevant topics, whereas the data (d) was used to
obtain more objective measurements of the keywords' relevance and associations
from publications made in complex systems science. Results revealed differences
and overlaps between public perception and actual usage of keywords in
publications on complex systems. Four topical communities were obtained from
the keyword association network, although they were highly intertwined with
each other. We hope that the resulting network visualization of complex systems
keywords provides a more up-to-date, accurate topic map of the field of complex
systems as of today.

### Discrete Mathematics

### 1. [Agglomeration based influential node ranking in path-type networks](http://arxiv.org/pdf/2509.11659v1)

Authors: Zeynep Nihan Berberler, Aysun Asena Kunt

Identification of vital nodes contributes to the research of network
robustness and vulnerability. The most influential nodes are effective in
maximizing the speed and accelerating the information propagation in complex
networks. Identifying and ranking the most influential nodes in complex
networks has not only theoretical but also practical significance in network
analysis since these nodes have a critical influence on the structure and
function of complex networks. This paper is devoted to the evaluating the
importance of nodes and ranking influential nodes in paths and path-type
networks such as comets, double comets, and lollipop networks by network
agglomeration based node contraction method.

### 2. [Statistical Model Checking Beyond Means: Quantiles, CVaR, and the DKW Inequality (extended version)](http://arxiv.org/pdf/2509.11859v1)

Authors: Carlos E. Budde, Arnd Hartmanns, Tobias Meggendorfer, Maximilian Weininger, Patrick Wienhöft

Statistical model checking (SMC) randomly samples probabilistic models to
approximate quantities of interest with statistical error guarantees. It is
traditionally used to estimate probabilities and expected rewards, i.e. means
of different random variables on paths. In this paper, we develop methods using
the Dvoretzky-Kiefer-Wolfowitz-Massart inequality (DKW) to extend SMC beyond
means to compute quantities such as quantiles, conditional value-at-risk, and
entropic risk. The DKW provides confidence bounds on the random variable's
entire cumulative distribution function, a much more versatile guarantee
compared to the statistical methods prevalent in SMC today. We have implemented
support for computing new quantities via the DKW in the 'modes' simulator of
the Modest Toolset. We highlight the implementation and its versatility on
benchmarks from the quantitative verification literature.

### 3. [Foundational theory for optimal decision tree problems. II. Optimal hypersurface decision tree algorithm](http://arxiv.org/pdf/2509.12057v1)

Authors: Xi He

Decision trees are a ubiquitous model for classification and regression tasks
due to their interpretability and efficiency. However, solving the optimal
decision tree (ODT) problem remains a challenging combinatorial optimization
task. Even for the simplest splitting rules--axis-parallel hyperplanes--it is
NP-hard to optimize. In Part I of this series, we rigorously defined the proper
decision tree model through four axioms and, based on these, introduced four
formal definitions of the ODT problem. From these definitions, we derived four
generic algorithms capable of solving ODT problems for arbitrary decision trees
satisfying the axioms. We also analyzed the combinatorial geometric properties
of hypersurfaces, showing that decision trees defined by polynomial
hypersurface splitting rules satisfy the proper axioms that we proposed.
  In this second paper (Part II) of this two-part series, building on the
algorithmic and geometric foundations established in Part I, we introduce the
first hypersurface decision tree (HODT) algorithm. To the best of our
knowledge, existing optimal decision tree methods are, to date, limited to
hyperplane splitting rules--a special case of hypersurfaces--and rely on
general-purpose solvers. In contrast, our HODT algorithm addresses the general
hypersurface decision tree model without requiring external solvers.
  Using synthetic datasets generated from ground-truth hyperplane decision
trees, we vary tree size, data size, dimensionality, and label and feature
noise. Results showing that our algorithm recovers the ground truth more
accurately than axis-parallel trees and exhibits greater robustness to noise.
We also analyzed generalization performance across 30 real-world datasets,
showing that HODT can achieve up to 30% higher accuracy than the
state-of-the-art optimal axis-parallel decision tree algorithm when tree
complexity is properly controlled.

### Data Structures and Algorithms

### 1. [On the Smallest Size of Internal Collage Systems](http://arxiv.org/pdf/2509.11602v1)

Authors: Soichiro Migita, Kyotaro Uehata, Tomohiro I

A Straight-Line Program (SLP) for a stirng $T$ is a context-free grammar in
Chomsky normal form that derives $T$ only, which can be seen as a compressed
form of $T$. Kida et al.\ introduced collage systems [Theor. Comput. Sci.,
2003] to generalize SLPs by adding repetition rules and truncation rules. The
smallest size $c(T)$ of collage systems for $T$ has gained attention to see how
these generalized rules improve the compression ability of SLPs. Navarro et al.
[IEEE Trans. Inf. Theory, 2021] showed that $c(T) \in O(z(T))$ and there is a
string family with $c(T) \in \Omega(b(T) \log |T|)$, where $z(T)$ is the number
of Lempel-Ziv parsing of $T$ and $b(T)$ is the smallest size of bidirectional
schemes for $T$. They also introduced a subclass of collage systems, called
internal collage systems, and proved that its smallest size $\hat{c}(T)$ for
$T$ is at least $b(T)$. While $c(T) \le \hat{c}(T)$ is obvious, it is unknown
how large $\hat{c}(T)$ is compared to $c(T)$. In this paper, we prove that
$\hat{c}(T) = \Theta(c(T))$ by showing that any collage system of size $m$ can
be transformed into an internal collage system of size $O(m)$ in $O(m^2)$ time.
Thanks to this result, we can focus on internal collage systems to study the
asymptotic behavior of $c(T)$, which helps to suppress excess use of truncation
rules. As a direct application, we get $b(T) = O(c(T))$, which answers an open
question posed in [Navarro et al., IEEE Trans. Inf. Theory, 2021]. We also give
a MAX-SAT formulation to compute $\hat{c}(T)$ for a given $T$.

### 2. [An ETH-Tight FPT Algorithm for Rejection-Proof Set Packing with Applications to Kidney Exchange](http://arxiv.org/pdf/2509.11965v1)

Authors: Bart M. P. Jansen, Jeroen S. K. Lamme, Ruben F. A. Verhaegh

We study the parameterized complexity of a recently introduced multi-agent
variant of the Kidney Exchange problem. Given a directed graph $G$ and integers
$d$ and $k$, the standard problem asks whether $G$ contains a packing of
vertex-disjoint cycles, each of length $\leq d$, covering at least $k$ vertices
in total. In the multi-agent setting we consider, the vertex set is partitioned
over several agents who reject a cycle packing as solution if it can be
modified into an alternative packing that covers more of their own vertices. A
cycle packing is called rejection-proof if no agent rejects it and the problem
asks whether such a packing exists that covers at least $k$ vertices.
  We exploit the sunflower lemma on a set packing formulation of the problem to
give a kernel for this $\Sigma_2^P$-complete problem that is polynomial in $k$
for all constant values of $d$. We also provide a $2^{\mathcal{O}(k \log k)} +
n^{\mathcal{O}(1)}$ algorithm based on it and show that this FPT algorithm is
asymptotically optimal under the ETH. Further, we generalize the problem by
including an additional positive integer $c$ in the input that naturally
captures how much agents can modify a given cycle packing to reject it. For
every constant $c$, the resulting problem simplifies from being
$\Sigma_2^P$-complete to NP-complete. With a single-exponential algorithm for
the setting where $c = 1$, we show this to be strictly easier under the ETH
than when $c = 2$. In turn, we show that any $c \geq 2$ yields a problem that
is essentially as hard as the original problem with $c$ unbounded. This
displays an interesting discrepancy between the classical and parameterized
complexity of the problem and gives a good view of what makes it hard.

### 3. [Liar's vertex-edge domination in unit disk graph](http://arxiv.org/pdf/2509.11775v1)

Authors: Debojyoti Bhattacharya, Subhabrata Paul

Let $G=(V, E)$ be a simple undirected graph. A closed neighbourhood of an
edge $e=uv$ between two vertices $u$ and $v$ of $G$, denoted by $N_G[e]$, is
the set of vertices in the neighbourhood of $u$ and $v$ including $\{u,v\}$. A
subset $L$ of $V$ is said to be liar's vertex-edge dominating set if $(i)$ for
every edge $e\in E$, $|N_G[e]\cap L|\geq 2$ and $(ii)$ for every pair of
distinct edges $e,e'$, $|(N_G[e]\cup N_G[e'])\cap L|\geq 3$. The minimum liar's
vertex-edge domination problem is to find the liar's vertex-edge dominating set
of minimum cardinality. In this article, we show that the liar's vertex-edge
domination problem is NP-complete in unit disk graphs, and we design a
polynomial time approximation scheme(PTAS) for the minimum liar's vertex-edge
domination problem in unit disk graphs.

### 4. [Foundational theory for optimal decision tree problems. II. Optimal hypersurface decision tree algorithm](http://arxiv.org/pdf/2509.12057v1)

Authors: Xi He

Decision trees are a ubiquitous model for classification and regression tasks
due to their interpretability and efficiency. However, solving the optimal
decision tree (ODT) problem remains a challenging combinatorial optimization
task. Even for the simplest splitting rules--axis-parallel hyperplanes--it is
NP-hard to optimize. In Part I of this series, we rigorously defined the proper
decision tree model through four axioms and, based on these, introduced four
formal definitions of the ODT problem. From these definitions, we derived four
generic algorithms capable of solving ODT problems for arbitrary decision trees
satisfying the axioms. We also analyzed the combinatorial geometric properties
of hypersurfaces, showing that decision trees defined by polynomial
hypersurface splitting rules satisfy the proper axioms that we proposed.
  In this second paper (Part II) of this two-part series, building on the
algorithmic and geometric foundations established in Part I, we introduce the
first hypersurface decision tree (HODT) algorithm. To the best of our
knowledge, existing optimal decision tree methods are, to date, limited to
hyperplane splitting rules--a special case of hypersurfaces--and rely on
general-purpose solvers. In contrast, our HODT algorithm addresses the general
hypersurface decision tree model without requiring external solvers.
  Using synthetic datasets generated from ground-truth hyperplane decision
trees, we vary tree size, data size, dimensionality, and label and feature
noise. Results showing that our algorithm recovers the ground truth more
accurately than axis-parallel trees and exhibits greater robustness to noise.
We also analyzed generalization performance across 30 real-world datasets,
showing that HODT can achieve up to 30% higher accuracy than the
state-of-the-art optimal axis-parallel decision tree algorithm when tree
complexity is properly controlled.

### 5. [SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation](http://arxiv.org/pdf/2509.12086v1)

Authors: Hui Li, Shiyuan Deng, Xiao Yan, Xiangyu Zhi, James Cheng

Approximate Nearest Neighbor Search (ANNS) plays a critical role in
applications such as search engines, recommender systems, and RAG for LLMs.
Vector quantization (VQ), a crucial technique for ANNS, is commonly used to
reduce space overhead and accelerate distance computations. However, despite
significant research advances, state-of-the-art VQ methods still face
challenges in balancing encoding efficiency and quantization accuracy. To
address these limitations, we propose a novel VQ method called SAQ. To improve
accuracy, SAQ employs a new dimension segmentation technique to strategically
partition PCA-projected vectors into segments along their dimensions. By
prioritizing leading dimension segments with larger magnitudes, SAQ allocates
more bits to high-impact segments, optimizing the use of the available space
quota. An efficient dynamic programming algorithm is developed to optimize
dimension segmentation and bit allocation, ensuring minimal quantization error.
To speed up vector encoding, SAQ devises a code adjustment technique to first
quantize each dimension independently and then progressively refine quantized
vectors using a coordinate-descent-like approach to avoid exhaustive
enumeration. Extensive experiments demonstrate SAQ's superiority over classical
methods (e.g., PQ, PCA) and recent state-of-the-art approaches (e.g., LVQ,
Extended RabitQ). SAQ achieves up to 80% reduction in quantization error and
accelerates encoding speed by over 80x compared to Extended RabitQ.

### Emerging Technologies

### 1. [Vital Signs Monitoring with mmWave OFDM JCAS System](http://arxiv.org/pdf/2509.11767v1)

Authors: Jakub Dobosz, Maximilian Engelhardt, Diego Dupleich, Maciej Stapor, Pawel Kulakowski

Wireless techniques for monitoring human vital signs, such as heart and
breathing rates, offer a promising solution in the context of joint
communication and sensing (JCAS) with applications in medicine, sports, safety,
security, and even the military. This paper reports experimental results
obtained at the Fraunhofer Institute for Integrated Circuits in Ilmenau,
demonstrating the effectiveness of an indoor orthogonal frequency-division
multiplexing (OFDM) JCAS system for detecting human heart and breathing rates.
The system operated in a bistatic configuration at an FR2 frequency of 26.5 GHz
with a variable bandwidth of up to 1 GHz. Measurements were taken under various
scenarios, including a subject lying down, sitting, or walking, in both
line-of-sight and non-line-of-sight conditions, and with one or two subjects
present simultaneously. The results indicate that while vital sign detection is
generally feasible, its effectiveness is influenced by several factors, such as
the subjects clothing, activity, as well as the distance and angle relative to
the sensing system. In addition, no significant influence of bandwidth was
detected since the vital signs information is encoded in the phase of the
signal.

### 2. [Regulating Ride-Sourcing Markets: Can Minimum Wage Regulation Protect Drivers Without Disrupting the Market?](http://arxiv.org/pdf/2509.11845v1)

Authors: Farnoud Ghasemi, Arjan de Ruijter, Rafal Kucharski, Oded Cats

Ride-sourcing platforms such as Uber and Lyft are prime examples of the gig
economy, recruiting drivers as independent contractors, thereby avoiding legal
and fiscal obligations. Although platforms offer flexibility in choosing work
shifts and areas, many drivers experience low income and poor working
conditions, leading to widespread strikes and protests. Minimum wage regulation
is adopted to improve drivers welfare. However, the impacts of this regulation
on drivers as well as on travelers and platforms, remain largely unknown. While
ride-sourcing platforms do not disclose the relevant data, state-of-the-art
models fail to explain the effects of minimum wage regulation on market
dynamics. In this study, we assess the effectiveness and implications of
minimum wage regulation in ride-sourcing markets while simulating the detailed
dynamics of ride-sourcing markets under varying regulation intensities, both
with and without the so-called platform lockout strategy. Our findings reveal
that minimum wage regulation impacts substantially drivers income, and may lead
to higher fares for travelers and threaten platforms survival. When platforms
adopt a lockout strategy, their profitability significantly improves and
drivers earn more, although many others lose their jobs, and service level for
travelers consequently declines.

### 3. [HiPARS: Highly-Parallel Atom Rearrangement Sequencer](http://arxiv.org/pdf/2509.12083v1)

Authors: Jonas Winklmann, Martin Schulz

Neutral atom quantum computing's great scaling potential has resulted in it
emerging as a popular modality in recent years. For state preparation, atoms
are loaded stochastically and have to be detected and rearranged at runtime to
create a predetermined initial configuration for circuit execution. Such
rearrangement schemes either suffer from low parallelizability for
acousto-optic deflector (AOD)-based approaches or are comparatively slow in
case of spatial light modulators (SLMs). In our work, we introduce an algorithm
that can improve the parallelizability of the former. Since the transfer of
atoms from static SLM traps to AOD-generated movable traps is detrimental both
in terms of atom loss rates and execution time, our approach is based on
highly-parallel composite moves where many atoms are picked up simultaneously
and maneuvered into target positions that may be comparatively distant. We see
that our algorithm outperforms its alternatives for near-term devices with up
to around 1000 qubits and has the potential to scale up to several thousand
with further optimizations.

### 4. [Cross-Platform Scaling of Vision-Language-Action Models from Edge to Cloud GPUs](http://arxiv.org/pdf/2509.11480v1)

Authors: Amir Taherin, Juyi Lin, Arash Akbari, Arman Akbari, Pu Zhao, Weiwei Chen, David Kaeli, Yanzhi Wang

Vision-Language-Action (VLA) models have emerged as powerful generalist
policies for robotic control, yet their performance scaling across model
architectures and hardware platforms, as well as their associated power
budgets, remain poorly understood. This work presents an evaluation of five
representative VLA models -- spanning state-of-the-art baselines and two newly
proposed architectures -- targeting edge and datacenter GPU platforms. Using
the LIBERO benchmark, we measure accuracy alongside system-level metrics,
including latency, throughput, and peak memory usage, under varying edge power
constraints and high-performance datacenter GPU configurations. Our results
identify distinct scaling trends: (1) architectural choices, such as action
tokenization and model backbone size, strongly influence throughput and memory
footprint; (2) power-constrained edge devices exhibit non-linear performance
degradation, with some configurations matching or exceeding older datacenter
GPUs; and (3) high-throughput variants can be achieved without significant
accuracy loss. These findings provide actionable insights when selecting and
optimizing VLAs across a range of deployment constraints. Our work challenges
current assumptions about the superiority of datacenter hardware for robotic
inference.

### Formal Languages and Automata Theory

### 1. [A Unifying Approach to Picture Automata](http://arxiv.org/pdf/2509.12077v1)

Authors: Yvo Ad Meeres, František Mráz

A directed acyclic graph (DAG) can represent a two-dimensional string or
picture. We propose recognizing picture languages using DAG automata by
encoding 2D inputs into DAGs. An encoding can be input-agnostic (based on input
size only) or input-driven (depending on symbols). Three distinct
input-agnostic encodings characterize classes of picture languages accepted by
returning finite automata, boustrophedon automata, and online tessellation
automata. Encoding a string as a simple directed path limits recognition to
regular languages. However, input-driven encodings allow DAG automata to
recognize some context-sensitive string languages and outperform online
tessellation automata in two dimensions.

### General Literature

### 1. [Nuclear Beavers](http://arxiv.org/pdf/2509.12055v1)

Authors: Joshua Wylie, Pablo Giuliani, Kyle Godbey, Sylvester Agbemava

Nuclear physics is a very abstract field with little accessibility for wider
audiences, and yet it is a field of physics with far reaching implications for
everyday life. The Nuclear Beavers demonstration is a hands-on experience that
offers an intuitive lens into nuclear structure and decay. We aim to provide a
more accessible entry point for students and educators by substituting complex
nuclear structures and interactions with tactile building blocks following
well-defined rules, thereby opening nuclear physics concepts to the general
public.

### Graphics

### 1. [HoloGarment: 360° Novel View Synthesis of In-the-Wild Garments](http://arxiv.org/pdf/2509.12187v1)

Authors: Johanna Karras, Yingwei Li, Yasamin Jafarian, Ira Kemelmacher-Shlizerman

Novel view synthesis (NVS) of in-the-wild garments is a challenging task due
significant occlusions, complex human poses, and cloth deformations. Prior
methods rely on synthetic 3D training data consisting of mostly unoccluded and
static objects, leading to poor generalization on real-world clothing. In this
paper, we propose HoloGarment (Hologram-Garment), a method that takes 1-3
images or a continuous video of a person wearing a garment and generates
360{\deg} novel views of the garment in a canonical pose. Our key insight is to
bridge the domain gap between real and synthetic data with a novel implicit
training paradigm leveraging a combination of large-scale real video data and
small-scale synthetic 3D data to optimize a shared garment embedding space.
During inference, the shared embedding space further enables dynamic
video-to-360{\deg} NVS through the construction of a garment "atlas"
representation by finetuning a garment embedding on a specific real-world
video. The atlas captures garment-specific geometry and texture across all
viewpoints, independent of body pose or motion. Extensive experiments show that
HoloGarment achieves state-of-the-art performance on NVS of in-the-wild
garments from images and videos. Notably, our method robustly handles
challenging real-world artifacts -- such as wrinkling, pose variation, and
occlusion -- while maintaining photorealism, view consistency, fine texture
details, and accurate geometry. Visit our project page for additional results:
https://johannakarras.github.io/HoloGarment

### Computer Science and Game Theory

### 1. [Nash Equilibrium and Belief Evolution in Differential Games](http://arxiv.org/pdf/2509.11739v1)

Authors: Jiangjing Zhou, Ovanes Petrosian, Ye Zhang, Hongwei Gao

This study investigates differential games with motion-payoff uncertainty in
continuous-time settings. We propose a framework where players update their
beliefs about uncertain parameters using continuous Bayesian updating.
Theoretical proofs leveraging key probability theorems demonstrate that
players' beliefs converge to the true parameter values, ensuring stability and
accuracy in long-term estimations. We further derive Nash Equilibrium
strategies with continuous Bayesian updating for players, emphasizing the role
of belief updates in decision-making processes. Additionally, we establish the
convergence of Nash Equilibrium strategies with continuous Bayesian updating.
The efficacy of both continuous and dynamic Bayesian updating is examined in
the context of pollution control games, showing convergence in players'
estimates under small time intervals in discrete scenarios.

### Human-Computer Interaction

### 1. [BioMetaphor: AI-Generated Biodata Representations for Virtual Co-Present Events](http://arxiv.org/pdf/2509.11600v1)

Authors: Lin Lin, Ming Wu, Anyu Ren, Zhanwei Wu, Daojun Gong, Ruowei Xiao

In virtual or hybrid co-present events, biodata is emerging as a new paradigm
of social cues. While it is able to reveal individuals' inner states, the
technology-mediated representation of biodata in social contexts remains
underexplored. This study aims to uncover human cognitive preferences and
patterns for biodata expression and leverage this knowledge to guide generative
AI (GenAI) in creating biodata representations for co-present experiences,
aligning with the broader concept of Human-in-the-loop. We conducted a user
elicitation workshop with 30 HCI experts and investigated the results using
qualitative analysis. Based on our findings, we further propose a GenAI-driven
framework: BioMetaphor. Our framework demonstration shows that current GenAI
can learn and express visual biodata cues in an event-adpated, human-like
manner. This human-centered approach engages users in research, revealing the
underlying cognition constructions for biodata expression while demonstrating
how such knowledge can inform the design and development of future empathic
technologies.

### 2. [Robots that Evolve with Us: Modular Co-Design for Personalization, Adaptability, and Sustainability](http://arxiv.org/pdf/2509.11622v1)

Authors: Lingyun Chen, Qing Xiao, Zitao Zhang, Eli Blevis, Selma Šabanović

Many current robot designs prioritize efficiency and one-size-fits-all
solutions, oftentimes overlooking personalization, adaptability, and
sustainability. To explore alternatives, we conducted two co-design workshops
with 23 participants, who engaged with a modular robot co-design framework.
Using components we provided as building blocks, participants combined,
removed, and invented modules to envision how modular robots could accompany
them from childhood through adulthood and into older adulthood. The
participants' designs illustrate how modularity (a) enables personalization
through open-ended configuration, (b) adaptability across shifting life-stage
needs, and (c) sustainability through repair, reuse, and continuity. We
therefore derive design principles that establish modularity as a foundation
for lifespan-oriented human-robot interaction. This work reframes modular
robotics as a flexible and expressive co-design approach, supporting robots
that evolve with people, rather than static products optimized for single
moments or contexts of use.

### 3. [Colour Perception in Immersive Virtual Reality: Emotional and Physiological Responses to Fifteen Munsell Hues](http://arxiv.org/pdf/2509.11644v1)

Authors: Francesco Febbraio, Simona Collina, Christina Lepida, Panagiotis Kourtesis

Colour is a fundamental determinant of affective experience in immersive
virtual reality (VR), yet the emotional and physiological impact of individual
hues remains poorly characterised. This study investigated how fifteen
calibrated Munsell hues influence subjective and autonomic responses when
presented in immersive VR. Thirty-six adults (18-45 years) viewed each hue in a
within-subject design while pupil diameter and skin conductance were recorded
continuously, and self-reported emotions were assessed using the
Self-Assessment Manikin across pleasure, arousal, and dominance.
Repeated-measures ANOVAs revealed robust hue effects on all three self-report
dimensions and on pupil dilation, with medium to large effect sizes. Reds and
red-purple hues elicited the highest arousal and dominance, whereas blue-green
hues were rated most pleasurable. Pupil dilation closely tracked arousal
ratings, while skin conductance showed no reliable hue differentiation, likely
due to the brief (30 s) exposures. Individual differences in cognitive style
and personality modulated overall reactivity but did not alter the relative
ranking of hues. Taken together, these findings provide the first systematic
hue-by-hue mapping of affective and physiological responses in immersive VR.
They demonstrate that calibrated colour shapes both experience and ocular
physiology, while also offering practical guidance for educational, clinical,
and interface design in virtual environments.

### 4. [See What I Mean? Mobile Eye-Perspective Rendering for Optical See-through Head-mounted Displays](http://arxiv.org/pdf/2509.11653v1)

Authors: Gerlinde Emsenhuber, Tobias Langlotz, Denis Kalkofen, Markus Tatzgern

Image-based scene understanding allows Augmented Reality systems to provide
contextual visual guidance in unprepared, real-world environments. While
effective on video see-through (VST) head-mounted displays (HMDs), such methods
suffer on optical see-through (OST) HMDs due to misregistration between the
world-facing camera and the user's eye perspective. To approximate the user's
true eye view, we implement and evaluate three software-based eye-perspective
rendering (EPR) techniques on a commercially available, untethered OST HMD
(Microsoft HoloLens 2): (1) Plane-Proxy EPR, projecting onto a fixed-distance
plane; (2) Mesh-Proxy EPR, using SLAM-based reconstruction for projection; and
(3) Gaze-Proxy EPR, a novel eye-tracking-based method that aligns the
projection with the user's gaze depth. A user study on real-world tasks
underscores the importance of accurate EPR and demonstrates gaze-proxy as a
lightweight alternative to geometry-based methods. We release our EPR framework
as open source.

### 5. [Lost in Data: How Older Adults Perceive and Navigate Health Data Representations](http://arxiv.org/pdf/2509.11876v1)

Authors: Peterson Jean, Emma Murphy, Enda Bates

As the ageing population grows, older adults increasingly rely on wearable
devices to monitor chronic conditions. However, conventional health data
representations (HDRs) often present accessibility challenges, particularly for
critical health parameters like blood pressure and sleep data. This study
explores how older adults interact with these representations, identifying key
barriers such as semantic inconsistency and difficulties in understanding.
While research has primarily focused on data collection, less attention has
been given to how information is output and understood by end-users. To address
this, an end-user evaluation was conducted with 16 older adults (65+) in a
structured workshop, using think-aloud protocols and participatory design
activities. The findings highlight the importance of affordance and familiarity
in improving accessibility, emphasising the familiarity and potential of
multimodal cues. This study bridges the gap between domain experts and
end-users, providing a replicable methodological approach for designing
intuitive, multisensory HDRs that better align with older adults' needs and
abilities.

### 6. [Generative AI in Game Development: A Qualitative Research Synthesis](http://arxiv.org/pdf/2509.11898v1)

Authors: Alexandru Ternar, Alena Denisova, João M. Cunha, Annakaisa Kultima, Christian Guckelsberger

Generative Artificial Intelligence (GenAI) has had a tremendous impact on
game production and promises lasting transformations. In the last five years
since GenAI's inception, several studies, typically via qualitative methods,
have explored its impact on game production from different settings and
demographic angles. However, these studies often contextualise and consolidate
their findings weakly with related work, and a big picture view is still
missing. Here, we aim to provide such a view of GenAI's impact on game
production in the form of a qualitative research synthesis via
meta-ethnography. We followed PRISMA-S to systematically search the relevant
literature from 2020-2025, including major HCI and games research databases. We
then synthesised the 10 eligible studies, conducting reciprocal translation and
line-of-argument synthesis guided by eMERGe, informed by CASP quality
appraisal. We identified nine overarching themes, provide recommendations, and
contextualise our insights in wider game production trends.

### 7. [PrivWeb: Unobtrusive and Content-aware Privacy Protection For Web Agents](http://arxiv.org/pdf/2509.11939v1)

Authors: Shuning Zhang, Yutong Jiang, Rongjun Ma, Yuting Yang, Mingyao Xu, Zhixin Huang, Xin Yi, Hewu Li

While web agents gained popularity by automating web interactions, their
requirement for interface access introduces significant privacy risks that are
understudied, particularly from users' perspective. Through a formative study
(N=15), we found users frequently misunderstand agents' data practices, and
desired unobtrusive, transparent data management. To achieve this, we designed
and implemented PrivWeb, a trusted add-on on web agents that utilizes a
localized LLM to anonymize private information on interfaces according to user
preferences. It features privacy categorization schema and adaptive
notifications that selectively pauses tasks for user control over information
collection for highly sensitive information, while offering non-disruptive
options for less sensitive information, minimizing human oversight. The user
study (N=14) across travel, information retrieval, shopping, and entertainment
tasks compared PrivWeb with baselines without notification and without control
for private information access, where PrivWeb reduced perceived privacy risks
with no associated increase in cognitive effort, and resulted in higher overall
satisfaction.

### 8. [Teaching the Teachers: Building Generative AI Literacy in Higher Ed Instructors](http://arxiv.org/pdf/2509.11999v1)

Authors: Si Chen, Xiuxiu Tang, Alison Cheng, Nitesh Chawla, G. Alex Ambrose, Ronald Metoyer

Generative AI is reshaping higher education, yet research has focused largely
on students, while instructors remain understudied despite their central role
in mediating adoption and modeling responsible use. We present the \textit{AI
Academy}, a faculty development program that combined AI exploration with
pedagogical reflection and peer learning. Rather than a course evaluated for
outcomes, the Academy provided a setting to study how instructors build AI
literacies in relation to tools, policies, peer practices, and institutional
supports. We studied 25 instructors through pre/post surveys, learning logs,
and facilitator interviews. Findings show AI literacy gains alongside new
insights. We position instructors as designers of responsible AI practices and
contribute a replicable program model, a co-constructed survey instrument, and
design insights for professional development that adapts to evolving tools and
fosters ethical discussion.

### 9. [Exploring Gaze Dynamics in VR Film Education: Gender, Avatar, and the Shift Between Male and Female Perspectives](http://arxiv.org/pdf/2509.12027v1)

Authors: Zheng Wei, Jia Sun, Junxiang Liao, Lik-Hang Lee, Pan Hui, Huamin Qu, Wai Tong, Xian Xu

In virtual reality (VR) education, especially in creative fields like film
production, avatar design and narrative style extend beyond appearance and
aesthetics. This study explores how the interaction between avatar gender, the
dominant narrative actor's gender, and the learner's gender influences film
production learning in VR, focusing on gaze dynamics and gender perspectives.
Using a 2*2*2 experimental design, 48 participants operated avatars of
different genders and interacted with male or female-dominant narratives. The
results show that the consistency between the avatar and gender affects
presence, and learners' control over the avatar is also influenced by gender
matching. Learners using avatars of the opposite gender reported stronger
control, suggesting gender incongruity prompted more focus on the avatar.
Additionally, female participants with female avatars were more likely to adopt
a "female gaze," favoring soft lighting and emotional shots, while male
participants with male avatars were more likely to adopt a "male gaze,"
choosing dynamic shots and high contrast. When male participants used female
avatars, they favored "female gaze," while female participants with male
avatars focused on "male gaze". These findings advance our understanding of how
avatar design and narrative style in VR-based education influence creativity
and the cultivation of gender perspectives, and they offer insights for
developing more inclusive and diverse VR teaching tools going forward.

### 10. [You Are Not Alone: Designing Body Doubling for ADHD in Virtual Reality](http://arxiv.org/pdf/2509.12153v1)

Authors: Zinat Ara, Imtiaz Bin Rahim, Puqi Zhou, Liuchuan Yu, Behzad Esmaeili, Lap-Fai Yu, Sungsoo Ray Hong

Adults with Attention Deficit Hyperactivity Disorder (ADHD) experience
challenges sustaining attention in the workplace. Body doubling, the concept of
working alongside another person, has been proposed as a productivity aid for
ADHD and other neurodivergent populations (NDs). However, prior work found no
conclusive effectiveness and noted NDs' discomfort with social presence. This
work investigates body doubling as an ADHD centered productivity strategy in
construction tasks. In Study 1, we explored challenges ADHD workers face in
construction and identified design insights. In Study 2, we implemented a
virtual reality bricklaying task under three conditions: (C1) alone, (C2) with
a human body double, and (C3) with an AI body double. Results from 12
participants show they finished tasks faster and perceived greater accuracy and
sustained attention in C2 and C3 compared to C1. While body doubling was
clearly preferred, opinions diverged between conditions. Our findings verify
its effect and offer design implications for future interventions.

### Information Retrieval

### 1. [Decoding in Latent Spaces for Efficient Inference in LLM-based Recommendation](http://arxiv.org/pdf/2509.11524v1)

Authors: Chengbing Wang, Yang Zhang, Zhicheng Wang, Tianhao Shi, Keqin Bao, Fuli Feng, Tat-Seng Chua

Fine-tuning large language models (LLMs) for recommendation in a generative
manner has delivered promising results, but encounters significant inference
overhead due to autoregressive decoding in the language space. This work
explores bypassing language-space decoding by directly matching candidate items
with the LLM's internal thought representations in the latent space,
eliminating the time-consuming autoregressive process to reduce computational
costs. Towards this, we introduce Light Latent-space Decoding (L2D), an
effective and efficient latent-space decoding method. L2D represents
user-preferred items by using the hidden states of test sequences reflecting
the LLM's internal thought, and obtains candidate item representations from the
hidden states of training sequences labeled with the corresponding candidate
items. It then matches the two types of representations to decode items,
achieving latent-space decoding. In this way, it enables efficient decoding
without altering the LLM's generative tuning paradigm, thereby preserving
performance. Extensive empirical results demonstrate that L2D is more than 10x
faster than language-space decoding while maintaining or enhancing performance.

### 2. [AEFS: Adaptive Early Feature Selection for Deep Recommender Systems](http://arxiv.org/pdf/2509.12076v1)

Authors: Fan Hu, Gaofeng Lu, Jun Chen, Chaonan Guo, Yuekui Yang, Xirong Li

Feature selection has emerged as a crucial technique in refining recommender
systems. Recent advancements leveraging Automated Machine Learning (AutoML) has
drawn significant attention, particularly in two main categories: early feature
selection and late feature selection, differentiated by whether the selection
occurs before or after the embedding layer. The early feature selection selects
a fixed subset of features and retrains the model, while the late feature
selection, known as adaptive feature selection, dynamically adjusts feature
choices for each data instance, recognizing the variability in feature
significance. Although adaptive feature selection has shown remarkable
improvements in performance, its main drawback lies in its post-embedding layer
feature selection. This process often becomes cumbersome and inefficient in
large-scale recommender systems with billions of ID-type features, leading to a
highly sparse and parameter-heavy embedding layer. To overcome this, we
introduce Adaptive Early Feature Selection (AEFS), a very simple method that
not only adaptively selects informative features for each instance, but also
significantly reduces the activated parameters of the embedding layer. AEFS
employs a dual-model architecture, encompassing an auxiliary model dedicated to
feature selection and a main model responsible for prediction. To ensure
effective alignment between these two models, we incorporate two collaborative
training loss constraints. Our extensive experiments on three benchmark
datasets validate the efficiency and effectiveness of our approach. Notably,
AEFS matches the performance of current state-of-theart Adaptive Late Feature
Selection methods while achieving a significant reduction of 37. 5% in the
activated parameters of the embedding layer. AEFS is open-source at
https://github. com/fly-dragon211/AEFS .

### 3. [Results of the 2025 Video Browser Showdown](http://arxiv.org/pdf/2509.12000v1)

Authors: Luca Rossetto, Klaus Schoeffmann, Cathal Gurrin, Jakub Lokoč, Werner Bailer

This report presents the results of the 14th Video Browser Showdown, held at
the 2025 International Conference on Multimedia Modeling on the 8th of January
2025 in Nara, Japan.

### 4. [Data-Driven Analysis of Text-Conditioned AI-Generated Music: A Case Study with Suno and Udio](http://arxiv.org/pdf/2509.11824v1)

Authors: Luca Casini, Laura Cros Vila, David Dalmazzo, Anna-Kaisa Kaila, Bob L. T. Sturm

Online AI platforms for creating music from text prompts (AI music), such as
Suno and Udio, are now being used by hundreds of thousands of users. Some AI
music is appearing in advertising, and even charting, in multiple countries.
How are these platforms being used? What subjects are inspiring their users?
This article answers these questions for Suno and Udio using a large collection
of songs generated by users of these platforms from May to October 2024. Using
a combination of state-of-the-art text embedding models, dimensionality
reduction and clustering methods, we analyze the prompts, tags and lyrics, and
automatically annotate and display the processed data in interactive plots. Our
results reveal prominent themes in lyrics, language preference, prompting
strategies, as well as peculiar attempts at steering models through the use of
metatags. To promote the musicological study of the developing cultural
practice of AI-generated music we share our code and resources.

### 5. [SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation](http://arxiv.org/pdf/2509.12086v1)

Authors: Hui Li, Shiyuan Deng, Xiao Yan, Xiangyu Zhi, James Cheng

Approximate Nearest Neighbor Search (ANNS) plays a critical role in
applications such as search engines, recommender systems, and RAG for LLMs.
Vector quantization (VQ), a crucial technique for ANNS, is commonly used to
reduce space overhead and accelerate distance computations. However, despite
significant research advances, state-of-the-art VQ methods still face
challenges in balancing encoding efficiency and quantization accuracy. To
address these limitations, we propose a novel VQ method called SAQ. To improve
accuracy, SAQ employs a new dimension segmentation technique to strategically
partition PCA-projected vectors into segments along their dimensions. By
prioritizing leading dimension segments with larger magnitudes, SAQ allocates
more bits to high-impact segments, optimizing the use of the available space
quota. An efficient dynamic programming algorithm is developed to optimize
dimension segmentation and bit allocation, ensuring minimal quantization error.
To speed up vector encoding, SAQ devises a code adjustment technique to first
quantize each dimension independently and then progressively refine quantized
vectors using a coordinate-descent-like approach to avoid exhaustive
enumeration. Extensive experiments demonstrate SAQ's superiority over classical
methods (e.g., PQ, PCA) and recent state-of-the-art approaches (e.g., LVQ,
Extended RabitQ). SAQ achieves up to 80% reduction in quantization error and
accelerates encoding speed by over 80x compared to Extended RabitQ.

### Machine Learning

### 1. [DARD: Dice Adversarial Robustness Distillation against Adversarial Attacks](http://arxiv.org/pdf/2509.11525v1)

Authors: Jing Zou, Shungeng Zhang, Meikang Qiu, Chong Li

Deep learning models are vulnerable to adversarial examples, posing critical
security challenges in real-world applications. While Adversarial Training (AT
) is a widely adopted defense mechanism to enhance robustness, it often incurs
a trade-off by degrading performance on unperturbed, natural data. Recent
efforts have highlighted that larger models exhibit enhanced robustness over
their smaller counterparts. In this paper, we empirically demonstrate that such
robustness can be systematically distilled from large teacher models into
compact student models. To achieve better performance, we introduce Dice
Adversarial Robustness Distillation (DARD), a novel method designed to transfer
robustness through a tailored knowledge distillation paradigm. Additionally, we
propose Dice Projected Gradient Descent (DPGD), an adversarial example
generalization method optimized for effective attack. Our extensive experiments
demonstrate that the DARD approach consistently outperforms adversarially
trained networks with the same architecture, achieving superior robustness and
standard accuracy.

### 2. [Compressed Sensing: Mathematical Foundations, Implementation, and Advanced Optimization Techniques](http://arxiv.org/pdf/2509.11550v1)

Authors: Shane Stevenson, Maryam Sabagh

Compressed sensing is a signal processing technique that allows for the
reconstruction of a signal from a small set of measurements. The key idea
behind compressed sensing is that many real-world signals are inherently
sparse, meaning that they can be efficiently represented in a different space
with only a few components compared to their original space representation. In
this paper we will explore the mathematical formulation behind compressed
sensing, its logic and pathologies, and apply compressed sensing to real world
signals.

### 3. [Topology Structure Optimization of Reservoirs Using GLMY Homology](http://arxiv.org/pdf/2509.11612v1)

Authors: Yu Chen, Shengwei Wang, Hongwei Lin

Reservoir is an efficient network for time series processing. It is well
known that network structure is one of the determinants of its performance.
However, the topology structure of reservoirs, as well as their performance, is
hard to analyzed, due to the lack of suitable mathematical tools. In this
paper, we study the topology structure of reservoirs using persistent GLMY
homology theory, and develop a method to improve its performance. Specifically,
it is found that the reservoir performance is closely related to the
one-dimensional GLMY homology groups. Then, we develop a reservoir structure
optimization method by modifying the minimal representative cycles of
one-dimensional GLMY homology groups. Finally, by experiments, it is validated
that the performance of reservoirs is jointly influenced by the reservoir
structure and the periodicity of the dataset.

### 4. [Adaptive-GraphSketch: Real-Time Edge Anomaly Detection via Multi-Layer Tensor Sketching and Temporal Decay](http://arxiv.org/pdf/2509.11633v1)

Authors: Ocheme Anthony Ekle, William Eberle

Anomaly detection in dynamic graphs is essential for identifying malicious
activities, fraud, and unexpected behaviors in real-world systems such as
cybersecurity and power grids. However, existing approaches struggle with
scalability, probabilistic interpretability, and adaptability to evolving
traffic patterns. In this paper, we propose ADAPTIVE-GRAPHSKETCH, a lightweight
and scalable framework for real-time anomaly detection in streaming edge data.
Our method integrates temporal multi-tensor sketching with Count-Min Sketch
using Conservative Update (CMS-CU) to compactly track edge frequency patterns
with bounded memory, while mitigating hash collision issues. We incorporate
Bayesian inference for probabilistic anomaly scoring and apply Exponentially
Weighted Moving Average (EWMA) for adaptive thresholding tuned to burst
intensity. Extensive experiments on four real-world intrusion detection
datasets demonstrate that ADAPTIVE-GRAPHSKETCH outperforms state-of-the-art
baselines such as ANOEDGE-G/L, MIDAS-R, and F-FADE, achieving up to 6.5% AUC
gain on CIC-IDS2018 and up to 15.6% on CIC-DDoS2019, while processing 20
million edges in under 3.4 seconds using only 10 hash functions. Our results
show that ADAPTIVE-GRAPHSKETCH is practical and effective for fast, accurate
anomaly detection in large-scale streaming graphs.
  Keywords: Anomaly Detection, Streaming, Real-time, Dynamic Graphs, Edge
Streams, Tensor Sketching

### 5. [Assessing On-the-Ground Disaster Impact Using Online Data Sources](http://arxiv.org/pdf/2509.11634v1)

Authors: Saketh Vishnubhatla, Ujun Jeong, Bohan Jiang, Paras Sheth, Zhen Tan, Adrienne Raglin, Huan Liu

Assessing the impact of a disaster in terms of asset losses and human
casualties is essential for preparing effective response plans. Traditional
methods include offline assessments conducted on the ground, where volunteers
and first responders work together to collect the estimate of losses through
windshield surveys or on-ground inspection. However, these methods have a time
delay and are prone to different biases. Recently, various online data sources,
including social media, news reports, aerial imagery, and satellite data, have
been utilized to evaluate the impact of disasters. Online data sources provide
real-time data streams for estimating the offline impact. Limited research
exists on how different online sources help estimate disaster impact at a given
administrative unit. In our work, we curate a comprehensive dataset by
collecting data from multiple online sources for a few billion-dollar disasters
at the county level. We also analyze how online estimates compare with
traditional offline-based impact estimates for the disaster. Our findings
provide insight into how different sources can provide complementary
information to assess the disaster.

### 6. [An Interventional Approach to Real-Time Disaster Assessment via Causal Attribution](http://arxiv.org/pdf/2509.11676v1)

Authors: Saketh Vishnubhatla, Alimohammad Beigi, Rui Heng Foo, Umang Goel, Ujun Jeong, Bohan Jiang, Adrienne Raglin, Huan Liu

Traditional disaster analysis and modelling tools for assessing the severity
of a disaster are predictive in nature. Based on the past observational data,
these tools prescribe how the current input state (e.g., environmental
conditions, situation reports) results in a severity assessment. However, these
systems are not meant to be interventional in the causal sense, where the user
can modify the current input state to simulate counterfactual "what-if"
scenarios. In this work, we provide an alternative interventional tool that
complements traditional disaster modelling tools by leveraging real-time data
sources like satellite imagery, news, and social media. Our tool also helps
understand the causal attribution of different factors on the estimated
severity, over any given region of interest. In addition, we provide actionable
recourses that would enable easier mitigation planning. Our source code is
publicly available.

### 7. [Fast and Interpretable Machine Learning Modelling of Atmospheric Molecular Clusters](http://arxiv.org/pdf/2509.11728v1)

Authors: Lauri Seppäläinen, Jakub Kubečka, Jonas Elm, Kai Puolamäki

Understanding how atmospheric molecular clusters form and grow is key to
resolving one of the biggest uncertainties in climate modelling: the formation
of new aerosol particles. While quantum chemistry offers accurate insights into
these early-stage clusters, its steep computational costs limit large-scale
exploration. In this work, we present a fast, interpretable, and surprisingly
powerful alternative: $k$-nearest neighbour ($k$-NN) regression model. By
leveraging chemically informed distance metrics, including a kernel-induced
metric and one learned via metric learning for kernel regression (MLKR), we
show that simple $k$-NN models can rival more complex kernel ridge regression
(KRR) models in accuracy, while reducing computational time by orders of
magnitude. We perform this comparison with the well-established
Faber-Christensen-Huang-Lilienfeld (FCHL19) molecular descriptor, but other
descriptors (e.g., FCHL18, MBDF, and CM) can be shown to have similar
performance. Applied to both simple organic molecules in the QM9 benchmark set
and large datasets of atmospheric molecular clusters (sulphuric acid-water and
sulphuric-multibase -base systems), our $k$-NN models achieve near-chemical
accuracy, scale seamlessly to datasets with over 250,000 entries, and even
appears to extrapolate to larger unseen clusters with minimal error (often
nearing 1 kcal/mol). With built-in interpretability and straightforward
uncertainty estimation, this work positions $k$-NN as a potent tool for
accelerating discovery in atmospheric chemistry and beyond.

### 8. [Stabilizing PINNs: A regularization scheme for PINN training to avoid unstable fixed points of dynamical systems](http://arxiv.org/pdf/2509.11768v1)

Authors: Milos Babic, Franz M. Rohrhofer, Bernhard C. Geiger

It was recently shown that the loss function used for training
physics-informed neural networks (PINNs) exhibits local minima at solutions
corresponding to fixed points of dynamical systems. In the forward setting,
where the PINN is trained to solve initial value problems, these local minima
can interfere with training and potentially leading to physically incorrect
solutions. Building on stability theory, this paper proposes a regularization
scheme that penalizes solutions corresponding to unstable fixed points.
Experimental results on four dynamical systems, including the Lotka-Volterra
model and the van der Pol oscillator, show that our scheme helps avoiding
physically incorrect solutions and substantially improves the training success
rate of PINNs.

### 9. [Watch Your Step: A Cost-Sensitive Framework for Accelerometer-Based Fall Detection in Real-World Streaming Scenarios](http://arxiv.org/pdf/2509.11789v1)

Authors: Timilehin B. Aderinola, Luca Palmerini, Ilaria D'Ascanio, Lorenzo Chiari, Jochen Klenk, Clemens Becker, Brian Caulfield, Georgiana Ifrim

Real-time fall detection is crucial for enabling timely interventions and
mitigating the severe health consequences of falls, particularly in older
adults. However, existing methods often rely on simulated data or assumptions
such as prior knowledge of fall events, limiting their real-world
applicability. Practical deployment also requires efficient computation and
robust evaluation metrics tailored to continuous monitoring. This paper
presents a real-time fall detection framework for continuous monitoring without
prior knowledge of fall events. Using over 60 hours of inertial measurement
unit (IMU) data from the FARSEEING real-world falls dataset, we employ recent
efficient classifiers to compute fall probabilities in streaming mode. To
enhance robustness, we introduce a cost-sensitive learning strategy that tunes
the decision threshold using a cost function reflecting the higher risk of
missed falls compared to false alarms. Unlike many methods that achieve high
recall only at the cost of precision, our framework achieved Recall of 1.00,
Precision of 0.84, and an F1 score of 0.91 on FARSEEING, detecting all falls
while keeping false alarms low, with average inference time below 5 ms per
sample. These results demonstrate that cost-sensitive threshold tuning enhances
the robustness of accelerometer-based fall detection. They also highlight the
potential of our computationally efficient framework for deployment in
real-time wearable sensor systems for continuous monitoring.

### 10. [Visualization and Analysis of the Loss Landscape in Graph Neural Networks](http://arxiv.org/pdf/2509.11792v1)

Authors: Samir Moustafa, Lorenz Kummer, Simon Fetzel, Nils M. Kriege, Wilfried N. Gansterer

Graph Neural Networks (GNNs) are powerful models for graph-structured data,
with broad applications. However, the interplay between GNN parameter
optimization, expressivity, and generalization remains poorly understood. We
address this by introducing an efficient learnable dimensionality reduction
method for visualizing GNN loss landscapes, and by analyzing the effects of
over-smoothing, jumping knowledge, quantization, sparsification, and
preconditioner on GNN optimization. Our learnable projection method surpasses
the state-of-the-art PCA-based approach, enabling accurate reconstruction of
high-dimensional parameters with lower memory usage. We further show that
architecture, sparsification, and optimizer's preconditioning significantly
impact the GNN optimization landscape and their training process and final
prediction performance. These insights contribute to developing more efficient
designs of GNN architectures and training strategies.

### Neural and Evolutionary Computing

### 1. [Time to Play: Simulating Early-Life Animal Dynamics Enhances Robotics Locomotion Discovery](http://arxiv.org/pdf/2509.11755v1)

Authors: Paul Templier, Hannah Janmohamed, David Labonte, Antoine Cully

Developmental changes in body morphology profoundly shape locomotion in
animals, yet artificial agents and robots are typically trained under static
physical parameters. Inspired by ontogenetic scaling of muscle power in
biology, we propose Scaling Mechanical Output over Lifetime (SMOL), a novel
curriculum that dynamically modulates robot actuator strength to mimic natural
variations in power-to-weight ratio during growth and ageing. Integrating SMOL
into the MAP-Elites quality-diversity framework, we vary the torque in standard
robotics tasks to mimic the evolution of strength in animals as they grow up
and as their body changes. Through comprehensive empirical evaluation, we show
that the SMOL schedule consistently elevates both performance and diversity of
locomotion behaviours across varied control scenarios, by allowing agents to
leverage advantageous physics early on to discover skills that act as stepping
stones when they reach their final standard body properties. Based on studies
of the total power output in humans, we also implement the SMOL-Human schedule
that models isometric body variations due to non-linear changes like puberty,
and study its impact on robotics locomotion.

### Networking and Internet Architecture

### 1. [Towards Dynamic Urban Scene Synthesis: The Digital Twin Descriptor Service](http://arxiv.org/pdf/2509.11810v1)

Authors: Ioannis Tsampras, Georgios Stergiopoulos, Tanya Politi, Spyros Denazis

Digital twins have been introduced as supporters to city operations, yet
existing scene-descriptor formats and digital twin platforms often lack the
integration, federation, and adaptable connectivity that urban environments
demand. Modern digital twin platforms decouple data streams and representations
into separate architectural planes, fusing them only at the visualization layer
and limiting potential for simulation or further processing of the combined
assets. At the same time, geometry-centric file standards for digital twin
description, and services built on top of them, focus primarily on explicitly
declaring geometry and additional structural or photorealistic parameters,
making integration with evolving context information a complicated process
while limiting compatibility with newer representation methods. Additionally,
multi-provider federation, critical in smart city services where multiple
stakeholders may control distinct infrastructure or representation assets, is
sparsely supported. Consequently, most pilots isolate context and
representation, fusing them per use case with ad hoc components and custom
description files or glue code, which hinders interoperability. To address
these gaps, this paper proposes a novel concept, the 'Digital Twin Descriptor
Service (DTDS)' that fuses abstracted references to geometry assets and context
information within a single, extensible descriptor service through NGSI-LD. The
proposed DTDS provides dynamic and federated integration of context data,
representations, and runtime synchronization across heterogeneous engines and
simulators. This concept paper outlines the DTDS architectural components and
description ontology that enable digital-twin processes in the modern smart
city.

### 2. [Optimization for Massive 3D-RIS Deployment: A Generative Diffusion Model-Based Approach](http://arxiv.org/pdf/2509.11969v1)

Authors: Kaining Wang, Bo Yang, Zhiwen Yu, Xuelin Cao, Mérouane Debbah, Chau Yuen

Reconfigurable Intelligent Surfaces (RISs) transform the wireless environment
by modifying the amplitude, phase, and polarization of incoming waves,
significantly improving coverage performance. Notably, optimizing the
deployment of RISs becomes vital, but existing optimization methods face
challenges such as high computational complexity, limited adaptability to
changing environments, and a tendency to converge on local optima. In this
paper, we propose to optimize the deployment of large-scale 3D RISs using a
diffusion model based on probabilistic generative learning. We begin by
dividing the target area into fixed grids, with each grid corresponding to a
potential deployment location. Then, a multi-RIS deployment optimization
problem is formulated, which is difficult to solve directly. By treating RIS
deployment as a conditional generation task, the well-trained diffusion model
can generate the distribution of deployment strategies, and thus, the optimal
deployment strategy can be obtained by sampling from this distribution.
Simulation results demonstrate that the proposed diffusion-based method
outperforms traditional benchmark approaches in terms of exceed ratio and
generalization.

### 3. [Beyond Regularity: Modeling Chaotic Mobility Patterns for Next Location Prediction](http://arxiv.org/pdf/2509.11713v1)

Authors: Yuqian Wu, Yuhong Peng, Jiapeng Yu, Xiangyu Liu, Zeting Yan, Kang Lin, Weifeng Su, Bingqing Qu, Raymond Lee, Dingqi Yang

Next location prediction is a key task in human mobility analysis, crucial
for applications like smart city resource allocation and personalized
navigation services. However, existing methods face two significant challenges:
first, they fail to address the dynamic imbalance between periodic and chaotic
mobile patterns, leading to inadequate adaptation over sparse trajectories;
second, they underutilize contextual cues, such as temporal regularities in
arrival times, which persist even in chaotic patterns and offer stronger
predictability than spatial forecasts due to reduced search spaces. To tackle
these challenges, we propose \textbf{\method}, a
\underline{\textbf{C}}h\underline{\textbf{A}}otic \underline{\textbf{N}}eural
\underline{\textbf{O}}scillator n\underline{\textbf{E}}twork for next location
prediction, which introduces a biologically inspired Chaotic Neural Oscillatory
Attention mechanism to inject adaptive variability into traditional attention,
enabling balanced representation of evolving mobility behaviors, and employs a
Tri-Pair Interaction Encoder along with a Cross Context Attentive Decoder to
fuse multimodal ``who-when-where'' contexts in a joint framework for enhanced
prediction performance. Extensive experiments on two real-world datasets
demonstrate that CANOE consistently and significantly outperforms a sizeable
collection of state-of-the-art baselines, yielding 3.17\%-13.11\% improvement
over the best-performing baselines across different cases. In particular, CANOE
can make robust predictions over mobility trajectories of different mobility
chaotic levels. A series of ablation studies also supports our key design
choices. Our code is available at: https://github.com/yuqian2003/CANOE.

### 4. [A Uniqueness Theorem for Distributed Computation under Physical Constraint](http://arxiv.org/pdf/2509.11754v1)

Authors: Zhiyuan Ren, Mingxuan Lu, Wenchi Cheng

Foundational models of computation often abstract away physical hardware
limitations. However, in extreme environments like In-Network Computing (INC),
these limitations become inviolable laws, creating an acute trilemma among
communication efficiency, bounded memory, and robust scalability. Prevailing
distributed paradigms, while powerful in their intended domains, were not
designed for this stringent regime and thus face fundamental challenges. This
paper demonstrates that resolving this trilemma requires a shift in perspective
- from seeking engineering trade-offs to deriving solutions from logical
necessity. We establish a rigorous axiomatic system that formalizes these
physical constraints and prove that for the broad class of computations
admitting an idempotent merge operator, there exists a unique, optimal
paradigm. Any system satisfying these axioms must converge to a single normal
form: Self-Describing Parallel Flows (SDPF), a purely data-centric model where
stateless executors process flows that carry their own control logic. We
further prove this unique paradigm is convergent, Turing-complete, and minimal.
In the same way that the CAP theorem established a boundary for what is
impossible in distributed state management, our work provides a constructive
dual: a uniqueness theorem that reveals what is \textit{inevitable} for
distributed computation flows under physical law.

### 5. [Task-Agnostic Learnable Weighted-Knowledge Base Scheme for Robust Semantic Communications](http://arxiv.org/pdf/2509.11636v1)

Authors: Shiyao Jiang, Jian Jiao, Xingjian Zhang, Ye Wang, Dusit Niyato, Qinyu Zhang

With the emergence of diverse and massive data in the upcoming
sixth-generation (6G) networks, the task-agnostic semantic communication system
is regarded to provide robust intelligent services. In this paper, we propose a
task-agnostic learnable weighted-knowledge base semantic communication (TALSC)
framework for robust image transmission to address the real-world heterogeneous
data bias in KB, including label flipping noise and class imbalance. The TALSC
framework incorporates a sample confidence module (SCM) as meta-learner and the
semantic coding networks as learners. The learners are updated based on the
empirical knowledge provided by the learnable weighted-KB (LW-KB). Meanwhile,
the meta-learner evaluates the significance of samples according to the task
loss feedback, and adjusts the update strategy of learners to enhance the
robustness in semantic recovery for unknown tasks. To strike a balance between
SCM parameters and precision of significance evaluation, we design an SCM-grid
extension (SCM-GE) approach by embedding the Kolmogorov-Arnold networks (KAN)
within SCM, which leverages the concept of spline refinement in KAN and enables
scalable SCM with customizable granularity without retraining. Simulations
demonstrate that the TALSC framework effectively mitigates the effects of
flipping noise and class imbalance in task-agnostic image semantic
communication, achieving at least 12% higher semantic recovery accuracy (SRA)
and multi-scale structural similarity (MS-SSIM) compared to state-of-the-art
methods.

### Robotics

### 1. [FR-Net: Learning Robust Quadrupedal Fall Recovery on Challenging Terrains through Mass-Contact Prediction](http://arxiv.org/pdf/2509.11504v1)

Authors: Yidan Lu, Yinzhao Dong, Jiahui Zhang, Ji Ma, Peng Lu

Fall recovery for legged robots remains challenging, particularly on complex
terrains where traditional controllers fail due to incomplete terrain
perception and uncertain interactions. We present \textbf{FR-Net}, a
learning-based framework that enables quadrupedal robots to recover from
arbitrary fall poses across diverse environments. Central to our approach is a
Mass-Contact Predictor network that estimates the robot's mass distribution and
contact states from limited sensory inputs, facilitating effective recovery
strategies. Our carefully designed reward functions ensure safe recovery even
on steep stairs without dangerous rolling motions common to existing methods.
Trained entirely in simulation using privileged learning, our framework guides
policy learning without requiring explicit terrain data during deployment. We
demonstrate the generalization capabilities of \textbf{FR-Net} across different
quadrupedal platforms in simulation and validate its performance through
extensive real-world experiments on the Go2 robot in 10 challenging scenarios.
Our results indicate that explicit mass-contact prediction is key to robust
fall recovery, offering a promising direction for generalizable quadrupedal
skills.

### 2. [Design and Development of a Remotely Wire-Driven Walking Robot](http://arxiv.org/pdf/2509.11506v1)

Authors: Takahiro Hattori, Kento Kawaharazuka, Kei Okada

Operating in environments too harsh or inaccessible for humans is one of the
critical roles expected of robots. However, such environments often pose risks
to electronic components as well. To overcome this, various approaches have
been developed, including autonomous mobile robots without electronics,
hydraulic remotely actuated mobile robots, and long-reach robot arms driven by
wires. Among these, electronics-free autonomous robots cannot make complex
decisions, while hydraulically actuated mobile robots and wire-driven robot
arms are used in harsh environments such as nuclear power plants. Mobile robots
offer greater reach and obstacle avoidance than robot arms, and wire mechanisms
offer broader environmental applicability than hydraulics. However, wire-driven
systems have not been used for remote actuation of mobile robots. In this
study, we propose a novel mechanism called Remote Wire Drive that enables
remote actuation of mobile robots via wires. This mechanism is a series
connection of decoupled joints, a mechanism used in wire-driven robot arms,
adapted for power transmission. We experimentally validated its feasibility by
actuating a wire-driven quadruped robot, which we also developed in this study,
through Remote Wire Drive.

### 3. [Shape control of simulated multi-segment continuum robots via Koopman operators with per-segment projection](http://arxiv.org/pdf/2509.11567v1)

Authors: Eron Ristich, Jiahe Wang, Lei Zhang, Sultan Haidar Ali, Wanxin Jin, Yi Ren, Jiefeng Sun

Soft continuum robots can allow for biocompatible yet compliant motions, such
as the ability of octopus arms to swim, crawl, and manipulate objects. However,
current state-of-the-art continuum robots can only achieve real-time task-space
control (i.e., tip control) but not whole-shape control, mainly due to the high
computational cost from its infinite degrees of freedom. In this paper, we
present a data-driven Koopman operator-based approach for the shape control of
simulated multi-segment tendon-driven soft continuum robots with the Kirchhoff
rod model. Using data collected from these simulated soft robots, we conduct a
per-segment projection scheme on the state of the robots allowing for the
identification of control-affine Koopman models that are an order of magnitude
more accurate than without the projection scheme. Using these learned Koopman
models, we use a linear model predictive control (MPC) to control the robots to
a collection of target shapes of varying complexity. Our method realizes
computationally efficient closed-loop control, and demonstrates the feasibility
of real-time shape control for soft robots. We envision this work can pave the
way for practical shape control of soft continuum robots.

### 4. [AssemMate: Graph-Based LLM for Robotic Assembly Assistance](http://arxiv.org/pdf/2509.11617v1)

Authors: Qi Zheng, Chaoran Zhang, Zijian Liang, EnTe Lin, Shubo Cui, Qinghongbing Xie, Zhaobo Xu, Long Zeng

Large Language Model (LLM)-based robotic assembly assistance has gained
significant research attention. It requires the injection of domain-specific
knowledge to guide the assembly process through natural language interaction
with humans. Despite some progress, existing methods represent knowledge in the
form of natural language text. Due to the long context and redundant content,
they struggle to meet the robots' requirements for real-time and precise
reasoning. In order to bridge this gap, we present AssemMate, which utilizes
the graph\textemdash a concise and accurate form of knowledge
representation\textemdash as input. This graph-based LLM enables knowledge
graph question answering (KGQA), supporting human-robot interaction and
assembly task planning for specific products. Beyond interactive QA, AssemMate
also supports sensing stacked scenes and executing grasping to assist with
assembly. Specifically, a self-supervised Graph Convolutional Network (GCN)
encodes knowledge graph entities and relations into a latent space and aligns
them with LLM's representation, enabling the LLM to understand graph
information. In addition, a vision-enhanced strategy is employed to address
stacked scenes in grasping. Through training and evaluation, AssemMate
outperforms existing methods, achieving 6.4\% higher accuracy, 3 times faster
inference, and 28 times shorter context length, while demonstrating strong
generalization ability on random graphs. And our approach further demonstrates
superiority through robotic grasping experiments in both simulated and
real-world settings. More details can be found on the project page:
https://github.com/cristina304/AssemMate.git

### 5. [Inference-stage Adaptation-projection Strategy Adapts Diffusion Policy to Cross-manipulators Scenarios](http://arxiv.org/pdf/2509.11621v1)

Authors: Xiangtong Yao, Yirui Zhou, Yuan Meng, Yanwen Liu, Liangyu Dong, Zitao Zhang, Zhenshan Bing, Kai Huang, Fuchun Sun, Alois Knoll

Diffusion policies are powerful visuomotor models for robotic manipulation,
yet they often fail to generalize to manipulators or end-effectors unseen
during training and struggle to accommodate new task requirements at inference
time. Addressing this typically requires costly data recollection and policy
retraining for each new hardware or task configuration. To overcome this, we
introduce an adaptation-projection strategy that enables a diffusion policy to
perform zero-shot adaptation to novel manipulators and dynamic task settings,
entirely at inference time and without any retraining. Our method first trains
a diffusion policy in SE(3) space using demonstrations from a base manipulator.
During online deployment, it projects the policy's generated trajectories to
satisfy the kinematic and task-specific constraints imposed by the new hardware
and objectives. Moreover, this projection dynamically adapts to physical
differences (e.g., tool-center-point offsets, jaw widths) and task requirements
(e.g., obstacle heights), ensuring robust and successful execution. We validate
our approach on real-world pick-and-place, pushing, and pouring tasks across
multiple manipulators, including the Franka Panda and Kuka iiwa 14, equipped
with a diverse array of end-effectors like flexible grippers, Robotiq 2F/3F
grippers, and various 3D-printed designs. Our results demonstrate consistently
high success rates in these cross-manipulator scenarios, proving the
effectiveness and practicality of our adaptation-projection strategy. The code
will be released after peer review.

### 6. [From Pixels to Shelf: End-to-End Algorithmic Control of a Mobile Manipulator for Supermarket Stocking and Fronting](http://arxiv.org/pdf/2509.11740v1)

Authors: Davide Peron, Victor Nan Fernandez-Ayala, Lukas Segelmark

Autonomous stocking in retail environments, particularly supermarkets,
presents challenges due to dynamic human interactions, constrained spaces, and
diverse product geometries. This paper introduces an efficient end-to-end
robotic system for autonomous shelf stocking and fronting, integrating
commercially available hardware with a scalable algorithmic architecture. A
major contribution of this work is the system integration of off-the-shelf
hardware and ROS2-based perception, planning, and control into a single
deployable platform for retail environments. Our solution leverages Behavior
Trees (BTs) for task planning, fine-tuned vision models for object detection,
and a two-step Model Predictive Control (MPC) framework for precise shelf
navigation using ArUco markers. Laboratory experiments replicating realistic
supermarket conditions demonstrate reliable performance, achieving over 98%
success in pick-and-place operations across a total of more than 700 stocking
events. However, our comparative benchmarks indicate that the performance and
cost-effectiveness of current autonomous systems remain inferior to that of
human workers, which we use to highlight key improvement areas and quantify the
progress still required before widespread commercial deployment can
realistically be achieved.

### 7. [Adaptive Motorized LiDAR Scanning Control for Robust Localization with OpenStreetMap](http://arxiv.org/pdf/2509.11742v1)

Authors: Jianping Li, Kaisong Zhu, Zhongyuan Liu, Rui Jin, Xinhang Xu, Pengfei Wan, Lihua Xie

LiDAR-to-OpenStreetMap (OSM) localization has gained increasing attention, as
OSM provides lightweight global priors such as building footprints. These
priors enhance global consistency for robot navigation, but OSM is often
incomplete or outdated, limiting its reliability in real-world deployment.
Meanwhile, LiDAR itself suffers from a limited field of view (FoV), where
motorized rotation is commonly used to achieve panoramic coverage. Existing
motorized LiDAR systems, however, typically employ constant-speed scanning that
disregards both scene structure and map priors, leading to wasted effort in
feature-sparse regions and degraded localization accuracy. To address these
challenges, we propose Adaptive LiDAR Scanning with OSM guidance, a framework
that integrates global priors with local observability prediction to improve
localization robustness. Specifically, we augment uncertainty-aware model
predictive control with an OSM-aware term that adaptively allocates scanning
effort according to both scene-dependent observability and the spatial
distribution of OSM features. The method is implemented in ROS with a motorized
LiDAR odometry backend and evaluated in both simulation and real-world
experiments. Results on campus roads, indoor corridors, and urban environments
demonstrate significant reductions in trajectory error compared to
constant-speed baselines, while maintaining scan completeness. These findings
highlight the potential of coupling open-source maps with adaptive LiDAR
scanning to achieve robust and efficient localization in complex environments.

### 8. [Igniting VLMs toward the Embodied Space](http://arxiv.org/pdf/2509.11766v1)

Authors: Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma, Ethan Yin, Hao Wang, Hugo Zhou, James Wang, Lights Shi, Lucy Liang, Make Wang, Qian Wang, Roy Gan, Ryan Yu, Shalfun Li, Starrick Liu, Sylas Chen, Vincent Chen, Zach Xu

While foundation models show remarkable progress in language and vision,
existing vision-language models (VLMs) still have limited spatial and
embodiment understanding. Transferring VLMs to embodied domains reveals
fundamental mismatches between modalities, pretraining distributions, and
training objectives, leaving action comprehension and generation as a central
bottleneck on the path to AGI.
  We introduce WALL-OSS, an end-to-end embodied foundation model that leverages
large-scale multimodal pretraining to achieve (1) embodiment-aware
vision-language understanding, (2) strong language-action association, and (3)
robust manipulation capability.
  Our approach employs a tightly coupled architecture and multi-strategies
training curriculum that enables Unified Cross-Level CoT-seamlessly unifying
instruction reasoning, subgoal decomposition, and fine-grained action synthesis
within a single differentiable framework.
  Our results show that WALL-OSS attains high success on complex long-horizon
manipulations, demonstrates strong instruction-following capabilities, complex
understanding and reasoning, and outperforms strong baselines, thereby
providing a reliable and scalable path from VLMs to embodied foundation models.

### 9. [Augmented Reality-Enhanced Robot Teleoperation for Collecting User Demonstrations](http://arxiv.org/pdf/2509.11783v1)

Authors: Shiqi Gong, Sebastian Zudaire, Chi Zhang, Zhen Li

Traditional industrial robot programming is often complex and time-consuming,
typically requiring weeks or even months of effort from expert programmers.
Although Programming by Demonstration (PbD) offers a more accessible
alternative, intuitive interfaces for robot control and demonstration
collection remain challenging. To address this, we propose an Augmented Reality
(AR)-enhanced robot teleoperation system that integrates AR-based control with
spatial point cloud rendering, enabling intuitive, contact-free demonstrations.
This approach allows operators to control robots remotely without entering the
workspace or using conventional tools like the teach pendant. The proposed
system is generally applicable and has been demonstrated on ABB robot
platforms, specifically validated with the IRB 1200 industrial robot and the
GoFa 5 collaborative robot. A user study evaluates the impact of real-time
environmental perception, specifically with and without point cloud rendering,
on task completion accuracy, efficiency, and user confidence. Results indicate
that enhanced perception significantly improves task performance by 28% and
enhances user experience, as reflected by a 12% increase in the System
Usability Scale (SUS) score. This work contributes to the advancement of
intuitive robot teleoperation, AR interface design, environmental perception,
and teleoperation safety mechanisms in industrial settings for demonstration
collection. The collected demonstrations may serve as valuable training data
for machine learning applications.

### 10. [UniPilot: Enabling GPS-Denied Autonomy Across Embodiments](http://arxiv.org/pdf/2509.11793v1)

Authors: Mihir Kulkarni, Mihir Dharmadhikari, Nikhil Khedekar, Morten Nissov, Mohit Singh, Philipp Weiss, Kostas Alexis

This paper presents UniPilot, a compact hardware-software autonomy payload
that can be integrated across diverse robot embodiments to enable autonomous
operation in GPS-denied environments. The system integrates a multi-modal
sensing suite including LiDAR, radar, vision, and inertial sensing for robust
operation in conditions where uni-modal approaches may fail. UniPilot runs a
complete autonomy software comprising multi-modal perception, exploration and
inspection path planning, and learning-based navigation policies. The payload
provides robust localization, mapping, planning, and safety and control
capabilities in a single unit that can be deployed across a wide range of
platforms. A large number of experiments are conducted across diverse
environments and on a variety of robot platforms to validate the mapping,
planning, and safe navigation capabilities enabled by the payload.

### Software Engineering

### 1. [VulAgent: Hypothesis-Validation based Multi-Agent Vulnerability Detection](http://arxiv.org/pdf/2509.11523v1)

Authors: Ziliang Wang, Ge Li, Jia Li, Hao Zhu, Zhi Jin

The application of language models to project-level vulnerability detection
remains challenging, owing to the dual requirement of accurately localizing
security-sensitive code and correctly correlating and reasoning over complex
program context. We present VulAgent, a multi-agent vulnerability detection
framework based on hypothesis validation. Our design is inspired by how human
auditors review code: when noticing a sensitive operation, they form a
hypothesis about a possible vulnerability, consider potential trigger paths,
and then verify the hypothesis against the surrounding context. VulAgent
implements a semantics-sensitive, multi-view detection pipeline: specialized
agents, each aligned to a specific analysis perspective (e.g., memory,
authorization), collaboratively surface and precisely localize sensitive code
sites with higher coverage. Building on this, VulAgent adopts a
hypothesis-validation paradigm: for each vulnerability report, it builds
hypothesis conditions and a trigger path, steering the LLM to target the
relevant program context and defensive checks during verification, which
reduces false positives. On average across the two datasets, VulAgent improves
overall accuracy by 6.6%, increases the correct identification rate of
vulnerable--fixed code pairs by up to 450% (246% on average), and reduces the
false positive rate by about 36% compared with state-of-the-art LLM-based
baselines.

### 2. [Sedeve-Kit, a Specification-Driven Development Framework for Building Distributed Systems](http://arxiv.org/pdf/2509.11566v1)

Authors: Hua Guo, Yunhong Ji, Xuan Zhou

Developing distributed systems presents significant challenges, primarily due
to the complexity introduced by non-deterministic concurrency and faults. To
address these, we propose a specification-driven development framework. Our
method encompasses three key stages. The first stage defines system
specifications and invariants using TLA${^+}$. It allows us to perform model
checking on the algorithm's correctness and generate test cases for subsequent
development phases. In the second stage, based on the established
specifications, we write code to ensure consistency and accuracy in the
implementation. Finally, after completing the coding process, we rigorously
test the system using the test cases generated in the initial stage. This
process ensures system quality by maintaining a strong connection between the
abstract design and the concrete implementation through continuous
verification.

### 3. [AI Asset Management for Manufacturing (AIM4M): Development of a Process Model for Operationalization](http://arxiv.org/pdf/2509.11691v1)

Authors: Lukas Rauh, Mel-Rick Süner, Daniel Schel, Thomas Bauernhansl

The benefits of adopting artificial intelligence (AI) in manufacturing are
undeniable. However, operationalizing AI beyond the prototype, especially when
involved with cyber-physical production systems (CPPS), remains a significant
challenge due to the technical system complexity, a lack of implementation
standards and fragmented organizational processes. To this end, this paper
proposes a new process model for the lifecycle management of AI assets designed
to address challenges in manufacturing and facilitate effective
operationalization throughout the entire AI lifecycle. The process model, as a
theoretical contribution, builds on machine learning operations (MLOps)
principles and refines three aspects to address the domain-specific
requirements from the CPPS context. As a result, the proposed process model
aims to support organizations in practice to systematically develop, deploy and
manage AI assets across their full lifecycle while aligning with CPPS-specific
constraints and regulatory demands.

### 4. [From Evaluation to Enhancement: Large Language Models for Zero-Knowledge Proof Code Generation](http://arxiv.org/pdf/2509.11708v1)

Authors: Zhantong Xue, Pingchuan Ma, Zhaoyu Wang, Shuai Wang

Zero-knowledge proofs (ZKPs) are increasingly deployed in domains such as
privacy-preserving authentication, blockchain scalability, and secure finance.
However, authoring ZK programs remains challenging: unlike mainstream
programming, ZK development requires reasoning about finite field arithmetic,
constraint systems, and gadgets, making it knowledge-intensive and error-prone.
While large language models (LLMs) have demonstrated strong code generation
capabilities in general-purpose languages, their effectiveness for ZK
programming, where correctness hinges on both language mastery and gadget-level
reasoning, remains unexplored. To address this gap, we propose
\textsc{ZK-Eval}, a domain-specific evaluation pipeline that probes LLM
capabilities at three levels: language knowledge, gadget competence, and
end-to-end program generation. Our evaluation of four state-of-the-art LLMs
reveals that models excel at surface-level syntax but struggle with gadget
usage and semantic correctness, often yielding incorrect programs. Based on
these insights, we introduce \textsc{ZK-Coder}, an agentic framework that
augments LLMs with constraint sketching, guided retrieval, and interactive
repair. Experiments on Circom and Noir show substantial gains, with success
rates improving from 17.35\% to 83.38\% and from 32.21\% to 90.05\%,
respectively. With \textsc{ZK-Eval} and \textsc{ZK-Coder}, we establish a
foundation for systematically measuring and augmenting LLMs in ZK code
generation to lower barriers for practitioners and advance trustworthy
computation.

### 5. [Toward Greener Background Processes -- Measuring Energy Cost of Autosave Feature](http://arxiv.org/pdf/2509.11738v1)

Authors: Maria Küüsvek, Hina Anwar

Background processes in desktop applications are often overlooked in energy
consumption studies, yet they represent continuous, automated workloads with
significant cumulative impact. This paper introduces a reusable process for
evaluating the energy behavior of such features at the level of operational
design. The process works in three phases: 1) decomposing background
functionality into core operations, 2) operational isolation, and 3) controlled
measurements enabling comparative profiling. We instantiate the process in a
case study of autosave implementations across three open-source Python-based
text editors. Using 900 empirical software-based energy measurements, we
identify key design factors affecting energy use, including save frequency,
buffering strategy, and auxiliary logic such as change detection. We give four
actionable recommendations for greener implementations of autosave features in
Python to support sustainable software practices.

### 6. [LitterBox+: An Extensible Framework for LLM-enhanced Scratch Static Code Analysis](http://arxiv.org/pdf/2509.12021v1)

Authors: Benedikt Fein, Florian Obermüller, Gordon Fraser

Large language models (LLMs) have become an essential tool to support
developers using traditional text-based programming languages, but the
graphical notation of the block-based Scratch programming environment inhibits
the use of LLMs. To overcome this limitation, we propose the LitterBox+
framework that extends the Scratch static code analysis tool LitterBox with the
generative abilities of LLMs. By converting block-based code to a textual
representation suitable for LLMs, LitterBox+ allows users to query LLMs about
their programs, about quality issues reported by LitterBox, and it allows
generating code fixes. Besides offering a programmatic API for these
functionalities, LitterBox+ also extends the Scratch user interface to make
these functionalities available directly in the environment familiar to
learners. The framework is designed to be easily extensible with other prompts,
LLM providers, and new features combining the program analysis capabilities of
LitterBox with the generative features of LLMs. We provide a screencast
demonstrating the tool at https://youtu.be/RZ6E0xgrIgQ.

### 7. [A New Benchmark for Evaluating Code Translation with Third-Party Libraries](http://arxiv.org/pdf/2509.12087v1)

Authors: Pengyu Xue, Kunwu Zheng, Zhen Yang, Yifei Pei, Linhao Wu, Jiahui Dong, Xiapu Luo, Yan Xiao, Fei Liu, Yuxuan Zhang, Xiran Lyu, Xianhang Li, Xuanyu Zhu, Chengyi Wang

In recent years, Large Language Models (LLMs) have been widely studied in the
code translation field on the method, class, and even repository levels.
However, most of these benchmarks are limited in terms of Third-Party Library
(TPL) categories and scales, making TPL-related errors hard to expose and
hindering the development of targeted solutions. Considering the high
dependence (over 90%) on TPLs in practical programming, demystifying and
analyzing LLMs' code translation performance involving various TPLs becomes
imperative. To address this gap, we construct TransLibEval, the first benchmark
dedicated to library-centric code translation. It consists of 200 real-world
tasks across Python, Java, and C++, each explicitly involving TPLs from diverse
categories such as data processing, machine learning, and web development, with
comprehensive dependency coverage and high-coverage test suites. We evaluate
seven recent LLMs of commercial, general, and code-specialized families under
six translation strategies of three categories: Direct, IR-guided, and
Retrieval-augmented. Experimental results show a dramatic performance drop
compared with library-free settings (average CA decline over 60%), while
diverse strategies demonstrate heterogeneous advantages. Furthermore, we
analyze 4,831 failed cases from GPT-4o, one of the State-of-the-Art (SOTA)
LLMs, revealing numerous third-party reference errors that were obscured
previously. These findings highlight the unique challenges of library-centric
translation and provide practical guidance for improving TPL-aware code
intelligence.

### 8. [Automated Creation and Enrichment Framework for Improved Invocation of Enterprise APIs as Tools](http://arxiv.org/pdf/2509.11626v1)

Authors: Prerna Agarwal, Himanshu Gupta, Soujanya Soni, Rohith Vallam, Renuka Sindhgatta, Sameep Mehta

Recent advancements in Large Language Models (LLMs) has lead to the
development of agents capable of complex reasoning and interaction with
external tools. In enterprise contexts, the effective use of such tools that
are often enabled by application programming interfaces (APIs), is hindered by
poor documentation, complex input or output schema, and large number of
operations. These challenges make tool selection difficult and reduce the
accuracy of payload formation by up to 25%. We propose ACE, an automated tool
creation and enrichment framework that transforms enterprise APIs into
LLM-compatible tools. ACE, (i) generates enriched tool specifications with
parameter descriptions and examples to improve selection and invocation
accuracy, and (ii) incorporates a dynamic shortlisting mechanism that filters
relevant tools at runtime, reducing prompt complexity while maintaining
scalability. We validate our framework on both proprietary and open-source APIs
and demonstrate its integration with agentic frameworks. To the best of our
knowledge, ACE is the first end-to-end framework that automates the creation,
enrichment, and dynamic selection of enterprise API tools for LLM agents.

### 9. [Do Code Semantics Help? A Comprehensive Study on Execution Trace-Based Information for Code Large Language Models](http://arxiv.org/pdf/2509.11686v1)

Authors: Jian Wang, Xiaofei Xie, Qiang Hu, Shangqing Liu, Yi Li

Code Large Language Models (Code LLMs) have opened a new era in programming
with their impressive capabilities. However, recent research has revealed
critical limitations in their ability to reason about runtime behavior and
understand the actual functionality of programs, which poses significant
challenges for their post-training and practical deployment. Specifically, Code
LLMs encounter two principal issues: (1) a lack of proficiency in reasoning
about program execution behavior, as they struggle to interpret what programs
actually do during runtime, and (2) the inconsistent and fragmented
representation of semantic information, such as execution traces, across
existing methods, which hinders their ability to generalize and reason
effectively. These challenges underscore the necessity for more systematic
approaches to enhance the reasoning capabilities of Code LLMs. To address these
issues, we introduce a generic framework to support integrating semantic
information~(e.g., execution trace) to code task-relevant prompts, and conduct
a comprehensive study to explore the role of semantic information in enhancing
the reasoning ability of Code LLMs accordingly. Specifically, we focus on
investigating the usefulness of trace-based semantic information in boosting
supervised fine-tuning~(SFT) and post-phase inference of Code LLMs. The
experimental results surprisingly disagree with previous works and demonstrate
that semantic information has limited usefulness for SFT and test time scaling
of Code LLM.

### 10. [A Holistic Approach to E-Commerce Innovation: Redefining Security and User Experience](http://arxiv.org/pdf/2509.11712v1)

Authors: Mohammad Olid Ali Akash, Priyangana Saha

In the modern, fast-moving world of e-commerce, many Android apps face
challenges in providing a simple and secure shopping experience. Many of these
apps, often enough, have complicated designs that prevent users from finding
what they want quickly, thus frustrating them and wasting their precious time.
Another major issue is that of security; with the limitation of payment options
and weak authentication mechanisms, users' sensitive information can be
compromised. This research presents a new e-commerce platform that responds to
the above challenges with an intuitive interface and strong security measures.
The platform makes online shopping easy with well-organized categories of
products and a fast, efficient checkout process. It also gives priority to
security by incorporating features such as Google authentication and
SSL-secured payment gateways to protect user data and ensure secure
transactions. This paper discusses how a focus on user-friendliness, security,
and personalization steps up the game for e-commerce platforms, providing
workable frameworks that match modern user needs and expectations. The findings
show the e-commerce user experience can be remodelled by the platform, hence
opening ways for future developments in that respect.

### Social and Information Networks

### 1. [No Community Detection Method to Rule Them All!](http://arxiv.org/pdf/2509.11490v1)

Authors: Shrabani Ghosh, Erik Saule

Community detection is a core tool for analyzing large realworld graphs. It
is often used to derive additional local features of vertices and edges that
will be used to perform a downstream task, yet the impact of community
detection on downstream tasks is poorly understood. Prior work largely
evaluates community detection algorithms by their intrinsic objectives (e.g.,
modularity). Or they evaluate the impact of using community detection onto on
the downstream task. But the impact of particular community detection algortihm
support the downstream task. We study the relationship between community
structure and downstream performance across multiple algorithms and two tasks.
Our analysis links community-level properties to task metrics (F1, precision,
recall, AUC) and reveals that the choice of detection method materially affects
outcomes. We explore thousands of community structures and show that while the
properties of communities are the reason behind the impact on task performance,
no single property explains performance in a direct way. Rather, results emerge
from complex interactions among properties. As such, no standard community
detection algorithm will derive the best downstream performance. We show that a
method combining random community generation and simple machine learning
techniques can derive better performance

### 2. [Fostering cultural change in research through innovative knowledge sharing, evaluation, and community engagement strategies](http://arxiv.org/pdf/2509.12045v1)

Authors: Junsuk Rho, Jinn-Kong Sheu, Andrew Forbes, Din Ping Tsai, Andrea Alú, Wei Li, Mark Brongersma, Joonhee Choi, Javier Garcia de Abajo, Laura Na Liu, Alexander Szameit, Tracy Schloemer, Andreas Tittl, Mario Chemnitz, Cheng Wang, Jiejun Zhang, Yuri Kivshar, Tie Jun Cui, Ren-Min Ma, Cheng-Wei Qiu, Cuicui Lu, Yao-Wei Huang, Miguel Angel Solis Prosser, Ileana-Cristina Benea-Chelmus, Rachel Grange, Sungjin Kim, Anderson S. L. Gomes, Davide Ramaccia, Yating Wan, Apostolos Argyris, Antonio G. Souza Filho, Tanmoy Chandrad, Cristiano Matricardi

Scientific research needs a new system that appropriately values science and
scientists. Key innovations, within institutions and funding agencies, are
driving better assessment of research, with open knowledge and FAIR (findable,
accessible, interoperable, and reusable) principles as central pillars.
Furthermore, coalitions, agreements, and robust infrastructures have emerged to
promote more accurate assessment metrics and efficient knowledge sharing.
However, despite these efforts, the system still relies on outdated methods
where standardized metrics such as h-index and journal impact factor dominate
evaluations. These metrics have had the unintended consequence of pushing
researchers to produce more outputs at the expense of integrity and
reproducibility. In this community paper, we bring together a global community
of researchers, funding institutions, industrial partners, and publishers from
14 different countries across the 5 continents. We aim at collectively envision
an evolved knowledge sharing and research evaluation along with the potential
positive impact on every stakeholder involved. We imagine these ideas to set
the groundwork for a cultural change to redefine a more fair and equitable
scientific landscape.

### 3. [Percolation and matrix spectrum through NIB message passing](http://arxiv.org/pdf/2509.11730v1)

Authors: Pedro Hack

Given its computational efficiency and versatility, belief propagation is the
most prominent message passing method in several applications. In order to
diminish the damaging effect of loops on its accuracy, the first explicit
version of generalized belief propagation for networks, the KCN-method, was
recently introduced. This approach was originally developed in the context of
two target problems: percolation and the calculation of the spectra of sparse
matrices. Later on, the KCN-method was extended in order to deal with inference
in the context of probabilistic graphical models on networks. It was in this
scenario where an improvement on the KCN-method, the NIB-method, was conceived.
We show here that this improvement can also achieved in the original
applications of the KCN-method, namely percolation and matrix spectra.

### 4. [Evidencing preferential attachment in dependency network evolution](http://arxiv.org/pdf/2509.12135v1)

Authors: Clement Lee

Preferential attachment is often suggested to be the underlying mechanism of
the growth of a network, largely due to that many real networks are, to a
certain extent, scale-free. However, such attribution is usually made under
debatable practices of determining scale-freeness and when only snapshots of
the degree distribution are observed. In the presence of the evolution history
of the network, modelling the increments of the evolution allows us to measure
preferential attachment directly. Therefore, we propose a generalised linear
model for such purpose, where the in-degrees and their increments are the
covariate and response, respectively. Not only are the parameters that describe
the preferential attachment directly incorporated, they also ensure that the
tail heaviness of the asymptotic degree distribution is realistic. The Bayesian
approach to inference enables the hierarchical version of the model to be
implemented naturally. The application to the dependency network of R packages
reveals subtly different behaviours between new dependencies by new and
existing packages, and between addition and removal of dependencies.

### 5. [The threshold and quasi-stationary distribution for the SIS model on networks](http://arxiv.org/pdf/2509.11706v1)

Authors: George Cantwell, Cristopher Moore

We study the Susceptible-Infectious-Susceptible (SIS) model on arbitrary
networks. The well-established pair approximation treats neighboring pairs of
nodes exactly while making a mean field approximation for the rest of the
network. We improve the method by expanding the state space dynamically, giving
nodes a memory of when they last became susceptible. The resulting
approximation is simple to implement and appears to be highly accurate, both in
locating the epidemic threshold and in computing the quasi-stationary fraction
of infected individuals above the threshold, for both finite graphs and
infinite random graphs.

### 6. [Updating the Complex Systems Keyword Diagram Using Collective Feedback and Latest Literature Data](http://arxiv.org/pdf/2509.11997v1)

Authors: Hiroki Sayama

The complex systems keyword diagram generated by the author in 2010 has been
used widely in a variety of educational and outreach purposes, but it
definitely needs a major update and reorganization. This short paper reports
our recent attempt to update the keyword diagram using information collected
from the following multiple sources: (a) collective feedback posted on social
media, (b) recent reference books on complex systems and network science, (c)
online resources on complex systems, and (d) keyword search hits obtained using
OpenAlex, an open-access bibliographic catalogue of scientific publications.
The data (a), (b) and (c) were used to incorporate the research community's
internal perceptions of the relevant topics, whereas the data (d) was used to
obtain more objective measurements of the keywords' relevance and associations
from publications made in complex systems science. Results revealed differences
and overlaps between public perception and actual usage of keywords in
publications on complex systems. Four topical communities were obtained from
the keyword association network, although they were highly intertwined with
each other. We hope that the resulting network visualization of complex systems
keywords provides a more up-to-date, accurate topic map of the field of complex
systems as of today.

### Systems and Control

### 1. [Model Predictive Control with High-Probability Safety Guarantee for Nonlinear Stochastic Systems](http://arxiv.org/pdf/2509.11584v1)

Authors: Zishun Liu, Liqian Ma, Yongxin Chen

We present a model predictive control (MPC) framework for nonlinear
stochastic systems that ensures safety guarantee with high probability. Unlike
most existing stochastic MPC schemes, our method adopts a set-erosion that
converts the probabilistic safety constraint into a tractable deterministic
safety constraint on a smaller safe set over deterministic dynamics. As a
result, our method is compatible with any off-the-shelf deterministic MPC
algorithm. The key to the effectiveness of our method is a tight bound on the
stochastic fluctuation of a stochastic trajectory around its nominal version.
Our method is scalable and can guarantee safety with high probability level
(e.g., 99.99%), making it particularly suitable for safety-critical
applications involving complex nonlinear dynamics. Rigorous analysis is
conducted to establish a theoretical safety guarantee, and numerical
experiments are provided to validate the effectiveness of the proposed MPC
method.

### 2. [$ε$-Optimal Multi-Agent Patrol using Recurrent Strategy](http://arxiv.org/pdf/2509.11640v1)

Authors: Deepak Mallya, Arpita Sinha, Leena Vachhani

The multi-agent patrol problem refers to repeatedly visiting different
locations in an environment using multiple autonomous agents. For over two
decades, researchers have studied this problem in various settings. While
providing valuable insights into the problem, the works in existing literature
have not commented on the nature of the optimal solutions to the problem. We
first show that an $\epsilon$-approximate recurrent patrol strategy exists for
every feasible patrol strategy. Then, we establish the existence of a recurrent
patrol strategy that is an $\epsilon$-optimal solution to the General Patrol
Problem. The factor $\epsilon$ is proportional to the discretisation constant
$D$, which can be arbitrarily small and is independent of the number of patrol
agents and the size of the environment. This result holds for a variety of
problem formulations already studied. We also provide an algorithmic approach
to determine an $\epsilon$-approximate recurrent patrol strategy for a patrol
strategy created by any method from the literature. We perform extensive
simulations in graphs based on real-life environments to validate the claims
made in this work.

### 3. [Continuous-Time Distributed Learning for Collective Wisdom Maximization](http://arxiv.org/pdf/2509.11808v1)

Authors: Luka Baković, Giacomo Como, Fabio Fagnani, Anton Proskurnikov, Emma Tegling

Motivated by the well established idea that collective wisdom is greater than
that of an individual, we propose a novel learning dynamics as a sort of
companion to the Abelson model of opinion dynamics. Agents are assumed to make
independent guesses about the true state of the world after which they engage
in opinion exchange leading to consensus. We investigate the problem of finding
the optimal parameters for this exchange, e.g. those that minimize the variance
of the consensus value. Specifically, the parameter we examine is
susceptibility to opinion change. We propose a dynamics for distributed
learning of the optimal parameters and analytically show that it converges for
all relevant initial conditions by linking to well established results from
consensus theory. Lastly, a numerical example provides intuition on both system
behavior and our proof methods.

### 4. [Varying Horizon Learning Economic MPC With Unknown Costs of Disturbed Nonlinear Systems](http://arxiv.org/pdf/2509.11823v1)

Authors: Weiliang Xiong, Defeng He, Haiping Du, Jianbin Mu

This paper proposes a novel varying horizon economic model predictive control
(EMPC) scheme without terminal constraints for constrained nonlinear systems
with additive disturbances and unknown economic costs. The general regression
learning framework with mixed kernels is first used to reconstruct the unknown
cost. Then an online iterative procedure is developed to adjust the horizon
adaptively. Again, an elegant horizon-dependent contraction constraint is
designed to ensure the convergence of the closed-loop system to a neighborhood
of the desired steady state. Moreover, sufficient conditions ensuring recursive
feasibility and input-to-state stability are established for the system in
closed-loop with the EMPC. The merits of the proposed scheme are verified by
the simulations of a continuous stirred tank reactor and a four-tank system in
terms of robustness, economic performance and online computational burden.

### 5. [Distributed Finite-Horizon Optimal Control for Consensus with Differential Privacy Guarantees](http://arxiv.org/pdf/2509.11917v1)

Authors: Yuwen Ma, Yongqiang Wang, Sarah K. Spurgeon, Boli Chen

This paper addresses the problem of privacy-preserving consensus control for
multi-agent systems (MAS) using differential privacy. We propose a novel
distributed finite-horizon linear quadratic regulator (LQR) framework, in which
agents share individual state information while preserving the confidentiality
of their local pairwise weight matrices, which are considered sensitive data in
MAS. Protecting these matrices effectively safeguards each agent's private cost
function and control preferences. Our solution injects consensus
error-dependent Laplace noise into the communicated state information and
employs a carefully designed time-dependent scaling factor in the local cost
functions. {This approach guarantees bounded consensus and achieves rigorous
$\epsilon$-differential privacy for the weight matrices without relying on
specific noise distribution assumptions.} Additionally, we analytically
characterize the trade-off between consensus accuracy and privacy level,
offering clear guidelines on how to enhance consensus performance through
appropriate scaling of the LQR weight matrices and the privacy budget.

### 6. [Compositional shield synthesis for safe reinforcement learning in partial observability](http://arxiv.org/pdf/2509.12085v1)

Authors: Steven Carr, Georgios Bakirtzis, Ufuk Topcu

Agents controlled by the output of reinforcement learning (RL) algorithms
often transition to unsafe states, particularly in uncertain and partially
observable environments. Partially observable Markov decision processes
(POMDPs) provide a natural setting for studying such scenarios with limited
sensing. Shields filter undesirable actions to ensure safe RL by preserving
safety requirements in the agents' policy. However, synthesizing holistic
shields is computationally expensive in complex deployment scenarios. We
propose the compositional synthesis of shields by modeling safety requirements
by parts, thereby improving scalability. In particular, problem formulations in
the form of POMDPs using RL algorithms illustrate that an RL agent equipped
with the resulting compositional shielding, beyond being safe, converges to
higher values of expected reward. By using subproblem formulations, we preserve
and improve the ability of shielded agents to require fewer training episodes
than unshielded agents, especially in sparse-reward settings. Concretely, we
find that compositional shield synthesis allows an RL agent to remain safe in
environments two orders of magnitude larger than other state-of-the-art
model-based approaches.

### 7. [Design and Optimization of EV Charging Infrastructure with Battery in Commercial Buildings](http://arxiv.org/pdf/2509.12160v1)

Authors: Quan Nguyen, Christine Holland, Siddharth Sridhar

The installation of electric vehicle (EV) charging stations in buildings is
inevitable, as states push for increased EV adoption to support decarbonization
efforts. This transition could force the need for grid infrastructure upgrades
and enhanced controls to support reliable power delivery to end-use loads, and
overall economic operation. This paper evaluates strategies that address these
needs on two fronts: i) optimal sizing of service transformers and battery
energy storage systems (BESS), and ii) optimized coordination between EV
charging, BESS operation, and building demand. These strategies are applied to
a school campus setting, consisting of building and EV charging loads, to
provide an illustration of energy management in commercial buildings with EV
fleets. A rolling-window optimization approach is applied to determine i)
optimal sizing of the service transformer and BESS and ii) optimal control of
EV charging and BESS charge/discharge schedules. The design and control
strategies are validated in a 20-year time horizon with an annually increasing
number of EVs (buses and vans). In addition, an economic analysis is also
carried out to show the costs and benefits of each design as a medium- and
long-term investment.

### 8. [PaiP: An Operational Aware Interactive Planner for Unknown Cabinet Environments](http://arxiv.org/pdf/2509.11516v1)

Authors: Chengjin Wang, Zheng Yan, Yanmin Zhou, Runjie Shen, Zhipeng Wang, Bin Cheng, Bin He

Box/cabinet scenarios with stacked objects pose significant challenges for
robotic motion due to visual occlusions and constrained free space. Traditional
collision-free trajectory planning methods often fail when no collision-free
paths exist, and may even lead to catastrophic collisions caused by invisible
objects. To overcome these challenges, we propose an operational aware
interactive motion planner (PaiP) a real-time closed-loop planning framework
utilizing multimodal tactile perception. This framework autonomously infers
object interaction features by perceiving motion effects at interaction
interfaces. These interaction features are incorporated into grid maps to
generate operational cost maps. Building upon this representation, we extend
sampling-based planning methods to interactive planning by optimizing both path
cost and operational cost. Experimental results demonstrate that PaiP achieves
robust motion in narrow spaces.

### 9. [Tensor Invariant Data-Assisted Control and Dynamic Decomposition of Multibody Systems](http://arxiv.org/pdf/2509.11688v1)

Authors: Mostafa Eslami, Maryam Babazadeh

The control of robotic systems in complex, shared collaborative workspaces
presents significant challenges in achieving robust performance and safety when
learning from experienced or simulated data is employed in the pipeline. A
primary bottleneck is the reliance on coordinate-dependent models, which leads
to profound data inefficiency by failing to generalize physical interactions
across different frames of reference. This forces learning algorithms to
rediscover fundamental physical principles in every new orientation,
artificially inflating the complexity of the learning task. This paper
introduces a novel framework that synergizes a coordinate-free, unreduced
multibody dynamics and kinematics model based on tensor mechanics with a
Data-Assisted Control (DAC) architecture. A non-recursive, closed-form
Newton-Euler model in an augmented matrix form is derived that is optimized for
tensor-based control design. This structure enables a principled decomposition
of the system into a structurally certain, physically grounded part and an
uncertain, empirical, and interaction-focused part, mediated by a virtual port
variable. Then, a complete, end-to-end tensor-invariant pipeline for modeling,
control, and learning is proposed. The coordinate-free control laws for the
structurally certain part provide a stable and abstract command interface,
proven via Lyapunov analysis. Eventually, the model and closed-loop system are
validated through simulations. This work provides a naturally ideal input for
data-efficient, frame-invariant learning algorithms, such as equivariant
learning, designed to learn the uncertain interaction. The synergy directly
addresses the data-inefficiency problem, increases explainability and
interpretability, and paves the way for more robust and generalizable robotic
control in interactive environments.

### 10. [Convergence Filters for Efficient Economic MPC of Non-dissipative Systems](http://arxiv.org/pdf/2509.11869v1)

Authors: Defeng He, Weiliang Xiong, Shiqiang He, Haiping Du

This note presents a novel, efficient economic model predictive control
(EMPC) scheme for non-dissipative systems subject to state and input
constraints. A new conception of convergence filters is defined to address the
stability issue of EMPC for constrained non-dissipative systems. Three
convergence filters are designed accordingly to be imposed into the receding
horizon optimization problem of EMPC. To improve online computational
efficiency, the variable horizon idea without terminal constraints is adopted
to compromise the convergence speed, economic performance, and computational
burden of EMPC. Moreover, sufficient conditions are derived to guarantee the
recursive feasibility and stability of the EMPC. The advantages of the proposed
EMPC are validated by a classical non-dissipative continuous stirred-tank
reactor.

### Machine Learning (Statistics Category)

### 1. [Learning Majority-to-Minority Transformations with MMD and Triplet Loss for Imbalanced Classification](http://arxiv.org/pdf/2509.11511v1)

Authors: Suman Cha, Hyunjoong Kim

Class imbalance in supervised classification often degrades model performance
by biasing predictions toward the majority class, particularly in critical
applications such as medical diagnosis and fraud detection. Traditional
oversampling techniques, including SMOTE and its variants, generate synthetic
minority samples via local interpolation but fail to capture global data
distributions in high-dimensional spaces. Deep generative models based on GANs
offer richer distribution modeling yet suffer from training instability and
mode collapse under severe imbalance. To overcome these limitations, we
introduce an oversampling framework that learns a parametric transformation to
map majority samples into the minority distribution. Our approach minimizes the
maximum mean discrepancy (MMD) between transformed and true minority samples
for global alignment, and incorporates a triplet loss regularizer to enforce
boundary awareness by guiding synthesized samples toward challenging borderline
regions. We evaluate our method on 29 synthetic and real-world datasets,
demonstrating consistent improvements over classical and generative baselines
in AUROC, G-mean, F1-score, and MCC. These results confirm the robustness,
computational efficiency, and practical utility of the proposed framework for
imbalanced classification tasks.

### 2. [E-ROBOT: a dimension-free method for robust statistics and machine learning via Schrödinger bridge](http://arxiv.org/pdf/2509.11532v1)

Authors: Davide La Vecchia, Hang Liu

We propose the Entropic-regularized Robust Optimal Transport (E-ROBOT)
framework, a novel method that combines the robustness of ROBOT with the
computational and statistical benefits of entropic regularization. We show
that, rooted in the Schr\"{o}dinger bridge problem theory, E-ROBOT defines the
robust Sinkhorn divergence $\overline{W}_{\varepsilon,\lambda}$, where the
parameter $\lambda$ controls robustness and $\varepsilon$ governs the
regularization strength. Letting $n\in \mathbb{N}$ denote the sample size, a
central theoretical contribution is establishing that the sample complexity of
$\overline{W}_{\varepsilon,\lambda}$ is $\mathcal{O}(n^{-1/2})$, thereby
avoiding the curse of dimensionality that plagues standard ROBOT. This
dimension-free property unlocks the use of $\overline{W}_{\varepsilon,\lambda}$
as a loss function in large-dimensional statistical and machine learning tasks.
With this regard, we demonstrate its utility through four applications:
goodness-of-fit testing; computation of barycenters for corrupted 2D and 3D
shapes; definition of gradient flows; and image colour transfer. From the
computation standpoint, a perk of our novel method is that it can be easily
implemented by modifying existing (\texttt{Python}) routines. From the
theoretical standpoint, our work opens the door to many research directions in
statistics and machine learning: we discuss some of them.

### 3. [SpaPool: Soft Partition Assignment Pooling for__Graph Neural Networks](http://arxiv.org/pdf/2509.11675v1)

Authors: Rodrigue Govan, Romane Scherrer, Philippe Fournier-Viger, Nazha Selmaoui-Folcher

This paper introduces SpaPool, a novel pooling method that combines the
strengths of both dense and sparse techniques for a graph neural network.
SpaPool groups vertices into an adaptive number of clusters, leveraging the
benefits of both dense and sparse approaches. It aims to maintain the
structural integrity of the graph while reducing its size efficiently.
Experimental results on several datasets demonstrate that SpaPool achieves
competitive performance compared to existing pooling techniques and excels
particularly on small-scale graphs. This makes SpaPool a promising method for
applications requiring efficient and effective graph processing.

### 4. [A comparison between geostatistical and machine learning models for spatio-temporal prediction of PM2.5 data](http://arxiv.org/pdf/2509.12051v1)

Authors: Zeinab Mohamed, Wenlong Gong

Ambient air pollution poses significant health and environmental challenges.
Exposure to high concentrations of PM$_{2.5}$ have been linked to increased
respiratory and cardiovascular hospital admissions, more emergency department
visits and deaths. Traditional air quality monitoring systems such as
EPA-certified stations provide limited spatial and temporal data. The advent of
low-cost sensors has dramatically improved the granularity of air quality data,
enabling real-time, high-resolution monitoring. This study exploits the
extensive data from PurpleAir sensors to assess and compare the effectiveness
of various statistical and machine learning models in producing accurate hourly
PM$_{2.5}$ maps across California. We evaluate traditional geostatistical
methods, including kriging and land use regression, against advanced machine
learning approaches such as neural networks, random forests, and support vector
machines, as well as ensemble model. Our findings enhanced the predictive
accuracy of PM2.5 concentration by correcting the bias in PurpleAir data with
an ensemble model, which incorporating both spatiotemporal dependencies and
machine learning models.

### 5. [Preconditioned subgradient method for composite optimization: overparameterization and fast convergence](http://arxiv.org/pdf/2509.11486v1)

Authors: Mateo Díaz, Liwei Jiang, Abdel Ghani Labassi

Composite optimization problems involve minimizing the composition of a
smooth map with a convex function. Such objectives arise in numerous data
science and signal processing applications, including phase retrieval, blind
deconvolution, and collaborative filtering. The subgradient method achieves
local linear convergence when the composite loss is well-conditioned. However,
if the smooth map is, in a certain sense, ill-conditioned or overparameterized,
the subgradient method exhibits much slower sublinear convergence even when the
convex function is well-conditioned. To overcome this limitation, we introduce
a Levenberg-Morrison-Marquardt subgradient method that converges linearly under
mild regularity conditions at a rate determined solely by the convex function.
Further, we demonstrate that these regularity conditions hold for several
problems of practical interest, including square-variable formulations, matrix
sensing, and tensor factorization. Numerical experiments illustrate the
benefits of our method.

### 6. [High Effort, Low Gain: Fundamental Limits of Active Learning for Linear Dynamical Systems](http://arxiv.org/pdf/2509.11907v1)

Authors: Nicolas Chatzikiriakos, Kevin Jamieson, Andrea Iannelli

In this work, we consider the problem of identifying an unknown linear
dynamical system given a finite hypothesis class. In particular, we analyze the
effect of the excitation input on the sample complexity of identifying the true
system with high probability. To this end, we present sample complexity lower
bounds that capture the choice of the selected excitation input. The sample
complexity lower bound gives rise to a system theoretic condition to determine
the potential benefit of experiment design. Informed by the analysis of the
sample complexity lower bound, we propose a persistent excitation (PE)
condition tailored to the considered setting, which we then use to establish
sample complexity upper bounds. Notably, the \acs{PE} condition is weaker than
in the case of an infinite hypothesis class and allows analyzing different
excitation inputs modularly. Crucially, the lower and upper bounds share the
same dependency on key problem parameters. Finally, we leverage these insights
to propose an active learning algorithm that sequentially excites the system
optimally with respect to the current estimate, and provide sample complexity
guarantees for the presented algorithm. Concluding simulations showcase the
effectiveness of the proposed algorithm.

### 7. [Identifiable Autoregressive Variational Autoencoders for Nonlinear and Nonstationary Spatio-Temporal Blind Source Separation](http://arxiv.org/pdf/2509.11962v1)

Authors: Mika Sipilä, Klaus Nordhausen, Sara Taskinen

The modeling and prediction of multivariate spatio-temporal data involve
numerous challenges. Dimension reduction methods can significantly simplify
this process, provided that they account for the complex dependencies between
variables and across time and space. Nonlinear blind source separation has
emerged as a promising approach, particularly following recent advances in
identifiability results. Building on these developments, we introduce the
identifiable autoregressive variational autoencoder, which ensures the
identifiability of latent components consisting of nonstationary autoregressive
processes. The blind source separation efficacy of the proposed method is
showcased through a simulation study, where it is compared against
state-of-the-art methods, and the spatio-temporal prediction performance is
evaluated against several competitors on air pollution and weather datasets.

### 8. [Contractive kinetic Langevin samplers beyond global Lipschitz continuity](http://arxiv.org/pdf/2509.12031v1)

Authors: Iosif Lytras, Panagiotis Mertikopoulos

In this paper, we examine the problem of sampling from log-concave
distributions with (possibly) superlinear gradient growth under kinetic
(underdamped) Langevin algorithms. Using a carefully tailored taming scheme, we
propose two novel discretizations of the kinetic Langevin SDE, and we show that
they are both contractive and satisfy a log-Sobolev inequality. Building on
this, we establish a series of non-asymptotic bounds in $2$-Wasserstein
distance between the law reached by each algorithm and the underlying target
measure.

### 9. [Learning Neural Networks by Neuron Pursuit](http://arxiv.org/pdf/2509.12154v1)

Authors: Akshay Kumar, Jarvis Haupt

The first part of this paper studies the evolution of gradient flow for
homogeneous neural networks near a class of saddle points exhibiting a sparsity
structure. The choice of these saddle points is motivated from previous works
on homogeneous networks, which identified the first saddle point encountered by
gradient flow after escaping the origin. It is shown here that, when
initialized sufficiently close to such saddle points, gradient flow remains
near the saddle point for a sufficiently long time, during which the set of
weights with small norm remain small but converge in direction. Furthermore,
important empirical observations are made on the behavior of gradient descent
after escaping these saddle points. The second part of the paper, motivated by
these results, introduces a greedy algorithm to train deep neural networks
called Neuron Pursuit (NP). It is an iterative procedure which alternates
between expanding the network by adding neuron(s) with carefully chosen
weights, and minimizing the training loss using this augmented network. The
efficacy of the proposed algorithm is validated using numerical experiments.

### 10. [MMM: Clustering Multivariate Longitudinal Mixed-type Data](http://arxiv.org/pdf/2509.12166v1)

Authors: Francesco Amato, Julien Jacques

Multivariate longitudinal data of mixed-type are increasingly collected in
many science domains. However, algorithms to cluster this kind of data remain
scarce, due to the challenge to simultaneously model the within- and
between-time dependence structures for multivariate data of mixed kind. We
introduce the Mixture of Mixed-Matrices (MMM) model: reorganizing the data in a
three-way structure and assuming that the non-continuous variables are
observations of underlying latent continuous variables, the model relies on a
mixture of matrix-variate normal distributions to perform clustering in the
latent dimension. The MMM model is thus able to handle continuous, ordinal,
binary, nominal and count data and to concurrently model the heterogeneity, the
association among the responses and the temporal dependence structure in a
parsimonious way and without assuming conditional independence. The inference
is carried out through an MCMC-EM algorithm, which is detailed. An evaluation
of the model through synthetic data shows its inference abilities. A real-world
application on financial data is presented.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-09-16 PST.

### 1. [Neuromorphic principles in self-attention hardware for efficient transformers](https://www.nature.com/articles/s43588-025-00868-9)

Authors: Nathan Leroux et al.

### 2. [On the compatibility of generative AI and generative linguistics](https://www.nature.com/articles/s43588-025-00861-2)

Authors: Eva Portelance et al.

