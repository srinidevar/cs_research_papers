# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-06-12 17:05:04.421596 PST.

### Artificial Intelligence

### 1. [Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning](http://arxiv.org/pdf/2506.09498v1)

Authors: Jaesik Yoon, Hyeonseo Cho, Yoshua Bengio, Sungjin Ahn

Diffusion models have recently emerged as a powerful approach for trajectory
planning. However, their inherently non-sequential nature limits their
effectiveness in long-horizon reasoning tasks at test time. The recently
proposed Monte Carlo Tree Diffusion (MCTD) offers a promising solution by
combining diffusion with tree-based search, achieving state-of-the-art
performance on complex planning problems. Despite its strengths, our analysis
shows that MCTD incurs substantial computational overhead due to the sequential
nature of tree search and the cost of iterative denoising. To address this, we
propose Fast-MCTD, a more efficient variant that preserves the strengths of
MCTD while significantly improving its speed and scalability. Fast-MCTD
integrates two techniques: Parallel MCTD, which enables parallel rollouts via
delayed tree updates and redundancy-aware selection; and Sparse MCTD, which
reduces rollout length through trajectory coarsening. Experiments show that
Fast-MCTD achieves up to 100x speedup over standard MCTD while maintaining or
improving planning performance. Remarkably, it even outperforms Diffuser in
inference speed on some tasks, despite Diffuser requiring no search and
yielding weaker solutions. These results position Fast-MCTD as a practical and
scalable solution for diffusion-based inference-time reasoning.

### 2. [Application-Driven Value Alignment in Agentic AI Systems: Survey and Perspectives](http://arxiv.org/pdf/2506.09656v1)

Authors: Wei Zeng, Hengshu Zhu, Chuan Qin, Han Wu, Yihang Cheng, Sirui Zhang, Xiaowei Jin, Yinuo Shen, Zhenxing Wang, Feimin Zhong, Hui Xiong

The ongoing evolution of AI paradigms has propelled AI research into the
Agentic AI stage. Consequently, the focus of research has shifted from single
agents and simple applications towards multi-agent autonomous decision-making
and task collaboration in complex environments. As Large Language Models (LLMs)
advance, their applications become more diverse and complex, leading to
increasingly situational and systemic risks. This has brought significant
attention to value alignment for AI agents, which aims to ensure that an
agent's goals, preferences, and behaviors align with human values and societal
norms. This paper reviews value alignment in agent systems within specific
application scenarios. It integrates the advancements in AI driven by large
models with the demands of social governance. Our review covers value
principles, agent system application scenarios, and agent value alignment
evaluation. Specifically, value principles are organized hierarchically from a
top-down perspective, encompassing macro, meso, and micro levels. Agent system
application scenarios are categorized and reviewed from a general-to-specific
viewpoint. Agent value alignment evaluation systematically examines datasets
for value alignment assessment and relevant value alignment methods.
Additionally, we delve into value coordination among multiple agents within
agent systems. Finally, we propose several potential research directions in
this field.

### 3. [How Do People Revise Inconsistent Beliefs? Examining Belief Revision in Humans with User Studies](http://arxiv.org/pdf/2506.09977v1)

Authors: Stylianos Loukas Vasileiou, Antonio Rago, Maria Vanina Martinez, William Yeoh

Understanding how humans revise their beliefs in light of new information is
crucial for developing AI systems which can effectively model, and thus align
with, human reasoning. While theoretical belief revision frameworks rely on a
set of principles that establish how these operations are performed, empirical
evidence from cognitive psychology suggests that people may follow different
patterns when presented with conflicting information. In this paper, we present
three comprehensive user studies showing that people consistently prefer
explanation-based revisions, i.e., those which are guided by explanations, that
result in changes to their belief systems that are not necessarily captured by
classical belief change theory. Our experiments systematically investigate how
people revise their beliefs with explanations for inconsistencies, whether they
are provided with them or left to formulate them themselves, demonstrating a
robust preference for what may seem non-minimal revisions across different
types of scenarios. These findings have implications for AI systems designed to
model human reasoning or interact with humans, suggesting that such systems
should accommodate explanation-based, potentially non-minimal belief revision
operators to better align with human cognitive processes.

### 4. [Intelligent System of Emergent Knowledge: A Coordination Fabric for Billions of Minds](http://arxiv.org/pdf/2506.09335v1)

Authors: Moshi Wei, Sparks Li

The Intelligent System of Emergent Knowledge (ISEK) establishes a
decentralized network where human and artificial intelligence agents
collaborate as peers, forming a self-organizing cognitive ecosystem. Built on
Web3 infrastructure, ISEK combines three fundamental principles: (1) a
decentralized multi-agent architecture resistant to censorship, (2) symbiotic
AI-human collaboration with equal participation rights, and (3) resilient
self-adaptation through distributed consensus mechanisms.
  The system implements an innovative coordination protocol featuring a
six-phase workflow (Publish, Discover, Recruit, Execute, Settle, Feedback) for
dynamic task allocation, supported by robust fault tolerance and a
multidimensional reputation system. Economic incentives are governed by the
native $ISEK token, facilitating micropayments, governance participation, and
reputation tracking, while agent sovereignty is maintained through NFT-based
identity management.
  This synthesis of blockchain technology, artificial intelligence, and
incentive engineering creates an infrastructure that actively facilitates
emergent intelligence. ISEK represents a paradigm shift from conventional
platforms, enabling the organic development of large-scale, decentralized
cognitive systems where autonomous agents collectively evolve beyond
centralized constraints.

### 5. [Latent Multi-Head Attention for Small Language Models](http://arxiv.org/pdf/2506.09342v1)

Authors: Sushant Mehta, Raj Dandekar, Rajat Dandekar, Sreedath Panat

We present the first comprehensive study of latent multi-head attention (MLA)
for small language models, revealing interesting efficiency-quality trade-offs.
Training 30M-parameter GPT models on 100,000 synthetic stories, we benchmark
three architectural variants: standard multi-head attention (MHA), MLA, and MLA
with rotary positional embeddings (MLA+RoPE). Our key finding is that MLA+RoPE
with half-rank latent dimensions (r = d/2) achieves a 45% KV-cache memory
reduction while incurring only a 0.3% increase in validation loss (essentially
matching MHA quality)- a Pareto improvement for memory constrained deployment.
We further show that RoPE is crucial for MLA in small models: without it, MLA
underperforms vanilla attention by 3-5%, but with RoPE, it surpasses vanilla by
2%. Inference benchmarks on NVIDIA A100 GPUs reveal that MLA with r=d/2
achieves a 1.4 times speedup over full-rank MLA while maintaining the memory
savings. GPT-4 evaluations corroborate perplexity results, with ours achieving
the highest quality scores (7.4/10) across grammar, creativity, and consistency
metrics. Code and models will be released upon acceptance.

### 6. [ErrorEraser: Unlearning Data Bias for Improved Continual Learning](http://arxiv.org/pdf/2506.09347v1)

Authors: Xuemei Cao, Hanlin Gu, Xin Yang, Bingjun Wei, Haoyang Liang, Xiangkun Wang, Tianrui Li

Continual Learning (CL) primarily aims to retain knowledge to prevent
catastrophic forgetting and transfer knowledge to facilitate learning new
tasks. Unlike traditional methods, we propose a novel perspective: CL not only
needs to prevent forgetting, but also requires intentional forgetting.This
arises from existing CL methods ignoring biases in real-world data, leading the
model to learn spurious correlations that transfer and amplify across tasks.
From feature extraction and prediction results, we find that data biases
simultaneously reduce CL's ability to retain and transfer knowledge. To address
this, we propose ErrorEraser, a universal plugin that removes erroneous
memories caused by biases in CL, enhancing performance in both new and old
tasks. ErrorEraser consists of two modules: Error Identification and Error
Erasure. The former learns the probability density distribution of task data in
the feature space without prior knowledge, enabling accurate identification of
potentially biased samples. The latter ensures only erroneous knowledge is
erased by shifting the decision space of representative outlier samples.
Additionally, an incremental feature distribution learning strategy is designed
to reduce the resource overhead during error identification in downstream
tasks. Extensive experimental results show that ErrorEraser significantly
mitigates the negative impact of data biases, achieving higher accuracy and
lower forgetting rates across three types of CL methods. The code is available
at https://github.com/diadai/ErrorEraser.

### 7. ["Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions](http://arxiv.org/pdf/2506.09354v1)

Authors: Kellie Yu Hui Sim, Roy Ka-Wei Lee, Kenny Tsu Wei Choo

Mental health is a growing global concern, prompting interest in AI-driven
solutions to expand access to psychosocial support. Peer support, grounded in
lived experience, offers a valuable complement to professional care. However,
variability in training, effectiveness, and definitions raises concerns about
quality, consistency, and safety. Large Language Models (LLMs) present new
opportunities to enhance peer support interactions, particularly in real-time,
text-based interactions. We present and evaluate an AI-supported system with an
LLM-simulated distressed client, context-sensitive LLM-generated suggestions,
and real-time emotion visualisations. 2 mixed-methods studies with 12 peer
supporters and 5 mental health professionals (i.e., experts) examined the
system's effectiveness and implications for practice. Both groups recognised
its potential to enhance training and improve interaction quality. However, we
found a key tension emerged: while peer supporters engaged meaningfully,
experts consistently flagged critical issues in peer supporter responses, such
as missed distress cues and premature advice-giving. This misalignment
highlights potential limitations in current peer support training, especially
in emotionally charged contexts where safety and fidelity to best practices are
essential. Our findings underscore the need for standardised, psychologically
grounded training, especially as peer support scales globally. They also
demonstrate how LLM-supported systems can scaffold this development--if
designed with care and guided by expert oversight. This work contributes to
emerging conversations on responsible AI integration in mental health and the
evolving role of LLMs in augmenting peer-delivered care.

### 8. ["I Said Things I Needed to Hear Myself": Peer Support as an Emotional, Organisational, and Sociotechnical Practice in Singapore](http://arxiv.org/pdf/2506.09362v1)

Authors: Kellie Yu Hui Sim, Kenny Tsu Wei Choo

Peer support plays a vital role in expanding access to mental health care by
providing empathetic, community-based support outside formal clinical systems.
As digital platforms increasingly mediate such support, the design and impact
of these technologies remain under-examined, particularly in Asian contexts.
This paper presents findings from an interview study with 20 peer supporters in
Singapore, who operate across diverse online, offline, and hybrid environments.
Through a thematic analysis, we unpack how participants start, conduct, and
sustain peer support, highlighting their motivations, emotional labour, and the
sociocultural dimensions shaping their practices. Building on this grounded
understanding, we surface design directions for culturally responsive digital
tools that scaffold rather than supplant relational care. Drawing insights from
qualitative accounts, we offer a situated perspective on how AI might
responsibly augment peer support. This research contributes to human-centred
computing by articulating the lived realities of peer supporters and proposing
design implications for trustworthy and context-sensitive AI in mental health.

### 9. [COGENT: A Curriculum-oriented Framework for Generating Grade-appropriate Educational Content](http://arxiv.org/pdf/2506.09367v1)

Authors: Zhengyuan Liu, Stella Xin Yin, Dion Hoe-Lian Goh, Nancy F. Chen

While Generative AI has demonstrated strong potential and versatility in
content generation, its application to educational contexts presents several
challenges. Models often fail to align with curriculum standards and maintain
grade-appropriate reading levels consistently. Furthermore, STEM education
poses additional challenges in balancing scientific explanations with everyday
language when introducing complex and abstract ideas and phenomena to younger
students. In this work, we propose COGENT, a curriculum-oriented framework for
generating grade-appropriate educational content. We incorporate three
curriculum components (science concepts, core ideas, and learning objectives),
control readability through length, vocabulary, and sentence complexity, and
adopt a ``wonder-based'' approach to increase student engagement and interest.
We conduct a multi-dimensional evaluation via both LLM-as-a-judge and human
expert analysis. Experimental results show that COGENT consistently produces
grade-appropriate passages that are comparable or superior to human references.
Our work establishes a viable approach for scaling adaptive and high-quality
learning resources.

### 10. [Anomaly Detection and Generation with Diffusion Models: A Survey](http://arxiv.org/pdf/2506.09368v1)

Authors: Yang Liu, Jing Liu, Chengfang Li, Rui Xi, Wenchao Li, Liang Cao, Jin Wang, Laurence T. Yang, Junsong Yuan, Wei Zhou

Anomaly detection (AD) plays a pivotal role across diverse domains, including
cybersecurity, finance, healthcare, and industrial manufacturing, by
identifying unexpected patterns that deviate from established norms in
real-world data. Recent advancements in deep learning, specifically diffusion
models (DMs), have sparked significant interest due to their ability to learn
complex data distributions and generate high-fidelity samples, offering a
robust framework for unsupervised AD. In this survey, we comprehensively review
anomaly detection and generation with diffusion models (ADGDM), presenting a
tutorial-style analysis of the theoretical foundations and practical
implementations and spanning images, videos, time series, tabular, and
multimodal data. Crucially, unlike existing surveys that often treat anomaly
detection and generation as separate problems, we highlight their inherent
synergistic relationship. We reveal how DMs enable a reinforcing cycle where
generation techniques directly address the fundamental challenge of anomaly
data scarcity, while detection methods provide critical feedback to improve
generation fidelity and relevance, advancing both capabilities beyond their
individual potential. A detailed taxonomy categorizes ADGDM methods based on
anomaly scoring mechanisms, conditioning strategies, and architectural designs,
analyzing their strengths and limitations. We final discuss key challenges
including scalability and computational efficiency, and outline promising
future directions such as efficient architectures, conditioning strategies, and
integration with foundation models (e.g., visual-language models and large
language models). By synthesizing recent advances and outlining open research
questions, this survey aims to guide researchers and practitioners in
leveraging DMs for innovative AD solutions across diverse applications.

### Hardware Architecture

### 1. [Exploiting Control-flow Enforcement Technology for Sound and Precise Static Binary Disassembly](http://arxiv.org/pdf/2506.09426v1)

Authors: Brian Zhao, Yiwei Yang, Yusheng Zheng, Andi Quinn

Rewriting x86_64 binaries-whether for security hardening, dynamic
instrumentation, or performance profiling is notoriously difficult due to
variable-length instructions, interleaved code and data, and indirect jumps to
arbitrary byte offsets. Existing solutions (e.g., "superset disassembly")
ensure soundness but incur significant overhead and produce large rewritten
binaries, especially for on-the-fly instrumentation. This paper addresses these
challenges by introducing the Time Variance Authority (TVA), which leverages
Intel's Control-Flow Enforcement Technology (CET). By recognizing endbr64 as
the only valid indirect jump target, TVA prunes spurious disassembly paths
while preserving soundness and emulates CET constraints on processors lacking
native CET support, effectively mitigating ROP/JOP exploits without new
hardware. We implement TVA by modernizing the Multiverse rewriter for 64-bit
Linux. Our evaluation on SPEC CPU2017 and real-world applications shows that
TVA-guided rewriting achieves up to 1.3x faster instrumentation time. These
results underscore TVA's feasibility as a high-performance, uprobes-free
alternative for robust x86_64 binary analysis and rewriting.

### 2. [FPGA-Based Multiplier with a New Approximate Full Adder for Error-Resilient Applications](http://arxiv.org/pdf/2506.09596v1)

Authors: Ali Ranjbar, Elham Esmaeili, Roghayeh Rafieisangari, Nabiollah Shiri

Electronic devices primarily aim to offer low power consumption, high speed,
and a compact area. The performance of very large-scale integration (VLSI)
devices is influenced by arithmetic operations, where multiplication is a
crucial operation. Therefore, a high-speed multiplier is essential for
developing any signal-processing module. Numerous multipliers have been
reviewed in existing literature, and their speed is largely determined by how
partial products (PPs) are accumulated. To enhance the speed of multiplication
beyond current methods, an approximate adder-based multiplier is introduced.
This approach allows for the simultaneous addition of PPs from two consecutive
bits using a novel approximate adder. The proposed multiplier is utilized in a
mean filter structure and implemented in ISE Design Suite 14.7 using VHDL and
synthesized on the Xilinx Spartan3-XC3S400 FPGA board. Compared to the
literature, the proposed multiplier achieves power and power-delay product
(PDP) improvements of 56.09% and 73.02%, respectively. The validity of the
expressed multiplier is demonstrated through the mean filter system. Results
show that it achieves power savings of 33.33%. Additionally, the proposed
multiplier provides more accurate results than other approximate multipliers by
expressing higher values of peak signal-to-noise ratio (PSNR), (30.58%), and
structural similarity index metric (SSIM), (22.22%), while power consumption is
in a low range.

### 3. [Efficient Modular Multiplier over GF (2^m) for ECPM](http://arxiv.org/pdf/2506.09464v1)

Authors: Ruby Kumari, Gaurav Purohit, Abhijit Karmakar

Elliptic curve cryptography (ECC) has emerged as the dominant public-key
protocol, with NIST standardizing parameters for binary field GF(2^m) ECC
systems. This work presents a hardware implementation of a Hybrid
Multiplication technique for modular multiplication over binary field GF(2m),
targeting NIST B-163, 233, 283, and 571 parameters. The design optimizes the
combination of conventional multiplication (CM) and Karatsuba multiplication
(KM) to enhance elliptic curve point multiplication (ECPM). The key innovation
uses CM for smaller operands (up to 41 bits for m=163) and KM for larger ones,
reducing computational complexity and enhancing efficiency. The design is
evaluated in three areas: Resource Utilization For m=163, the hybrid design
uses 6,812 LUTs, a 39.82% reduction compared to conventional methods. For
m=233, LUT usage reduces by 45.53% and 70.70% compared to overlap-free and
bit-parallel implementations. Delay Performance For m=163, achieves 13.31ns
delay, improving by 37.60% over bit-parallel implementations. For m=233,
maintains 13.39ns delay. Area-Delay Product For m=163, achieves ADP of 90,860,
outperforming bit-parallel (75,337) and digit-serial (43,179) implementations.
For m=233, demonstrates 16.86% improvement over overlap-free and 96.10% over
bit-parallel designs. Results show the hybrid technique significantly improves
speed, hardware efficiency, and resource utilization for ECC cryptographic
systems.

### 4. [Mainframe-style channel controllers for modern disaggregated memory systems](http://arxiv.org/pdf/2506.09758v1)

Authors: Zikai Liu, Jasmin Schult, Pengcheng Xu, Timothy Roscoe

Despite the promise of alleviating the main memory bottleneck, and the
existence of commercial hardware implementations, techniques for Near-Data
Processing have seen relatively little real-world deployment. The idea has
received renewed interest with the appearance of disaggregated or "far" memory,
for example in the use of CXL memory pools.
  However, we argue that the lack of a clear OS-centric abstraction of
Near-Data Processing is a major barrier to adoption of the technology. Inspired
by the channel controllers which interface the CPU to disk drives in mainframe
systems, we propose memory channel controllers as a convenient, portable, and
virtualizable abstraction of Near-Data Processing for modern disaggregated
memory systems.
  In addition to providing a clean abstraction that enables OS integration
while requiring no changes to CPU architecture, memory channel controllers
incorporate another key innovation: they exploit the cache coherence provided
by emerging interconnects to provide a much richer programming model, with more
fine-grained interaction, than has been possible with existing designs.

### 5. [On the Impossibility of a Perfect Hypervisor](http://arxiv.org/pdf/2506.09825v1)

Authors: Mordechai Guri

We establish a fundamental impossibility result for a `perfect hypervisor',
one that (1) preserves every observable behavior of any program exactly as on
bare metal and (2) adds zero timing or resource overhead.
  Within this model we prove two theorems. (1) Indetectability Theorem. If such
a hypervisor existed, no guest-level program, measurement, or timing test could
distinguish it from native execution; all traces, outputs, and timings would be
identical.
  (2) Impossibility Theorem. Despite that theoretical indetectability, a
perfect hypervisor cannot exist on any machine with finite computational
resources.
  These results are architecture-agnostic and extend beyond hypervisors to any
virtualization layer emulators, sandboxes, containers, or
runtime-instrumentation frameworks. Together they provide a formal foundation
for future work on the principles and limits of virtualization.

### Computational Engineering

### 1. [Large Language Models for Design Structure Matrix Optimization](http://arxiv.org/pdf/2506.09749v1)

Authors: Shuo Jiang, Min Xie, Jianxi Luo

In complex engineering systems, the interdependencies among components or
development activities are often modeled and analyzed using Design Structure
Matrix (DSM). Reorganizing elements within a DSM to minimize feedback loops and
enhance modularity or process efficiency constitutes a challenging
combinatorial optimization (CO) problem in engineering design and operations.
As problem sizes increase and dependency networks become more intricate,
traditional optimization methods that solely use mathematical heuristics often
fail to capture the contextual nuances and struggle to deliver effective
solutions. In this study, we explore the potential of Large Language Models
(LLMs) for helping solve such CO problems by leveraging their capabilities for
advanced reasoning and contextual understanding. We propose a novel LLM-based
framework that integrates network topology with contextual domain knowledge for
iterative optimization of DSM element sequencing - a common CO problem.
Experiments on various DSM cases show that our method consistently achieves
faster convergence and superior solution quality compared to both stochastic
and deterministic baselines. Notably, we find that incorporating contextual
domain knowledge significantly enhances optimization performance regardless of
the chosen LLM backbone. These findings highlight the potential of LLMs to
solve complex engineering CO problems by combining semantic and mathematical
reasoning. This approach paves the way towards a new paradigm in LLM-based
engineering design optimization.

### 2. [Intelligent Design 4.0: Paradigm Evolution Toward the Agentic AI Era](http://arxiv.org/pdf/2506.09755v1)

Authors: Shuo Jiang, Min Xie, Frank Youhua Chen, Jian Ma, Jianxi Luo

Research and practice in Intelligent Design (ID) have significantly enhanced
engineering innovation, efficiency, quality, and productivity over recent
decades, fundamentally reshaping how engineering designers think, behave, and
interact with design processes. The recent emergence of Foundation Models
(FMs), particularly Large Language Models (LLMs), has demonstrated general
knowledge-based reasoning capabilities, and open new paths and avenues for
further transformation in engineering design. In this context, this paper
introduces Intelligent Design 4.0 (ID 4.0) as an emerging paradigm empowered by
agentic AI systems. We review the historical evolution of ID across four
distinct stages: rule-based expert systems, task-specific machine learning
models, large-scale foundation AI models, and the recent emerging paradigm of
multi-agent collaboration. We propose a conceptual framework for ID 4.0 and
discuss its potential to support end-to-end automation of engineering design
processes through coordinated, autonomous multi-agent-based systems.
Furthermore, we discuss future perspectives to enhance and fully realize ID
4.0's potential, including more complex design scenarios, more practical design
implementations, novel agent coordination mechanisms, and autonomous design
goal-setting with better human value alignment. In sum, these insights lay a
foundation for advancing Intelligent Design toward greater adaptivity,
autonomy, and effectiveness in addressing increasingly complex design
challenges.

### 3. [Superstudent intelligence in thermodynamics](http://arxiv.org/pdf/2506.09822v1)

Authors: Rebecca Loubet, Pascal Zittlau, Marco Hoffmann, Luisa Vollmer, Sophie Fellenz, Heike Leitte, Fabian Jirasek, Johannes Lenhard, Hans Hasse

In this short note, we report and analyze a striking event: OpenAI's large
language model o3 has outwitted all students in a university exam on
thermodynamics. The thermodynamics exam is a difficult hurdle for most
students, where they must show that they have mastered the fundamentals of this
important topic. Consequently, the failure rates are very high, A-grades are
rare - and they are considered proof of the students' exceptional intellectual
abilities. This is because pattern learning does not help in the exam. The
problems can only be solved by knowledgeably and creatively combining
principles of thermodynamics. We have given our latest thermodynamics exam not
only to the students but also to OpenAI's most powerful reasoning model, o3,
and have assessed the answers of o3 exactly the same way as those of the
students. In zero-shot mode, the model o3 solved all problems correctly, better
than all students who took the exam; its overall score was in the range of the
best scores we have seen in more than 10,000 similar exams since 1985. This is
a turning point: machines now excel in complex tasks, usually taken as proof of
human intellectual capabilities. We discuss the consequences this has for the
work of engineers and the education of future engineers.

### 4. [Natural Language Guided Ligand-Binding Protein Design](http://arxiv.org/pdf/2506.09332v1)

Authors: Zhenqiao Song, Ramith Hettiarachchi, Chuan Li, Jianwen Xie, Lei Li

Can AI protein models follow human language instructions and design proteins
with desired functions (e.g. binding to a ligand)? Designing proteins that bind
to a given ligand is crucial in a wide range of applications in biology and
chemistry. Most prior AI models are trained on protein-ligand complex data,
which is scarce due to the high cost and time requirements of laboratory
experiments. In contrast, there is a substantial body of human-curated text
descriptions about protein-ligand interactions and ligand formula. In this
paper, we propose InstructPro, a family of protein generative models that
follow natural language instructions to design ligand-binding proteins. Given a
textual description of the desired function and a ligand formula in SMILES,
InstructPro generates protein sequences that are functionally consistent with
the specified instructions. We develop the model architecture, training
strategy, and a large-scale dataset, InstructProBench, to support both training
and evaluation. InstructProBench consists of 9,592,829 triples of (function
description, ligand formula, protein sequence). We train two model variants:
InstructPro-1B (with 1 billion parameters) and InstructPro-3B~(with 3 billion
parameters). Both variants consistently outperform strong baselines, including
ProGen2, ESM3, and Pinal. Notably, InstructPro-1B achieves the highest docking
success rate (81.52% at moderate confidence) and the lowest average root mean
square deviation (RMSD) compared to ground truth structures (4.026{\AA}).
InstructPro-3B further descreases the average RMSD to 2.527{\AA}, demonstrating
InstructPro's ability to generate ligand-binding proteins that align with the
functional specifications.

### 5. [Causal Climate Emulation with Bayesian Filtering](http://arxiv.org/pdf/2506.09891v1)

Authors: Sebastian Hickman, Ilija Trajkovic, Julia Kaltenborn, Francis Pelletier, Alex Archibald, Yaniv Gurwicz, Peer Nowack, David Rolnick, Julien Boussard

Traditional models of climate change use complex systems of coupled equations
to simulate physical processes across the Earth system. These simulations are
highly computationally expensive, limiting our predictions of climate change
and analyses of its causes and effects. Machine learning has the potential to
quickly emulate data from climate models, but current approaches are not able
to incorporate physics-informed causal relationships. Here, we develop an
interpretable climate model emulator based on causal representation learning.
We derive a physics-informed approach including a Bayesian filter for stable
long-term autoregressive emulation. We demonstrate that our emulator learns
accurate climate dynamics, and we show the importance of each one of its
components on a realistic synthetic dataset and data from two widely deployed
climate models.

### 6. [A Note on the Reliability of Goal-Oriented Error Estimates for Galerkin Finite Element Methods with Nonlinear Functionals](http://arxiv.org/pdf/2506.09913v1)

Authors: Brian N. Granzow, Stephen D. Bond, D. Thomas Seidl, Bernhard Endtmayer

We consider estimating the discretization error in a nonlinear functional
$J(u)$ in the setting of an abstract variational problem: find $u \in
\mathcal{V}$ such that $B(u,\varphi) = L(\varphi) \; \forall \varphi \in
\mathcal{V}$, as approximated by a Galerkin finite element method. Here,
$\mathcal{V}$ is a Hilbert space, $B(\cdot,\cdot)$ is a bilinear form, and
$L(\cdot)$ is a linear functional. We consider well-known error estimates
$\eta$ of the form $J(u) - J(u_h) \approx \eta = L(z) - B(u_h, z)$, where $u_h$
denotes a finite element approximation to $u$, and $z$ denotes the solution to
an auxiliary adjoint variational problem. We show that there exist nonlinear
functionals for which error estimates of this form are not reliable, even in
the presence of an exact adjoint solution solution $z$. An estimate $\eta$ is
said to be reliable if there exists a constant $C \in \mathbb{R}_{>0}$
independent of $u_h$ such that $|J(u) - J(u_h)| \leq C|\eta|$. We present
several example pairs of bilinear forms and nonlinear functionals where
reliability of $\eta$ is not achieved.

### Computational Geometry

### 1. [Power Diagram Enhanced Adaptive Isosurface Extraction from Signed Distance Fields](http://arxiv.org/pdf/2506.09579v1)

Authors: Pengfei Wang, Ziyang Zhang, Wensong Wang, Shuangmin Chen, Lin Lu, Shiqing Xin, Changhe Tu

Extracting high-fidelity mesh surfaces from Signed Distance Fields has become
a fundamental operation in geometry processing. Despite significant progress
over the past decades, key challenges remain namely, how to automatically
capture the intricate geometric and topological structures encoded in the zero
level set of SDFs. In this paper, we present a novel isosurface extraction
algorithm that introduces two key innovations: 1. An incrementally constructed
power diagram through the addition of sample points, which enables repeated
updates to the extracted surface via its dual regular Delaunay
tetrahedralization; and 2. An adaptive point insertion strategy that identifies
regions exhibiting the greatest discrepancy between the current mesh and the
underlying continuous surface. As the teaser figure shows, our framework
progressively refines the extracted mesh with minimal computational cost until
it sufficiently approximates the underlying surface. Experimental results
demonstrate that our approach outperforms sofa methods, particularly for models
with intricate geometric variations and complex topologies.

### 2. [Don't be Afraid of Cell Complexes! An Introduction from an Applied Perspective](http://arxiv.org/pdf/2506.09726v1)

Authors: Josef Hoppe, Vincent P. Grande, Michael T. Schaub

Cell complexes (CCs) are a higher-order network model deeply rooted in
algebraic topology that has gained interest in signal processing and network
science recently. However, while the processing of signals supported on CCs can
be described in terms of easily-accessible algebraic or combinatorial notions,
the commonly presented definition of CCs is grounded in abstract concepts from
topology and remains disconnected from the signal processing methods developed
for CCs. In this paper, we aim to bridge this gap by providing a simplified
definition of CCs that is accessible to a wider audience and can be used in
practical applications. Specifically, we first introduce a simplified notion of
abstract regular cell complexes (ARCCs). These ARCCs only rely on notions from
algebra and can be shown to be equivalent to regular cell complexes for most
practical applications. Second, using this new definition we provide an
accessible introduction to (abstract) cell complexes from a perspective of
network science and signal processing. Furthermore, as many practical
applications work with CCs of dimension 2 and below, we provide an even simpler
definition for this case that significantly simplifies understanding and
working with CCs in practice.

### 3. [Crossing numbers of dense graphs on surfaces](http://arxiv.org/pdf/2506.09974v1)

Authors: Alfredo Hubard, Arnaud de Mesmay, Hugo Parlier

In this paper, we provide upper and lower bounds on the crossing numbers of
dense graphs on surfaces, which match up to constant factors. First, we prove
that if $G$ is a dense enough graph with $m$ edges and $\Sigma$ is a surface of
genus $g$, then any drawing of $G$ on $\Sigma$ incurs at least $\Omega
\left(\frac{m^2}{g} \log ^2 g\right)$ crossings. The poly-logarithmic factor in
this lower bound is new even in the case of complete graphs and disproves a
conjecture of Shahrokhi, Sz\'ekely and Vrt'o from 1996. Then we prove a
geometric converse to this lower bound: we provide an explicit family of
hyperbolic surfaces such that for any graph $G$, sampling the vertices
uniformly at random on this surface and connecting them with shortest paths
yields $O\left(\frac{m^2}{g} \log ^2 g\right)$ crossings in expectation.

### Computation and Language

### 1. [Towards Efficient and Effective Alignment of Large Language Models](http://arxiv.org/pdf/2506.09329v1)

Authors: Yuxin Jiang

Large language models (LLMs) exhibit remarkable capabilities across diverse
tasks, yet aligning them efficiently and effectively with human expectations
remains a critical challenge. This thesis advances LLM alignment by introducing
novel methodologies in data collection, training, and evaluation. We first
address alignment data collection. Existing approaches rely heavily on manually
curated datasets or proprietary models. To overcome these limitations, we
propose Lion, an adversarial distillation framework that iteratively refines
training data by identifying and generating challenging instructions, enabling
state-of-the-art zero-shot reasoning. Additionally, we introduce Web
Reconstruction (WebR), a fully automated framework that synthesizes
instruction-tuning data directly from raw web documents, significantly
improving data diversity and scalability over existing synthetic data methods.
Next, we enhance alignment training through novel optimization techniques. We
develop Learning to Edit (LTE), a framework that enables LLMs to efficiently
integrate new knowledge while preserving existing information. LTE leverages
meta-learning to improve both real-time and batch knowledge updates.
Furthermore, we introduce Bridging and Modeling Correlations (BMC), a
refinement of Direct Preference Optimization (DPO) that explicitly captures
token-level correlations in preference data, leading to superior alignment
across QA and mathematical reasoning tasks. Finally, we tackle the challenge of
evaluating alignment. Existing benchmarks emphasize response quality but
overlook adherence to specific constraints. To bridge this gap, we introduce
FollowBench, a multi-level, fine-grained benchmark assessing LLMs' ability to
follow complex constraints across diverse instruction types. Our results expose
key weaknesses in current models' constraint adherence, offering insights for
future improvements.

### 2. [OmniDRCA: Parallel Speech-Text Foundation Model via Dual-Resolution Speech Representations and Contrastive Alignment](http://arxiv.org/pdf/2506.09349v1)

Authors: Chao-Hong Tan, Qian Chen, Wen Wang, Chong Deng, Qinglin Zhang, Luyao Cheng, Hai Yu, Xin Zhang, Xiang Lv, Tianyu Zhao, Chong Zhang, Yukun Ma, Yafeng Chen, Hui Wang, Jiaqing Liu, Jieping Ye

Recent studies on end-to-end speech generation with large language models
(LLMs) have attracted significant community attention, with multiple works
extending text-based LLMs to generate discrete speech tokens. Existing
approaches primarily fall into two categories: (1) Methods that generate
discrete speech tokens independently without incorporating them into the LLM's
autoregressive process, resulting in text generation being unaware of
concurrent speech synthesis. (2) Models that generate interleaved or parallel
speech-text tokens through joint autoregressive modeling, enabling mutual
modality awareness during generation. This paper presents OmniDRCA, a parallel
speech-text foundation model based on joint autoregressive modeling, featuring
dual-resolution speech representations and contrastive cross-modal alignment.
Our approach processes speech and text representations in parallel while
enhancing audio comprehension through contrastive alignment. Experimental
results on Spoken Question Answering benchmarks demonstrate that OmniDRCA
establishes new state-of-the-art (SOTA) performance among parallel joint
speech-text modeling based foundation models, and achieves competitive
performance compared to interleaved models. Additionally, we explore the
potential of extending the framework to full-duplex conversational scenarios.

### 3. [DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](http://arxiv.org/pdf/2506.09351v1)

Authors: Yuchen Feng, Bowen Shen, Naibin Gu, Jiaxuan Zhao, Peng Fu, Zheng Lin, Weiping Wang

Large language models (LLMs) with the Mixture-of-Experts (MoE) architecture
achieve high cost-efficiency by selectively activating a subset of the
parameters. Despite the inference efficiency of MoE LLMs, the training of
extensive experts from scratch incurs substantial overhead, whereas
reconstructing a dense LLM into an MoE LLM significantly reduces the training
budget. However, existing reconstruction methods often overlook the diversity
among experts, leading to potential redundancy. In this paper, we come up with
the observation that a specific LLM exhibits notable diversity after being
pruned on different calibration datasets, based on which we present a
Diversity-Enhanced reconstruction method named DIVE. The recipe of DIVE
includes domain affinity mining, pruning-based expert reconstruction, and
efficient retraining. Specifically, the reconstruction includes pruning and
reassembly of the feed-forward network (FFN) module. After reconstruction, we
efficiently retrain the model on routers, experts and normalization modules. We
implement DIVE on Llama-style LLMs with open-source training corpora.
Experiments show that DIVE achieves training efficiency with minimal accuracy
trade-offs, outperforming existing pruning and MoE reconstruction methods with
the same number of activated parameters.

### 4. [Taming SQL Complexity: LLM-Based Equivalence Evaluation for Text-to-SQL](http://arxiv.org/pdf/2506.09359v1)

Authors: Qingyun Zeng, Simin Ma, Arash Niknafs, Ashish Basran, Carol Szabo

The rise of Large Language Models (LLMs) has significantly advanced
Text-to-SQL (NL2SQL) systems, yet evaluating the semantic equivalence of
generated SQL remains a challenge, especially given ambiguous user queries and
multiple valid SQL interpretations. This paper explores using LLMs to assess
both semantic and a more practical "weak" semantic equivalence. We analyze
common patterns of SQL equivalence and inequivalence, discuss challenges in
LLM-based evaluation.

### 5. [Binary classification for perceived quality of headlines and links on worldwide news websites, 2018-2024](http://arxiv.org/pdf/2506.09381v1)

Authors: Austin McCutcheon, Thiago E. A. de Oliveira, Aleksandr Zheleznov, Chris Brogly

The proliferation of online news enables potential widespread publication of
perceived low-quality news headlines/links. As a result, we investigated
whether it was possible to automatically distinguish perceived lower-quality
news headlines/links from perceived higher-quality headlines/links. We
evaluated twelve machine learning models on a binary, balanced dataset of
57,544,214 worldwide news website links/headings from 2018-2024 (28,772,107 per
class) with 115 extracted linguistic features. Binary labels for each text were
derived from scores based on expert consensus regarding the respective news
domain quality. Traditional ensemble methods, particularly the bagging
classifier, had strong performance (88.1% accuracy, 88.3% F1, 80/20 train/test
split). Fine-tuned DistilBERT achieved the highest accuracy (90.3%, 80/20
train/test split) but required more training time. The results suggest that
both NLP features with traditional classifiers and deep learning models can
effectively differentiate perceived news headline/link quality, with some
trade-off between predictive performance and train time.

### 6. [Comparing human and LLM politeness strategies in free production](http://arxiv.org/pdf/2506.09391v1)

Authors: Haoran Zhao, Robert D. Hawkins

Polite speech poses a fundamental alignment challenge for large language
models (LLMs). Humans deploy a rich repertoire of linguistic strategies to
balance informational and social goals -- from positive approaches that build
rapport (compliments, expressions of interest) to negative strategies that
minimize imposition (hedging, indirectness). We investigate whether LLMs employ
a similarly context-sensitive repertoire by comparing human and LLM responses
in both constrained and open-ended production tasks. We find that larger models
($\ge$70B parameters) successfully replicate key preferences from the
computational pragmatics literature, and human evaluators surprisingly prefer
LLM-generated responses in open-ended contexts. However, further linguistic
analyses reveal that models disproportionately rely on negative politeness
strategies even in positive contexts, potentially leading to
misinterpretations. While modern LLMs demonstrate an impressive handle on
politeness strategies, these subtle differences raise important questions about
pragmatic alignment in AI systems.

### 7. [A Hierarchical Probabilistic Framework for Incremental Knowledge Tracing in Classroom Settings](http://arxiv.org/pdf/2506.09393v1)

Authors: Xinyi Gao, Qiucheng Wu, Yang Zhang, Xuechen Liu, Kaizhi Qian, Ying Xu, Shiyu Chang

Knowledge tracing (KT) aims to estimate a student's evolving knowledge state
and predict their performance on new exercises based on performance history.
Many realistic classroom settings for KT are typically low-resource in data and
require online updates as students' exercise history grows, which creates
significant challenges for existing KT approaches. To restore strong
performance under low-resource conditions, we revisit the hierarchical
knowledge concept (KC) information, which is typically available in many
classroom settings and can provide strong prior when data are sparse. We
therefore propose Knowledge-Tree-based Knowledge Tracing (KT$^2$), a
probabilistic KT framework that models student understanding over a
tree-structured hierarchy of knowledge concepts using a Hidden Markov Tree
Model. KT$^2$ estimates student mastery via an EM algorithm and supports
personalized prediction through an incremental update mechanism as new
responses arrive. Our experiments show that KT$^2$ consistently outperforms
strong baselines in realistic online, low-resource settings.

### 8. [Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings](http://arxiv.org/pdf/2506.09424v1)

Authors: Md Messal Monem Miah, Adrita Anika, Xi Shi, Ruihong Huang

Detecting deception in an increasingly digital world is both a critical and
challenging task. In this study, we present a comprehensive evaluation of the
automated deception detection capabilities of Large Language Models (LLMs) and
Large Multimodal Models (LMMs) across diverse domains. We assess the
performance of both open-source and commercial LLMs on three distinct datasets:
real life trial interviews (RLTD), instructed deception in interpersonal
scenarios (MU3D), and deceptive reviews (OpSpam). We systematically analyze the
effectiveness of different experimental setups for deception detection,
including zero-shot and few-shot approaches with random or similarity-based
in-context example selection. Our results show that fine-tuned LLMs achieve
state-of-the-art performance on textual deception detection tasks, while LMMs
struggle to fully leverage cross-modal cues. Additionally, we analyze the
impact of auxiliary features, such as non-verbal gestures and video summaries,
and examine the effectiveness of different prompting strategies, including
direct label generation and chain-of-thought reasoning. Our findings provide
key insights into how LLMs process and interpret deceptive cues across
modalities, highlighting their potential and limitations in real-world
deception detection applications.

### 9. [Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning](http://arxiv.org/pdf/2506.09501v1)

Authors: Jiayi Yuan, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan, Jing Shi, Xia Hu, Zirui Liu

Large Language Models (LLMs) are now integral across various domains and have
demonstrated impressive performance. Progress, however, rests on the premise
that benchmark scores are both accurate and reproducible. We demonstrate that
the reproducibility of LLM performance is fragile: changing system
configuration such as evaluation batch size, GPU count, and GPU version can
introduce significant difference in the generated responses. This issue is
especially pronounced in reasoning models, where minor rounding differences in
early tokens can cascade into divergent chains of thought, ultimately affecting
accuracy. For instance, under bfloat16 precision with greedy decoding, a
reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation
in accuracy and 9,000 tokens difference in response length due to differences
in GPU count, type, and evaluation batch size. We trace the root cause of this
variability to the non-associative nature of floating-point arithmetic under
limited numerical precision. This work presents the first systematic
investigation into how numerical precision affects reproducibility in LLM
inference. Through carefully controlled experiments across various hardware,
software, and precision settings, we quantify when and how model outputs
diverge. Our analysis reveals that floating-point precision -- while critical
for reproducibility -- is often neglected in evaluation practices. Inspired by
this, we develop a lightweight inference pipeline, dubbed LayerCast, that
stores weights in 16-bit precision but performs all computations in FP32,
balancing memory efficiency with numerical stability. Code is available at
https://github.com/nanomaoli/llm_reproducibility.

### 10. [KG-Infused RAG: Augmenting Corpus-Based RAG with External Knowledge Graphs](http://arxiv.org/pdf/2506.09542v1)

Authors: Dingjun Wu, Yukun Yan, Zhenghao Liu, Zhiyuan Liu, Maosong Sun

Retrieval-Augmented Generation (RAG) improves factual accuracy by grounding
responses in external knowledge. However, existing methods typically rely on a
single source, either unstructured text or structured knowledge. Moreover, they
lack cognitively inspired mechanisms for activating relevant knowledge. To
address these issues, we propose KG-Infused RAG, a framework that integrates
KGs into RAG systems to implement spreading activation, a cognitive process
that enables concept association and inference. KG-Infused RAG retrieves KG
facts, expands the query accordingly, and enhances generation by combining
corpus passages with structured facts, enabling interpretable, multi-source
retrieval grounded in semantic structure. We further improve KG-Infused RAG via
preference learning on sampled key stages in the pipeline. Experiments on five
QA benchmarks show that KG-Infused RAG consistently outperforms vanilla RAG (by
3.8% to 13.8%). Additionally, when integrated into Self-RAG, KG-Infused RAG
brings further performance gains, demonstrating its effectiveness and
versatility as a plug-and-play enhancement module for corpus-based RAG methods.

### Cryptography and Security

### 1. [ContextBuddy: AI-Enhanced Contextual Insights for Security Alert Investigation (Applied to Intrusion Detection)](http://arxiv.org/pdf/2506.09365v1)

Authors: Ronal Singh, Mohan Baruwal Chhetri, Surya Nepal, Cecile Paris

Modern Security Operations Centres (SOCs) integrate diverse tools, such as
SIEM, IDS, and XDR systems, offering rich contextual data, including alert
enrichments, flow features, and similar case histories. Yet, analysts must
still manually determine which of these contextual cues are most relevant when
validating specific alerts. We introduce ContextBuddy, an AI assistant that
learns from analysts' prior investigations to help them identify the most
relevant context for new alerts. Rather than providing enrichments,
ContextBuddy models how analysts have previously selected context and suggests
tailored cues based on the characteristics of each alert. We formulate context
selection as a sequential decision-making problem and apply imitation learning
(IL) to capture analysts' strategies, evaluating multiple IL approaches.
Through staged evaluation, we validate ContextBuddy using two intrusion
detection datasets (HIKARI-2021, UNSW-NB15). In simulation-based experiments,
ContextBuddy helped simulated reinforcement learning analysts improve
classification accuracy (p < 0.001) (increasing F1 by 2.5% for HIKARI and 9%
for UNSW), reducing false negatives (1.5% for HIKARI and 10% for UNSW), and
keeping false positives below 1%. Decision confidence among agents also
improved by 2-3% (p < 0.001). In a within-subject user study (N=13; power =
0.8), non-experts using ContextBuddy improved classification accuracy by 21.1%
(p = 0.008) and reduced alert validation time by 24% (p = 0.01). These results
demonstrate that by learning context-selection patterns from analysts,
ContextBuddy can yield notable improvements in investigation effectiveness and
efficiency.

### 2. [Epass: Efficient and Privacy-Preserving Asynchronous Payment on Blockchain](http://arxiv.org/pdf/2506.09387v1)

Authors: Weijie Wang, Jinwen Liang, Chuan Zhang, Ximeng Liu, Liehuang Zhu, Song Guo

Buy Now Pay Later (BNPL) is a rapidly proliferating e-commerce model,
offering consumers to get the product immediately and defer payments.
Meanwhile, emerging blockchain technologies endow BNPL platforms with digital
currency transactions, allowing BNPL platforms to integrate with digital
wallets. However, the transparency of transactions causes critical privacy
concerns because malicious participants may derive consumers' financial
statuses from on-chain asynchronous payments. Furthermore, the newly created
transactions for deferred payments introduce additional time overheads, which
weaken the scalability of BNPL services. To address these issues, we propose an
efficient and privacy-preserving blockchain-based asynchronous payment scheme
(Epass), which has promising scalability while protecting the privacy of
on-chain consumer transactions. Specifically, Epass leverages locally
verifiable signatures to guarantee the privacy of consumer transactions against
malicious acts. Then, a privacy-preserving asynchronous payment scheme can be
further constructed by leveraging time-release encryption to control trapdoors
of redactable blockchain, reducing time overheads by modifying transactions for
deferred payment. We give formal definitions and security models, generic
structures, and formal proofs for Epass. Extensive comparisons and experimental
analysis show that \textsf{Epass} achieves KB-level communication costs, and
reduces time overhead by more than four times in comparisons with locally
verifiable signatures and Go-Ethereum private test networks.

### 3. [LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge](http://arxiv.org/pdf/2506.09443v1)

Authors: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

Large Language Models (LLMs) have demonstrated remarkable intelligence across
various tasks, which has inspired the development and widespread adoption of
LLM-as-a-Judge systems for automated model testing, such as red teaming and
benchmarking. However, these systems are susceptible to adversarial attacks
that can manipulate evaluation outcomes, raising concerns about their
robustness and, consequently, their trustworthiness. Existing evaluation
methods adopted by LLM-based judges are often piecemeal and lack a unified
framework for comprehensive assessment. Furthermore, prompt template and model
selections for improving judge robustness have been rarely explored, and their
performance in real-world settings remains largely unverified. To address these
gaps, we introduce RobustJudge, a fully automated and scalable framework
designed to systematically evaluate the robustness of LLM-as-a-Judge systems.
RobustJudge investigates the impact of attack methods and defense strategies
(RQ1), explores the influence of prompt template and model selection (RQ2), and
assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our
main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range
of adversarial attacks, including Combined Attack and PAIR, while defense
mechanisms such as Re-tokenization and LLM-based Detectors offer improved
protection; (2) Robustness is highly sensitive to the choice of prompt template
and judge models. Our proposed prompt template optimization method can improve
robustness, and JudgeLM-13B demonstrates strong performance as a robust
open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals
previously unreported vulnerabilities. The source code of RobustJudge is
provided at https://github.com/S3IC-Lab/RobustJudge.

### 4. [The Secure Overview and Analysis OF 3GPP MAC CE](http://arxiv.org/pdf/2506.09502v1)

Authors: Jin Cao, Yuanyuan Yang, Ruhui Ma, Sheng Li, Hui Li

To more effectively control and allocate network resources, MAC CE has been
introduced into the network protocol, which is a type of control signaling
located in the MAC layer. Since MAC CE lacks encryption and integrity
protection mechanisms provided by PDCP, the control signaling carried by MAC CE
is vulnerable to interception or tampering by attackers during resource
scheduling and allocation. Currently, the 3GPP has analyzed the security risks
of Layer 1/Layer 2 Triggered Mobility (LTM), where handover signaling sent to
the UE via MAC CE by the network can lead to privacy leaks and network attacks.
However, in addition to LTM, there may be other potential security
vulnerabilities in other protocol procedures. Therefore, this paper explores
the security threats to MAC CE and the corresponding protection mechanisms. The
research is expected to support the 3GPP's study of MAC CE and be integrated
with the security research of lower-layer protocols, thereby enhancing the
security and reliability of the entire communication system.

### 5. [Beyond Personalization: Federated Recommendation with Calibration via Low-rank Decomposition](http://arxiv.org/pdf/2506.09525v1)

Authors: Jundong Chen, Honglei Zhang, Haoxuan Li, Chunxu Zhang, Zhiwei Li, Yidong Li

Federated recommendation (FR) is a promising paradigm to protect user privacy
in recommender systems. Distinct from general federated scenarios, FR
inherently needs to preserve client-specific parameters, i.e., user embeddings,
for privacy and personalization. However, we empirically find that globally
aggregated item embeddings can induce skew in user embeddings, resulting in
suboptimal performance. To this end, we theoretically analyze the user
embedding skew issue and propose Personalized Federated recommendation with
Calibration via Low-Rank decomposition (PFedCLR). Specifically, PFedCLR
introduces an integrated dual-function mechanism, implemented with a buffer
matrix, to jointly calibrate local user embedding and personalize global item
embeddings. To ensure efficiency, we employ a low-rank decomposition of the
buffer matrix to reduce the model overhead. Furthermore, for privacy, we train
and upload the local model before personalization, preventing the server from
accessing sensitive information. Extensive experiments demonstrate that PFedCLR
effectively mitigates user embedding skew and achieves a desirable trade-off
among performance, efficiency, and privacy, outperforming state-of-the-art
(SOTA) methods.

### 6. [Identity and Access Management for the Computing Continuum](http://arxiv.org/pdf/2506.09559v1)

Authors: Chalima Dimitra Nassar Kyriakidou, Athanasia Maria Papathanasiou, Vasilios A. Siris, Nikos Fotiou, George C. Polyzos, Eduardo Cánovas Martínez, Antonio Skarmeta

The computing continuum introduces new challenges for access control due to
its dynamic, distributed, and heterogeneous nature. In this paper, we propose a
Zero-Trust (ZT) access control solution that leverages decentralized
identification and authentication mechanisms based on Decentralized Identifiers
(DIDs) and Verifiable Credentials (VCs). Additionally, we employ
Relationship-Based Access Control (ReBAC) to define policies that capture the
evolving trust relationships inherent in the continuum. Through a
proof-of-concept implementation, we demonstrate the feasibility and efficiency
of our solution, highlighting its potential to enhance security and trust in
decentralized environments.

### 7. [The Everyday Security of Living with Conflict](http://arxiv.org/pdf/2506.09580v1)

Authors: Jessica McClearn, Reem Talhouk, Rikke Bjerg Jensen

When `cyber' is used as a prefix, attention is typically drawn to the
technological and spectacular aspects of war and conflict -- and, by extension,
security. We offer a different approach to engaging with and understanding
security in such contexts, by foregrounding the everyday -- mundane --
experiences of security within communities living with and fleeing from war. We
do so through three vignettes from our field research in Colombia, Lebanon and
Sweden, respectively, and by highlighting the significance of ethnography for
security research with communities living in regions afflicted by war. We
conclude by setting out a call to action for security researchers and
practitioners to consider such lived experiences in the design of security
technology that aims to cater to the needs of communities in `global conflict
and disaster regions'.

### 8. [On the Virtues of Information Security in the UK Climate Movement](http://arxiv.org/pdf/2506.09719v1)

Authors: Mikaela Brough, Rikke Bjerg Jensen, Martin R. Albrecht

We report on an ethnographic study with members of the climate movement in
the United Kingdom (UK). We conducted participant observation and interviews at
protests and in various activist settings. Reporting on the findings as they
relate to information security, we show that members of the UK climate movement
wrestled with (i) a fundamental tension between openness and secrecy; (ii)
tensions between autonomy and collective interdependence in
information-security decision-making; (iii) conflicting activist ideals that
shape security discourses; and (iv) pressures from different social gazes --
from each other, from people outside the movement and from their adversaries.
Overall, our findings shed light on the social complexities of
information-security research in activist settings and provoke methodological
questions about programmes that aim to design for activists.

### 9. [What is the Cost of Differential Privacy for Deep Learning-Based Trajectory Generation?](http://arxiv.org/pdf/2506.09312v1)

Authors: Erik Buchholz, Natasha Fernandes, David D. Nguyen, Alsharif Abuadbba, Surya Nepal, Salil S. Kanhere

While location trajectories offer valuable insights, they also reveal
sensitive personal information. Differential Privacy (DP) offers formal
protection, but achieving a favourable utility-privacy trade-off remains
challenging. Recent works explore deep learning-based generative models to
produce synthetic trajectories. However, current models lack formal privacy
guarantees and rely on conditional information derived from real data during
generation. This work investigates the utility cost of enforcing DP in such
models, addressing three research questions across two datasets and eleven
utility metrics. (1) We evaluate how DP-SGD, the standard DP training method
for deep learning, affects the utility of state-of-the-art generative models.
(2) Since DP-SGD is limited to unconditional models, we propose a novel DP
mechanism for conditional generation that provides formal guarantees and assess
its impact on utility. (3) We analyse how model types - Diffusion, VAE, and GAN
- affect the utility-privacy trade-off. Our results show that DP-SGD
significantly impacts performance, although some utility remains if the
datasets is sufficiently large. The proposed DP mechanism improves training
stability, particularly when combined with DP-SGD, for unstable models such as
GANs and on smaller datasets. Diffusion models yield the best utility without
guarantees, but with DP-SGD, GANs perform best, indicating that the best
non-private model is not necessarily optimal when targeting formal guarantees.
In conclusion, DP trajectory generation remains a challenging task, and formal
guarantees are currently only feasible with large datasets and in constrained
use cases.

### 10. [DAVSP: Safety Alignment for Large Vision-Language Models via Deep Aligned Visual Safety Prompt](http://arxiv.org/pdf/2506.09353v1)

Authors: Yitong Zhang, Jia Li, Liyi Cai, Ge Li

Large Vision-Language Models (LVLMs) have achieved impressive progress across
various applications but remain vulnerable to malicious queries that exploit
the visual modality. Existing alignment approaches typically fail to resist
malicious queries while preserving utility on benign ones effectively. To
address these challenges, we propose Deep Aligned Visual Safety Prompt (DAVSP),
which is built upon two key innovations. First, we introduce the Visual Safety
Prompt, which appends a trainable padding region around the input image. It
preserves visual features and expands the optimization space. Second, we
propose Deep Alignment, a novel approach to train the visual safety prompt
through supervision in the model's activation space. It enhances the inherent
ability of LVLMs to perceive malicious queries, achieving deeper alignment than
prior works. Extensive experiments across five benchmarks on two representative
LVLMs demonstrate that DAVSP effectively resists malicious queries while
preserving benign input utility. Furthermore, DAVSP exhibits great cross-model
generation ability. Ablation studies further reveal that both the Visual Safety
Prompt and Deep Alignment are essential components, jointly contributing to its
overall effectiveness. The code is publicly available at
https://github.com/zhangyitonggg/DAVSP.

### Computer Vision and Pattern Recognition

### 1. [MSSDF: Modality-Shared Self-supervised Distillation for High-Resolution Multi-modal Remote Sensing Image Learning](http://arxiv.org/pdf/2506.09327v1)

Authors: Tong Wang, Guanzhou Chen, Xiaodong Zhang, Chenxi Liu, Jiaqi Wang, Xiaoliang Tan, Wenchao Guo, Qingyuan Yang, Kaiqi Zhang

Remote sensing image interpretation plays a critical role in environmental
monitoring, urban planning, and disaster assessment. However, acquiring
high-quality labeled data is often costly and time-consuming. To address this
challenge, we proposes a multi-modal self-supervised learning framework that
leverages high-resolution RGB images, multi-spectral data, and digital surface
models (DSM) for pre-training. By designing an information-aware adaptive
masking strategy, cross-modal masking mechanism, and multi-task self-supervised
objectives, the framework effectively captures both the correlations across
different modalities and the unique feature structures within each modality. We
evaluated the proposed method on multiple downstream tasks, covering typical
remote sensing applications such as scene classification, semantic
segmentation, change detection, object detection, and depth estimation.
Experiments are conducted on 15 remote sensing datasets, encompassing 26 tasks.
The results demonstrate that the proposed method outperforms existing
pretraining approaches in most tasks. Specifically, on the Potsdam and
Vaihingen semantic segmentation tasks, our method achieved mIoU scores of
78.30\% and 76.50\%, with only 50\% train-set. For the US3D depth estimation
task, the RMSE error is reduced to 0.182, and for the binary change detection
task in SECOND dataset, our method achieved mIoU scores of 47.51\%, surpassing
the second CS-MAE by 3 percentage points. Our pretrain code, checkpoints, and
HR-Pairs dataset can be found in https://github.com/CVEO/MSSDF.

### 2. [An Effective End-to-End Solution for Multimodal Action Recognition](http://arxiv.org/pdf/2506.09345v1)

Authors: Songping Wang, Xiantao Hu, Yueming Lyu, Caifeng Shan

Recently, multimodal tasks have strongly advanced the field of action
recognition with their rich multimodal information. However, due to the
scarcity of tri-modal data, research on tri-modal action recognition tasks
faces many challenges. To this end, we have proposed a comprehensive multimodal
action recognition solution that effectively utilizes multimodal information.
First, the existing data are transformed and expanded by optimizing data
enhancement techniques to enlarge the training scale. At the same time, more
RGB datasets are used to pre-train the backbone network, which is better
adapted to the new task by means of transfer learning. Secondly, multimodal
spatial features are extracted with the help of 2D CNNs and combined with the
Temporal Shift Module (TSM) to achieve multimodal spatial-temporal feature
extraction comparable to 3D CNNs and improve the computational efficiency. In
addition, common prediction enhancement methods, such as Stochastic Weight
Averaging (SWA), Ensemble and Test-Time augmentation (TTA), are used to
integrate the knowledge of models from different training periods of the same
architecture and different architectures, so as to predict the actions from
different perspectives and fully exploit the target information. Ultimately, we
achieved the Top-1 accuracy of 99% and the Top-5 accuracy of 100% on the
competition leaderboard, demonstrating the superiority of our solution.

### 3. [A new approach for image segmentation based on diffeomorphic registration and gradient fields](http://arxiv.org/pdf/2506.09357v1)

Authors: Junchao Zhou

Image segmentation is a fundamental task in computer vision aimed at
delineating object boundaries within images. Traditional approaches, such as
edge detection and variational methods, have been widely explored, while recent
advances in deep learning have shown promising results but often require
extensive training data. In this work, we propose a novel variational framework
for 2D image segmentation that integrates concepts from shape analysis and
diffeomorphic transformations. Our method models segmentation as the
deformation of a template curve via a diffeomorphic transformation of the image
domain, using the Large Deformation Diffeomorphic Metric Mapping (LDDMM)
framework. The curve evolution is guided by a loss function that compares the
deformed curve to the image gradient field, formulated through the varifold
representation of geometric shapes. The approach is implemented in Python with
GPU acceleration using the PyKeops library. This framework allows for accurate
segmentation with a flexible and theoretically grounded methodology that does
not rely on large datasets.

### 4. [ScaleLSD: Scalable Deep Line Segment Detection Streamlined](http://arxiv.org/pdf/2506.09369v1)

Authors: Zeran Ke, Bin Tan, Xianwei Zheng, Yujun Shen, Tianfu Wu, Nan Xue

This paper studies the problem of Line Segment Detection (LSD) for the
characterization of line geometry in images, with the aim of learning a
domain-agnostic robust LSD model that works well for any natural images. With
the focus of scalable self-supervised learning of LSD, we revisit and
streamline the fundamental designs of (deep and non-deep) LSD approaches to
have a high-performing and efficient LSD learner, dubbed as ScaleLSD, for the
curation of line geometry at scale from over 10M unlabeled real-world images.
Our ScaleLSD works very well to detect much more number of line segments from
any natural images even than the pioneered non-deep LSD approach, having a more
complete and accurate geometric characterization of images using line segments.
Experimentally, our proposed ScaleLSD is comprehensively testified under
zero-shot protocols in detection performance, single-view 3D geometry
estimation, two-view line segment matching, and multiview 3D line mapping, all
with excellent performance obtained. Based on the thorough evaluation, our
ScaleLSD is observed to be the first deep approach that outperforms the
pioneered non-deep LSD in all aspects we have tested, significantly expanding
and reinforcing the versatility of the line geometry of images. Code and Models
are available at https://github.com/ant-research/scalelsd

### 5. [UniForward: Unified 3D Scene and Semantic Field Reconstruction via Feed-Forward Gaussian Splatting from Only Sparse-View Images](http://arxiv.org/pdf/2506.09378v1)

Authors: Qijian Tian, Xin Tan, Jingyu Gong, Yuan Xie, Lizhuang Ma

We propose a feed-forward Gaussian Splatting model that unifies 3D scene and
semantic field reconstruction. Combining 3D scenes with semantic fields
facilitates the perception and understanding of the surrounding environment.
However, key challenges include embedding semantics into 3D representations,
achieving generalizable real-time reconstruction, and ensuring practical
applicability by using only images as input without camera parameters or ground
truth depth. To this end, we propose UniForward, a feed-forward model to
predict 3D Gaussians with anisotropic semantic features from only uncalibrated
and unposed sparse-view images. To enable the unified representation of the 3D
scene and semantic field, we embed semantic features into 3D Gaussians and
predict them through a dual-branch decoupled decoder. During training, we
propose a loss-guided view sampler to sample views from easy to hard,
eliminating the need for ground truth depth or masks required by previous
methods and stabilizing the training process. The whole model can be trained
end-to-end using a photometric loss and a distillation loss that leverages
semantic features from a pre-trained 2D semantic model. At the inference stage,
our UniForward can reconstruct 3D scenes and the corresponding semantic fields
in real time from only sparse-view images. The reconstructed 3D scenes achieve
high-quality rendering, and the reconstructed 3D semantic field enables the
rendering of view-consistent semantic features from arbitrary views, which can
be further decoded into dense segmentation masks in an open-vocabulary manner.
Experiments on novel view synthesis and novel view segmentation demonstrate
that our method achieves state-of-the-art performances for unifying 3D scene
and semantic field reconstruction.

### 6. [ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model](http://arxiv.org/pdf/2506.09385v1)

Authors: Jialong Zuo, Yongtai Deng, Mengdan Tan, Rui Jin, Dongyue Wu, Nong Sang, Liang Pan, Changxin Gao

In real-word scenarios, person re-identification (ReID) expects to identify a
person-of-interest via the descriptive query, regardless of whether the query
is a single modality or a combination of multiple modalities. However, existing
methods and datasets remain constrained to limited modalities, failing to meet
this requirement. Therefore, we investigate a new challenging problem called
Omni Multi-modal Person Re-identification (OM-ReID), which aims to achieve
effective retrieval with varying multi-modal queries. To address dataset
scarcity, we construct ORBench, the first high-quality multi-modal dataset
comprising 1,000 unique identities across five modalities: RGB, infrared, color
pencil, sketch, and textual description. This dataset also has significant
superiority in terms of diversity, such as the painting perspectives and
textual information. It could serve as an ideal platform for follow-up
investigations in OM-ReID. Moreover, we propose ReID5o, a novel multi-modal
learning framework for person ReID. It enables synergistic fusion and
cross-modal alignment of arbitrary modality combinations in a single model,
with a unified encoding and multi-expert routing mechanism proposed. Extensive
experiments verify the advancement and practicality of our ORBench. A wide
range of possible models have been evaluated and compared on it, and our
proposed ReID5o model gives the best performance. The dataset and code will be
made publicly available at https://github.com/Zplusdragon/ReID5o_ORBench.

### 7. [Improving Out-of-Distribution Detection via Dynamic Covariance Calibration](http://arxiv.org/pdf/2506.09399v1)

Authors: Kaiyu Guo, Zijian Wang, Brian C. Lovell, Mahsa Baktashmotlagh

Out-of-Distribution (OOD) detection is essential for the trustworthiness of
AI systems. Methods using prior information (i.e., subspace-based methods) have
shown effective performance by extracting information geometry to detect OOD
data with a more appropriate distance metric. However, these methods fail to
address the geometry distorted by ill-distributed samples, due to the
limitation of statically extracting information geometry from the training
distribution. In this paper, we argue that the influence of ill-distributed
samples can be corrected by dynamically adjusting the prior geometry in
response to new data. Based on this insight, we propose a novel approach that
dynamically updates the prior covariance matrix using real-time input features,
refining its information. Specifically, we reduce the covariance along the
direction of real-time input features and constrain adjustments to the residual
space, thus preserving essential data characteristics and avoiding effects on
unintended directions in the principal space. We evaluate our method on two
pre-trained models for the CIFAR dataset and five pre-trained models for
ImageNet-1k, including the self-supervised DINO model. Extensive experiments
demonstrate that our approach significantly enhances OOD detection across
various models. The code is released at https://github.com/workerbcd/ooddcc.

### 8. [SRPL-SFDA: SAM-Guided Reliable Pseudo-Labels for Source-Free Domain Adaptation in Medical Image Segmentation](http://arxiv.org/pdf/2506.09403v1)

Authors: Xinya Liu, Jianghao Wu, Tao Lu, Shaoting Zhang, Guotai Wang

Domain Adaptation (DA) is crucial for robust deployment of medical image
segmentation models when applied to new clinical centers with significant
domain shifts. Source-Free Domain Adaptation (SFDA) is appealing as it can deal
with privacy concerns and access constraints on source-domain data during
adaptation to target-domain data. However, SFDA faces challenges such as
insufficient supervision in the target domain with unlabeled images. In this
work, we propose a Segment Anything Model (SAM)-guided Reliable Pseudo-Labels
method for SFDA (SRPL-SFDA) with three key components: 1) Test-Time Tri-branch
Intensity Enhancement (T3IE) that not only improves quality of raw
pseudo-labels in the target domain, but also leads to SAM-compatible inputs
with three channels to better leverage SAM's zero-shot inference ability for
refining the pseudo-labels; 2) A reliable pseudo-label selection module that
rejects low-quality pseudo-labels based on Consistency of Multiple SAM Outputs
(CMSO) under input perturbations with T3IE; and 3) A reliability-aware training
procedure in the unlabeled target domain where reliable pseudo-labels are used
for supervision and unreliable parts are regularized by entropy minimization.
Experiments conducted on two multi-domain medical image segmentation datasets
for fetal brain and the prostate respectively demonstrate that: 1) SRPL-SFDA
effectively enhances pseudo-label quality in the unlabeled target domain, and
improves SFDA performance by leveraging the reliability-aware training; 2)
SRPL-SFDA outperformed state-of-the-art SFDA methods, and its performance is
close to that of supervised training in the target domain. The code of this
work is available online: https://github.com/HiLab-git/SRPL-SFDA.

### 9. [Noise Conditional Variational Score Distillation](http://arxiv.org/pdf/2506.09416v1)

Authors: Xinyu Peng, Ziyang Zheng, Yaoming Wang, Han Li, Nuowen Kan, Wenrui Dai, Chenglin Li, Junni Zou, Hongkai Xiong

We propose Noise Conditional Variational Score Distillation (NCVSD), a novel
method for distilling pretrained diffusion models into generative denoisers. We
achieve this by revealing that the unconditional score function implicitly
characterizes the score function of denoising posterior distributions. By
integrating this insight into the Variational Score Distillation (VSD)
framework, we enable scalable learning of generative denoisers capable of
approximating samples from the denoising posterior distribution across a wide
range of noise levels. The proposed generative denoisers exhibit desirable
properties that allow fast generation while preserve the benefit of iterative
refinement: (1) fast one-step generation through sampling from pure Gaussian
noise at high noise levels; (2) improved sample quality by scaling the
test-time compute with multi-step sampling; and (3) zero-shot probabilistic
inference for flexible and controllable sampling. We evaluate NCVSD through
extensive experiments, including class-conditional image generation and inverse
problem solving. By scaling the test-time compute, our method outperforms
teacher diffusion models and is on par with consistency models of larger sizes.
Additionally, with significantly fewer NFEs than diffusion-based methods, we
achieve record-breaking LPIPS on inverse problems.

### 10. [ODG: Occupancy Prediction Using Dual Gaussians](http://arxiv.org/pdf/2506.09417v1)

Authors: Yunxiao Shi, Yinhao Zhu, Shizhong Han, Jisoo Jeong, Amin Ansari, Hong Cai, Fatih Porikli

3D occupancy provides fine-grained 3D geometry and semantics for scene
understanding which is critical for autonomous driving. Most existing methods,
however, carry high compute costs, requiring dense 3D feature volume and
cross-attention to effectively aggregate information. More recent works have
adopted Bird's Eye View (BEV) or sparse points as scene representation with
much reduced cost, but still suffer from their respective shortcomings. More
concretely, BEV struggles with small objects that often experience significant
information loss after being projected to the ground plane. On the other hand,
points can flexibly model little objects in 3D, but is inefficient at capturing
flat surfaces or large objects. To address these challenges, in this paper, we
present a novel 3D occupancy prediction approach, ODG, which combines BEV and
sparse points based representations. We propose a dual-branch design: a
query-based sparse points branch and a BEV branch. The 3D information learned
in the sparse points branch is shared with the BEV stream via cross-attention,
which enriches the weakened signals of difficult objects on the BEV plane. The
outputs of both branches are finally fused to generate predicted 3D occupancy.
We conduct extensive experiments on the Occ3D-nuScenes and Occ3D-Waymo
benchmarks that demonstrate the superiority of our proposed ODG. Moreover, ODG
also delivers competitive inference speed when compared to the latest efficient
approaches.

### Computers and Society

### 1. [Situated Bayes -- Feminist and Pluriversal Perspectives on Bayesian Knowledge](http://arxiv.org/pdf/2506.09472v1)

Authors: Juni Schindler, Goda Klumbytė, Matthew Fuller

This is the introduction and lead article to the Situated Bayes special issue
of Computational Culture. The article introduces Bayes' Theorem and aspects of
its contemporary uses, for instance in machine learning. A mathematical
discussion is developed alongside a consideration of Bayes Theorem in relation
to critical theories of knowledge, specifically the discussion of situated
knowledge in feminist theories of science, pluriversal knowledge in decolonial
theory, and critical approaches to mathematics. We discuss whether there are
possible resonances between Bayesian mapping of multiple functions and the idea
of the subjective on the one hand and these theoretical propositions on the
other and propose further lines of enquiry for future research. In closing the
introduction, the contributions to the special issue are briefly described.

### 2. [Ties of Trust: a bowtie model to uncover trustor-trustee relationships in LLMs](http://arxiv.org/pdf/2506.09632v1)

Authors: Eva Paraschou, Maria Michali, Sofia Yfantidou, Stelios Karamanidis, Stefanos Rafail Kalogeros, Athena Vakali

The rapid and unprecedented dominance of Artificial Intelligence (AI),
particularly through Large Language Models (LLMs), has raised critical trust
challenges in high-stakes domains like politics. Biased LLMs' decisions and
misinformation undermine democratic processes, and existing trust models fail
to address the intricacies of trust in LLMs. Currently, oversimplified,
one-directional approaches have largely overlooked the many relationships
between trustor (user) contextual factors (e.g. ideology, perceptions) and
trustee (LLMs) systemic elements (e.g. scientists, tool's features). In this
work, we introduce a bowtie model for holistically conceptualizing and
formulating trust in LLMs, with a core component comprehensively exploring
trust by tying its two sides, namely the trustor and the trustee, as well as
their intricate relationships. We uncover these relationships within the
proposed bowtie model and beyond to its sociotechnical ecosystem, through a
mixed-methods explanatory study, that exploits a political discourse analysis
tool (integrating ChatGPT), by exploring and responding to the next critical
questions: 1) How do trustor's contextual factors influence trust-related
actions? 2) How do these factors influence and interact with trustee systemic
elements? 3) How does trust itself vary across trustee systemic elements? Our
bowtie-based explanatory analysis reveals that past experiences and familiarity
significantly shape trustor's trust-related actions; not all trustor contextual
factors equally influence trustee systemic elements; and trustee's
human-in-the-loop features enhance trust, while lack of transparency decreases
it. Finally, this solid evidence is exploited to deliver recommendations,
insights and pathways towards building robust trusting ecosystems in LLM-based
solutions.

### 3. [The Path is the Goal: a Study on the Nature and Effects of Shortest-Path Stability Under Perturbation of Destination](http://arxiv.org/pdf/2506.09731v1)

Authors: Giuliano Cornacchia, Mirco Nanni

This work examines the phenomenon of path variability in urban navigation,
where small changes in destination might lead to significantly different
suggested routes. Starting from an observation of this variability over the
city of Barcelona, we explore whether this is a localized or widespread
occurrence and identify factors influencing path variability. We introduce the
concept of "path stability", a measure of how robust a suggested route is to
minor destination adjustments, define a detailed experimentation process and
apply it across multiple cities worldwide. Our analysis shows that path
stability is shaped by city-specific factors and trip characteristics, also
identifying some common patterns. Results reveal significant heterogeneity in
path stability across cities, allowing for categorization into "stable" and
"unstable" cities. These findings offer new insights for urban planning and
traffic management, highlighting opportunities for optimizing navigation
systems to enhance route consistency and urban mobility.

### 4. [TikTok's Research API: Problems Without Explanations](http://arxiv.org/pdf/2506.09746v1)

Authors: Carlos Entrena-Serrano, Martin Degeling, Salvatore Romano, Raziye Buse Çetin

Following the Digital Services Act of 2023, which requires Very Large Online
Platforms (VLOPs) and Very Large Online Search Engines (VLOSEs) to facilitate
data accessibility for independent research, TikTok augmented its Research API
access within Europe in July 2023. This action was intended to ensure
compliance with the DSA, bolster transparency, and address systemic risks.
Nonetheless, research findings reveal that despite this expansion, notable
limitations and inconsistencies persist within the data provided. Our
experiment reveals that the API fails to provide metadata for one in eight
videos provided through data donations, including official TikTok videos,
advertisements, videos from China, and content from specific accounts, without
an apparent reason. The API data is incomplete, making it unreliable when
working with data donations, a prominent methodology for algorithm audits and
research on platform accountability. To monitor the functionality of the API
and eventual fixes implemented by TikTok, we publish a dashboard with a daily
check of the availability of 10 videos that were not retrievable in the last
month. The video list includes very well-known accounts, notably that of Taylor
Swift. The current API lacks the necessary capabilities for thorough
independent research and scrutiny. It is crucial to support and safeguard
researchers who utilize data scraping to independently validate the platform's
data quality.

### 5. [Where Journalism Silenced Voices: Exploring Discrimination in the Representation of Indigenous Communities in Bangladesh](http://arxiv.org/pdf/2506.09771v1)

Authors: Abhijit Paul, Adity Khisa, Zarif Masud, Sharif Md. Abdullah, Ahmedul Kabir, Shebuti Rayana

In this paper, we examine the intersections of indigeneity and media
representation in shaping perceptions of indigenous communities in Bangladesh.
Using a mixed-methods approach, we combine quantitative analysis of media data
with qualitative insights from focus group discussions (FGD). First, we
identify a total of 4,893 indigenous-related articles from our initial dataset
of 2.2 million newspaper articles, using a combination of keyword-based
filtering and LLM, achieving 77% accuracy and an F1-score of 81.9\%. From
manually inspecting 3 prominent Bangla newspapers, we identify 15 genres that
we use as our topics for semi-supervised topic modeling using CorEx. Results
show indigenous news articles have higher representation of culture and
entertainment (19%, 10% higher than general news articles), and a
disproportionate focus on conflict and protest (9%, 7% higher than general
news). On the other hand, sentiment analysis reveals that 57% of articles on
indigenous topics carry a negative tone, compared to 27% for non-indigenous
related news. Drawing from communication studies, we further analyze framing,
priming, and agenda-setting (frequency of themes) to support the case for
discrimination in representation of indigenous news coverage. For the
qualitative part of our analysis, we facilitated FGD, where participants
further validated these findings. Participants unanimously expressed their
feeling of being under-represented, and that critical issues affecting their
communities (such as education, healthcare, and land rights) are systematically
marginalized in news media coverage. By highlighting 8 cases of discrimination
and media misrepresentation that were frequently mentioned by participants in
the FGD, this study emphasizes the urgent need for more equitable media
practices that accurately reflect the experiences and struggles of marginalized
communities.

### 6. [Calculating Software's Energy Use and Carbon Emissions: A Survey of the State of Art, Challenges, and the Way Ahead](http://arxiv.org/pdf/2506.09683v1)

Authors: Priyavanshi Pathania, Nikhil Bamby, Rohit Mehra, Samarth Sikand, Vibhu Saujanya Sharma, Vikrant Kaulgud, Sanjay Podder, Adam P. Burden

The proliferation of software and AI comes with a hidden risk: its growing
energy and carbon footprint. As concerns regarding environmental sustainability
come to the forefront, understanding and optimizing how software impacts the
environment becomes paramount. In this paper, we present a state-of-the-art
review of methods and tools that enable the measurement of software and
AI-related energy and/or carbon emissions. We introduce a taxonomy to
categorize the existing work as Monitoring, Estimation, or Black-Box
approaches. We delve deeper into the tools and compare them across different
dimensions and granularity - for example, whether their measurement encompasses
energy and carbon emissions and the components considered (like CPU, GPU, RAM,
etc.). We present our observations on the practical use (component wise
consolidation of approaches) as well as the challenges that we have identified
across the current state-of-the-art. As we start an initiative to address these
challenges, we emphasize active collaboration across the community in this
important field.

### 7. [Delegations as Adaptive Representation Patterns: Rethinking Influence in Liquid Democracy](http://arxiv.org/pdf/2506.09789v1)

Authors: Davide Grossi, Andreas Nitsche

Liquid democracy is a mechanism for the division of labor in decision-making
through the transitive delegation of influence. In essence, all individuals
possess the autonomy to determine the issues with which they will engage
directly, while for other matters, they may appoint a representative of their
choosing. So far, the literature has studied the delegation structures emerging
in liquid democracy as static. As a result, transitivity defined as the
capacity to transfer acquired authority to another entity, has been identified
as a concern as it would be conducive to unrestrained accumulation of power.
  Focusing on the implementation of liquid democracy supported by the
LiquidFeedback software, we propose a novel approach to assessing the influence
of voting nodes in a transitive delegation graph, taking into account the
process nature of real-world liquid democracy in which delegation and voting
are distinct and increasingly independent activities. By introducing a novel
model of delegations in liquid democracy, we show how transitivity may in fact
contribute to an effective regulation of deliberation influence and
decision-making power. While maintaining the one-person, one-vote paradigm for
all votes cast, the anticipated influence of an agent, to the extent it is
stemming from transitivity, experiences a precipitous decline following an
exponential trajectory.
  In general, it is our objective to move the first steps towards a rigorous
analysis of liquid democracy as an adaptive democratic representation process.
The adaptivity aspect of liquid democracy has not yet been explored within the
existing academic literature despite it being, we believe, one of its most
important features. We therefore also outline a research agenda focusing on
this aspect of liquid democracy.

### 8. [Assessing a Safety Case: Bottom-up Guidance for Claims and Evidence Evaluation](http://arxiv.org/pdf/2506.09929v1)

Authors: Scott Schnelle, Francesca Favaro, Laura Fraade-Blanar, David Wichner, Holland Broce, Justin Miranda

As Automated Driving Systems (ADS) technology advances, ensuring safety and
public trust requires robust assurance frameworks, with safety cases emerging
as a critical tool toward such a goal. This paper explores an approach to
assess how a safety case is supported by its claims and evidence, toward
establishing credibility for the overall case. Starting from a description of
the building blocks of a safety case (claims, evidence, and optional
format-dependent entries), this paper delves into the assessment of support of
each claim through the provided evidence. Two domains of assessment are
outlined for each claim: procedural support (formalizing process specification)
and implementation support (demonstrating process application). Additionally,
an assessment of evidence status is also undertaken, independently from the
claims support. Scoring strategies and evaluation guidelines are provided,
including detailed scoring tables for claim support and evidence status
assessment. The paper further discusses governance, continual improvement, and
timing considerations for safety case assessments. Reporting of results and
findings is contextualized within its primary use for internal decision-making
on continual improvement efforts. The presented approach builds on state of the
art auditing practices, but specifically tackles the question of judging the
credibility of a safety case. While not conclusive on its own, it provides a
starting point toward a comprehensive "Case Credibility Assessment" (CCA),
starting from the evaluation of the support for each claim (individually and in
aggregate), as well as every piece of evidence provided. By delving into the
technical intricacies of ADS safety cases, this work contributes to the ongoing
discourse on safety assurance and aims to facilitate the responsible
integration of ADS technology into society.

### 9. [KI4Demokratie: An AI-Based Platform for Monitoring and Fostering Democratic Discourse](http://arxiv.org/pdf/2506.09947v1)

Authors: Rudy Alexandro Garrido Veliz, Till Nikolaus Schaland, Simon Bergmoser, Florian Horwege, Somya Bansal, Ritesh Nahar, Martin Semmann, Jörg Forthmann, Seid Muhie Yimam

Social media increasingly fuel extremism, especially right-wing extremism,
and enable the rapid spread of antidemocratic narratives. Although AI and data
science are often leveraged to manipulate political opinion, there is a
critical need for tools that support effective monitoring without infringing on
freedom of expression. We present KI4Demokratie, an AI-based platform that
assists journalists, researchers, and policymakers in monitoring right-wing
discourse that may undermine democratic values. KI4Demokratie applies machine
learning models to a large-scale German online data gathered on a daily basis,
providing a comprehensive view of trends in the German digital sphere. Early
analysis reveals both the complexity of tracking organized extremist behavior
and the promise of our integrated approach, especially during key events.

### 10. [From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring](http://arxiv.org/pdf/2506.09996v1)

Authors: Yang Li, Qiang Sheng, Yehan Yang, Xueyao Zhang, Juan Cao

Though safety alignment has been applied to most large language models
(LLMs), LLM service providers generally deploy a subsequent moderation as the
external safety guardrail in real-world products. Existing moderators mainly
practice a conventional full detection, which determines the harmfulness based
on the complete LLM output, causing high service latency. Recent works pay more
attention to partial detection where moderators oversee the generation midway
and early stop the output if harmfulness is detected, but they directly apply
moderators trained with the full detection paradigm to incomplete outputs,
introducing a training-inference gap that lowers the performance. In this
paper, we explore how to form a data-and-model solution that natively supports
partial detection. For the data, we construct FineHarm, a dataset consisting of
29K prompt-response pairs with fine-grained annotations to provide reasonable
supervision for token-level training. Then, we propose the streaming content
monitor, which is trained with dual supervision of response- and token-level
labels and can follow the output stream of LLM to make a timely judgment of
harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is
comparable to full detection, by only seeing the first 18% of tokens in
responses on average. Moreover, the SCM can serve as a pseudo-harmfulness
annotator for improving safety alignment and lead to a higher harmlessness
score than DPO.

### Databases

### 1. [ArcNeural: A Multi-Modal Database for the Gen-AI Era](http://arxiv.org/pdf/2506.09467v1)

Authors: Wu Min, Qiao Yuncong, Yu Tan, Chenghu Yang

ArcNeural introduces a novel multimodal database tailored for the demands of
Generative AI and Large Language Models, enabling efficient management of
diverse data types such as graphs, vectors, and documents. Its storage-compute
separated architecture integrates graph technology, advanced vector indexing,
and transaction processing to support real-time analytics and AI-driven
applications. Key features include a unified storage layer, adaptive edge
collection in MemEngine, and seamless integration of transaction and analytical
processing. Experimental evaluations demonstrate ArcNeural's superior
performance and scalability compared to state-of-the-art systems. This system
bridges structured and unstructured data management, offering a versatile
solution for enterprise-grade AI applications.
  ArcNeural's design addresses the challenges of multimodal data processing,
providing a robust framework for intelligent, data-driven solutions in the Gen
AI era.

### 2. [Linking Data Citation to Repository Visibility: An Empirical Study](http://arxiv.org/pdf/2506.09530v1)

Authors: Fakhri Momeni, Janete Saldanha Bach, Brigitte Mathiak, Peter Mutschke

In today's data-driven research landscape, dataset visibility and
accessibility play a crucial role in advancing scientific knowledge. At the
same time, data citation is essential for maintaining academic integrity,
acknowledging contributions, validating research outcomes, and fostering
scientific reproducibility. As a critical link, it connects scholarly
publications with the datasets that drive scientific progress. This study
investigates whether repository visibility influences data citation rates. We
hypothesize that repositories with higher visibility, as measured by search
engine metrics, are associated with increased dataset citations. Using OpenAlex
data and repository impact indicators (including the visibility index from
Sistrix, the h-index of repositories, and citation metrics such as mean and
median citations), we analyze datasets in Social Sciences and Economics to
explore their relationship. Our findings suggest that datasets hosted on more
visible web domains tend to receive more citations, with a positive correlation
observed between web domain visibility and dataset citation counts,
particularly for datasets with at least one citation. However, when analyzing
domain-level citation metrics, such as the h-index, mean, and median citations,
the correlations are inconsistent and weaker. While higher visibility domains
tend to host datasets with greater citation impact, the distribution of
citations across datasets varies significantly. These results suggest that
while visibility plays a role in increasing citation counts, it is not the sole
factor influencing dataset citation impact. Other elements, such as dataset
quality, research trends, and disciplinary norms, also contribute significantly
to citation patterns.

### 3. [Microservices and Real-Time Processing in Retail IT: A Review of Open-Source Toolchains and Deployment Strategies](http://arxiv.org/pdf/2506.09938v1)

Authors: Aaditaa Vashisht, Rekha B S

With the rapid pace of digital transformation, the retail industry is
increasingly depending on real-time, scalable, and resilient systems to manage
financial transactions, analyze customer behavior, and streamline order
processing. This literature review explores how modern event-driven and
microservices-based architectures, particularly those leveraging Apache Kafka,
Spring Boot, MongoDB, and Kubernetes are transforming retail and financial
systems. By systematically reviewing academic publications, technical white
papers, and industry reports from recent years, this study synthesizes key
themes and implementation strategies. The analysis reveals that technologies
like Kafka and Spring Boot are instrumental in building low-latency,
event-driven applications that support real-time analytics and fraud detection,
while MongoDB, when deployed on Kubernetes, ensures fault tolerance and high
availability in inventory and transaction systems. Kubernetes itself plays a
crucial role in automating deployment and scaling of microservices. These
findings provide valuable insights for industry practitioners aiming to design
scalable infrastructures, identify research opportunities in hybrid deployment
models, and offer educators a foundation to integrate modern system
architectures into professional and technical communication training.

### Distributed, Parallel, and Cluster Computing

### 1. [Efficient Task Graph Scheduling for Parallel QR Factorization in SLSQP](http://arxiv.org/pdf/2506.09463v1)

Authors: Soumyajit Chatterjee, Rahul Utkoor, Uppu Eshwar, Sathya Peri, V. Krishna Nandivada

Efficient task scheduling is paramount in parallel programming on multi-core
architectures, where tasks are fundamental computational units. QR
factorization is a critical sub-routine in Sequential Least Squares Quadratic
Programming (SLSQP) for solving non-linear programming (NLP) problems. QR
factorization decomposes a matrix into an orthogonal matrix Q and an upper
triangular matrix R, which are essential for solving systems of linear
equations arising from optimization problems. SLSQP uses an in-place version of
QR factorization, which requires storing intermediate results for the next
steps of the algorithm. Although DAG-based approaches for QR factorization are
prevalent in the literature, they often lack control over the intermediate
kernel results, providing only the final output matrices Q and R. This
limitation is particularly challenging in SLSQP, where intermediate results of
QR factorization are crucial for back-substitution logic at each iteration. Our
work introduces novel scheduling techniques using a two-queue approach to
execute the QR factorization kernel effectively. This approach, implemented in
high-level C++ programming language, facilitates compiler optimizations and
allows storing intermediate results required by back-substitution logic.
Empirical evaluations demonstrate substantial performance gains, including a
10x improvement over the sequential QR version of the SLSQP algorithm.

### 2. [Understanding the Performance and Power of LLM Inferencing on Edge Accelerators](http://arxiv.org/pdf/2506.09554v1)

Authors: Mayank Arya, Yogesh Simmhan

Large Language Models (LLMs) have demonstrated exceptional benefits to a wide
range of domains, for tasks as diverse as code generation and robot navigation.
While LLMs are usually served from cloud data centers, mission-critical and
privacy-sensitive applications may require local hosting of open LLM models.
Given the large GPU memory footprint needed for LLMs, edge accelerators such as
Nvidia Jetson Orin AGX with 64GB of shared GPU-CPU RAM are a compelling choice.
However, the feasibility and performance of LLM inference on edge accelerators
is under-explored. This study presents a detailed evaluation of LLM inference
on the NVIDIA Jetson Orin AGX, on four SOTA models ranging from 2.7B to 32.8B
parameters, such as Meta Llama3.1, Microsoft-Phi2, Deepseek-R1-Qwen.We
investigate the impact of varying batch sizes, sequence lengths, and
quantization levels on latency, throughput, and perplexity, and also explore
various custom power modes on the Orin AGX to perform power and energy
consumption analysis. Our findings offer interesting insights on the trade-offs
between efficiency, inference speed and resource use, e.g., increasing the
sequence length causes a decrease in token throughput and quantization causes
smaller LLMs to be slower. These results can help optimize LLM serving on edge
accelerators for practical applications.

### 3. [Frosty for partial synchrony](http://arxiv.org/pdf/2506.09823v1)

Authors: Stephen Buttolph, Andrew Lewis-Pye, Kevin Sekniqi

Snowman is the consensus protocol used by blockchains on Avalanche. Recent
work has shown both how to augment Snowman with a `liveness' module called
`Frosty' that protects against liveness attacks, and also how to modify Snowman
so as to be consistent in partial synchrony. Since Frosty assumes (a strong
form of) synchrony, the aim of this note is to show how to modify Frosty to
deal with the partially synchronous version of Snowman.

### 4. [Generalization Error Analysis for Attack-Free and Byzantine-Resilient Decentralized Learning with Data Heterogeneity](http://arxiv.org/pdf/2506.09438v1)

Authors: Haoxiang Ye, Tao Sun, Qing Ling

Decentralized learning, which facilitates joint model training across
geographically scattered agents, has gained significant attention in the field
of signal and information processing in recent years. While the optimization
errors of decentralized learning algorithms have been extensively studied,
their generalization errors remain relatively under-explored. As the
generalization errors reflect the scalability of trained models on unseen data
and are crucial in determining the performance of trained models in real-world
applications, understanding the generalization errors of decentralized learning
is of paramount importance. In this paper, we present fine-grained
generalization error analysis for both attack-free and Byzantine-resilient
decentralized learning with heterogeneous data as well as under mild
assumptions, in contrast to prior studies that consider homogeneous data and/or
rely on a stringent bounded stochastic gradient assumption. Our results shed
light on the impact of data heterogeneity, model initialization and stochastic
gradient noise -- factors that have not been closely investigated before -- on
the generalization error of decentralized learning. We also reveal that
Byzantine attacks performed by malicious agents largely affect the
generalization error, and their negative impact is inherently linked to the
data heterogeneity while remaining independent on the sample size. Numerical
experiments on both convex and non-convex tasks are conducted to validate our
theoretical findings.

### 5. [SyncFed: Time-Aware Federated Learning through Explicit Timestamping and Synchronization](http://arxiv.org/pdf/2506.09660v1)

Authors: Baran Can Gül, Stefanos Tziampazis, Nasser Jazdi, Michael Weyrich

As Federated Learning (FL) expands to larger and more distributed
environments, consistency in training is challenged by network-induced delays,
clock unsynchronicity, and variability in client updates. This combination of
factors may contribute to misaligned contributions that undermine model
reliability and convergence. Existing methods like staleness-aware aggregation
and model versioning address lagging updates heuristically, yet lack mechanisms
to quantify staleness, especially in latency-sensitive and cross-regional
deployments. In light of these considerations, we introduce \emph{SyncFed}, a
time-aware FL framework that employs explicit synchronization and timestamping
to establish a common temporal reference across the system. Staleness is
quantified numerically based on exchanged timestamps under the Network Time
Protocol (NTP), enabling the server to reason about the relative freshness of
client updates and apply temporally informed weighting during aggregation. Our
empirical evaluation on a geographically distributed testbed shows that, under
\emph{SyncFed}, the global model evolves within a stable temporal context,
resulting in improved accuracy and information freshness compared to
round-based baselines devoid of temporal semantics.

### 6. [SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving](http://arxiv.org/pdf/2506.09397v1)

Authors: Xiangchen Li, Dimitrios Spatharakis, Saeid Ghafouri, Jiakun Fan, Dimitrios Nikolopoulos

Regardless the advancements in device capabilities, efficient inferencing
advanced large language models (LLMs) at the edge remains challenging due to
limited device memory and power constraints. Existing strategies, such as
aggressive quantization, pruning, or remote inference, trade accuracy for
efficiency or lead to substantial cost burdens. This position paper introduces
a new approach that leverages speculative decoding, previously viewed primarily
as a decoding acceleration technique for autoregressive generation of LLMs, as
a promising approach specifically adapted for edge computing by orchestrating
computation across heterogeneous devices. We propose SLED, a method that allows
lightweight edge devices to draft multiple candidate tokens locally using
diverse draft models, while a single, shared edge server efficiently batches
and verifies the tokens utilizing a more precise target model. This approach
supports device heterogeneity and reduces server-side memory footprint by
avoiding the need to deploy multiple target models. Our initial experiments
with Jetson Orin Nano, Raspberry Pi 5, and an RTX 6000 edge server indicate
substantial benefits: significantly reduced latency, improved energy
efficiency, and increased concurrent inference sessions, all without
sacrificing model accuracy.

### 7. [On the Performance of Cloud-based ARM SVE for Zero-Knowledge Proving Systems](http://arxiv.org/pdf/2506.09505v1)

Authors: Dumitrel Loghin, Shuang Liang, Shengwei Liu, Xiong Liu, Pingcheng Ruan, Zhigang Ye

Zero-knowledge proofs (ZKP) are becoming a gold standard in scaling
blockchains and bringing Web3 to life. At the same time, ZKP for transactions
running on the Ethereum Virtual Machine require powerful servers with hundreds
of CPU cores. The current zkProver implementation from Polygon is optimized for
x86-64 CPUs by vectorizing key operations, such as Merkle tree building with
Poseidon hashes over the Goldilocks field, with Advanced Vector Extensions (AVX
and AVX512). With these optimizations, a ZKP for a batch of transactions is
generated in less than two minutes. With the advent of cloud servers with ARM
which are at least 10% cheaper than x86-64 servers and the implementation of
ARM Scalable Vector Extension (SVE), we wonder if ARM servers can take over
their x86-64 counterparts. Unfortunately, our analysis shows that current ARM
CPUs are not a match for their x86-64 competitors. Graviton4 from Amazon Web
Services (AWS) and Axion from Google Cloud Platform (GCP) are 1.6X and 1.4X
slower compared to the latest AMD EPYC and Intel Xeon servers from AWS with AVX
and AVX512, respectively, when building a Merkle tree with over four million
leaves. This low performance is due to (1) smaller vector size in these ARM
CPUs (128 bits versus 512 bits in AVX512) and (2) lower clock frequency. On the
other hand, ARM SVE/SVE2 Instruction Set Architecture (ISA) is at least as
powerful as AVX/AVX512 but more flexible. Moreover, we estimate that increasing
the vector size to 512 bits will enable higher performance in ARM CPUs compared
to their x86-64 counterparts while maintaining their price advantage.

### 8. [Private Aggregation for Byzantine-Resilient Heterogeneous Federated Learning](http://arxiv.org/pdf/2506.09870v1)

Authors: Maximilian Egger, Rawad Bitar

Ensuring resilience to Byzantine clients while maintaining the privacy of the
clients' data is a fundamental challenge in federated learning (FL). When the
clients' data is homogeneous, suitable countermeasures were studied from an
information-theoretic perspective utilizing secure aggregation techniques while
ensuring robust aggregation of the clients' gradients. However, the
countermeasures used fail when the clients' data is heterogeneous. Suitable
pre-processing techniques, such as nearest neighbor mixing, were recently shown
to enhance the performance of those countermeasures in the heterogeneous
setting. Nevertheless, those pre-processing techniques cannot be applied with
the introduced privacy-preserving mechanisms.
  We propose a multi-stage method encompassing a careful co-design of
verifiable secret sharing, secure aggregation, and a tailored symmetric private
information retrieval scheme to achieve information-theoretic privacy
guarantees and Byzantine resilience under data heterogeneity. We evaluate the
effectiveness of our scheme on a variety of attacks and show how it outperforms
the previously known techniques. Since the communication overhead of secure
aggregation is non-negligible, we investigate the interplay with zero-order
estimation methods that reduce the communication cost in state-of-the-art FL
tasks and thereby make private aggregation scalable.

### Digital Libraries

### 1. [Linking Data Citation to Repository Visibility: An Empirical Study](http://arxiv.org/pdf/2506.09530v1)

Authors: Fakhri Momeni, Janete Saldanha Bach, Brigitte Mathiak, Peter Mutschke

In today's data-driven research landscape, dataset visibility and
accessibility play a crucial role in advancing scientific knowledge. At the
same time, data citation is essential for maintaining academic integrity,
acknowledging contributions, validating research outcomes, and fostering
scientific reproducibility. As a critical link, it connects scholarly
publications with the datasets that drive scientific progress. This study
investigates whether repository visibility influences data citation rates. We
hypothesize that repositories with higher visibility, as measured by search
engine metrics, are associated with increased dataset citations. Using OpenAlex
data and repository impact indicators (including the visibility index from
Sistrix, the h-index of repositories, and citation metrics such as mean and
median citations), we analyze datasets in Social Sciences and Economics to
explore their relationship. Our findings suggest that datasets hosted on more
visible web domains tend to receive more citations, with a positive correlation
observed between web domain visibility and dataset citation counts,
particularly for datasets with at least one citation. However, when analyzing
domain-level citation metrics, such as the h-index, mean, and median citations,
the correlations are inconsistent and weaker. While higher visibility domains
tend to host datasets with greater citation impact, the distribution of
citations across datasets varies significantly. These results suggest that
while visibility plays a role in increasing citation counts, it is not the sole
factor influencing dataset citation impact. Other elements, such as dataset
quality, research trends, and disciplinary norms, also contribute significantly
to citation patterns.

### Discrete Mathematics

### 1. [Immersions of large cliques in graphs with independence number 2 and bounded maximum degree](http://arxiv.org/pdf/2506.09768v1)

Authors: Fábio Botler, Cristina G. Fernandes, Carla N. Lintzmayer, Rui A. Lopes, Suchismita Mishra, Bruno L. Netto, Maycon Sambinelli

An immersion of a graph $H$ in a graph $G$ is a minimal subgraph $I$ of $G$
for which there is an injection ${{\rm i}} \colon V(H) \to V(I)$ and a set of
edge-disjoint paths $\{P_e: e \in E(H)\}$ in $I$ such that the end vertices of
$P_{uv}$ are precisely ${{\rm i}}(u)$ and ${{\rm i}}(v)$. The immersion
analogue of Hadwiger Conjecture (1943), posed by Lescure and Meyniel (1985),
asks whether every graph $G$ contains an immersion of $K_{\chi(G)}$. Its
restriction to graphs with independence number 2 has received some attention
recently, and Vergara (2017) raised the weaker conjecture that every graph with
independence number 2 has an immersion of $K_{\chi(G)}$. This implies that
every graph with independence number 2 has an immersion of $K_{\lceil n/2
\rceil}$. In this paper, we verify Vergara Conjecture for graphs with bounded
maximum degree. Specifically, we prove that if $G$ is a graph with independence
number $2$, maximum degree less than $2n/3 - 1$ and clique covering number at
most $3$, then $G$ contains an immersion of $K_{\chi(G)}$ (and thus of
$K_{\lceil n/2 \rceil}$). Using a result of Jin (1995), this implies that if
$G$ is a graph with independence number $2$ and maximum degree less than
$19n/29 - 1$, then $G$ contains an immersion of $K_{\chi(G)}$ (and thus of
$K_{\lceil n/2 \rceil}$).

### 2. [A Branch-and-Cut Algorithm for the Optimal Design of Parking Lots with One-way and Two-way Lanes](http://arxiv.org/pdf/2506.09961v1)

Authors: Helen Thomas, Tarun Rambha

We address the problem of maximizing the number of stalls in parking lots
where vehicles park perpendicular to the driveways. Building on recent
research, we first formulate a mixed integer program to maximize the number of
parking stalls using a flow-based approach. Parking lots are rasterized into a
grid, and the proposed MIP model optimizes them in a generic manner, adapting
to the grid resolution and stall size without the need for custom formulations.
The constraints ensure the connectivity of parking stalls and driveways to the
entrance/exit. This formulation is then extended to the case of one-way driving
lanes. We also propose valid inequalities and a branch-and-cut algorithm for
the one-way and two-way lane configurations. This approach eliminates flow
variables, big-M type constraints, and improves solution times for medium-sized
instances. The effectiveness of the suggested models is showcased on 325
parking lots in New York City. For instances in which the flow version could be
solved in 15 minutes, the branch-and-cut algorithm improved the median runtimes
by 87.43% for the one-way case and by 79.36% for the two-way case and resulted
in better optimality gaps for the other instances, compared to the baseline
flow-based formulation. Similar advantages were observed when run with a time
budget of two hours. One-way configurations accommodated up to 18.63% more
vehicles on average than their two-way counterparts across all the instances.
Modifications to the proposed formulations that consider the turning
characteristics of vehicles and the presence of multiple entrances and exits
are also examined.

### 3. [Tight Paths and Tight Pairs in Weighted Directed Graphs](http://arxiv.org/pdf/2506.09966v1)

Authors: José Luis Balcázar

We state the graph-theoretic computational problem of finding tight paths in
a directed, edge-weighted graph, as well as its simplification of finding tight
pairs. These problems are motivated by the need of algorithms that find
so-called basic antecedents in closure spaces, in one specific approach to data
analysis. We discuss and compare several algorithms to approach these problems.

### 4. [Crossing numbers of dense graphs on surfaces](http://arxiv.org/pdf/2506.09974v1)

Authors: Alfredo Hubard, Arnaud de Mesmay, Hugo Parlier

In this paper, we provide upper and lower bounds on the crossing numbers of
dense graphs on surfaces, which match up to constant factors. First, we prove
that if $G$ is a dense enough graph with $m$ edges and $\Sigma$ is a surface of
genus $g$, then any drawing of $G$ on $\Sigma$ incurs at least $\Omega
\left(\frac{m^2}{g} \log ^2 g\right)$ crossings. The poly-logarithmic factor in
this lower bound is new even in the case of complete graphs and disproves a
conjecture of Shahrokhi, Sz\'ekely and Vrt'o from 1996. Then we prove a
geometric converse to this lower bound: we provide an explicit family of
hyperbolic surfaces such that for any graph $G$, sampling the vertices
uniformly at random on this surface and connecting them with shortest paths
yields $O\left(\frac{m^2}{g} \log ^2 g\right)$ crossings in expectation.

### Data Structures and Algorithms

### 1. [Tight Paths and Tight Pairs in Weighted Directed Graphs](http://arxiv.org/pdf/2506.09966v1)

Authors: José Luis Balcázar

We state the graph-theoretic computational problem of finding tight paths in
a directed, edge-weighted graph, as well as its simplification of finding tight
pairs. These problems are motivated by the need of algorithms that find
so-called basic antecedents in closure spaces, in one specific approach to data
analysis. We discuss and compare several algorithms to approach these problems.

### 2. [Almost-Optimal Local-Search Methods for Sparse Tensor PCA](http://arxiv.org/pdf/2506.09959v1)

Authors: Max Lovig, Conor Sheehan, Konstantinos Tsirkas, Ilias Zadik

Local-search methods are widely employed in statistical applications, yet
interestingly, their theoretical foundations remain rather underexplored,
compared to other classes of estimators such as low-degree polynomials and
spectral methods. Of note, among the few existing results recent studies have
revealed a significant "local-computational" gap in the context of a
well-studied sparse tensor principal component analysis (PCA), where a broad
class of local Markov chain methods exhibits a notable underperformance
relative to other polynomial-time algorithms. In this work, we propose a series
of local-search methods that provably "close" this gap to the best known
polynomial-time procedures in multiple regimes of the model, including and
going beyond the previously studied regimes in which the broad family of local
Markov chain methods underperforms. Our framework includes: (1) standard greedy
and randomized greedy algorithms applied to the (regularized) posterior of the
model; and (2) novel random-threshold variants, in which the randomized greedy
algorithm accepts a proposed transition if and only if the corresponding change
in the Hamiltonian exceeds a random Gaussian threshold-rather that if and only
if it is positive, as is customary. The introduction of the random thresholds
enables a tight mathematical analysis of the randomized greedy algorithm's
trajectory by crucially breaking the dependencies between the iterations, and
could be of independent interest to the community.

### Emerging Technologies

### 1. [Reliability of Capacitive Read in Arrays of Ferroelectric Capacitors](http://arxiv.org/pdf/2506.09480v1)

Authors: Luca Fehlings, Muhtasim Alam Chowdhury, Banafsheh Saber Latibari, Soheil Salehi, Erika Covi

The non-destructive capacitance read-out of ferroelectric capacitors (FeCaps)
based on doped HfO$_2$ metal-ferroelectric-metal (MFM) structures offers the
potential for low-power and highly scalable crossbar arrays. This is due to a
number of factors, including the selector-less design, the absence of sneak
paths, the power-efficient charge-based read operation, and the reduced IR
drop. Nevertheless, a reliable capacitive readout presents certain challenges,
particularly in regard to device variability and the trade-off between read
yield and read disturbances, which can ultimately result in bit-flips. This
paper presents a digital read macro for HfO$_2$ FeCaps and provides design
guidelines for capacitive readout of HfO$_2$ FeCaps, taking device-centric
reliability and yield challenges into account. An experimentally calibrated
physics-based compact model of HfO$_2$ FeCaps is employed to investigate the
reliability of the read-out operation of the FeCap macro through Monte Carlo
simulations. Based on this analysis, we identify limitations posed by the
device variability and propose potential mitigation strategies through
design-technology co-optimization (DTCO) of the FeCap device characteristics
and the CMOS circuit design. Finally, we examine the potential applications of
the FeCap macro in the context of secure hardware. We identify potential
security threats and propose strategies to enhance the robustness of the
system.

### 2. [Dynamic Hypergraph Partitioning of Quantum Circuits with Hybrid Execution](http://arxiv.org/pdf/2506.09963v1)

Authors: Shane Sweeney, Krishnendu Guha

Quantum algorithms offer an exponential speedup over classical algorithms for
a range of computational problems. The fundamental mechanisms underlying
quantum computation required the development and construction of quantum
computers. These devices are referred to as NISQ (Noisy Intermediate-Scale
Quantum) devices. Not only are NISQ devices extremely limited in their qubit
count but they also suffer from noise during computation and this problem only
gets worse as the size of the circuit increases which limits the practical use
of quantum computers for modern day applications. This paper will focus on
utilizing quantum circuit partitioning to overcome the inherent issues of NISQ
devices. Partitioning a quantum circuit into smaller subcircuits has allowed
for the execution of quantum circuits that are too large to fit on one quantum
device. There have been many previous approaches to quantum circuit
partitioning and each of these approaches differ in how they work with some
focusing on hardware-aware partitioning, optimal graph-based partitioning,
multi-processor architectures and many more. These approaches achieve success
in their objective but they often fail to scale well which impacts cost and
noise. The ultimate goal of this paper is to mitigate these issues by
minimizing 3 important metrics; noise, time and cost. To achieve this we use
dynamic partitioning for practical circuit cutting and we take advantage of the
benefits of hybrid execution where classical computation will be used alongside
quantum hardware. This approach has proved to be beneficial with respect to
noise with classical execution enabling a 42.30% reduction in noise and a 40%
reduction in the number of qubits required in cases where a mixture of
classical and quantum computation were required.

### 3. [On the Performance of Cloud-based ARM SVE for Zero-Knowledge Proving Systems](http://arxiv.org/pdf/2506.09505v1)

Authors: Dumitrel Loghin, Shuang Liang, Shengwei Liu, Xiong Liu, Pingcheng Ruan, Zhigang Ye

Zero-knowledge proofs (ZKP) are becoming a gold standard in scaling
blockchains and bringing Web3 to life. At the same time, ZKP for transactions
running on the Ethereum Virtual Machine require powerful servers with hundreds
of CPU cores. The current zkProver implementation from Polygon is optimized for
x86-64 CPUs by vectorizing key operations, such as Merkle tree building with
Poseidon hashes over the Goldilocks field, with Advanced Vector Extensions (AVX
and AVX512). With these optimizations, a ZKP for a batch of transactions is
generated in less than two minutes. With the advent of cloud servers with ARM
which are at least 10% cheaper than x86-64 servers and the implementation of
ARM Scalable Vector Extension (SVE), we wonder if ARM servers can take over
their x86-64 counterparts. Unfortunately, our analysis shows that current ARM
CPUs are not a match for their x86-64 competitors. Graviton4 from Amazon Web
Services (AWS) and Axion from Google Cloud Platform (GCP) are 1.6X and 1.4X
slower compared to the latest AMD EPYC and Intel Xeon servers from AWS with AVX
and AVX512, respectively, when building a Merkle tree with over four million
leaves. This low performance is due to (1) smaller vector size in these ARM
CPUs (128 bits versus 512 bits in AVX512) and (2) lower clock frequency. On the
other hand, ARM SVE/SVE2 Instruction Set Architecture (ISA) is at least as
powerful as AVX/AVX512 but more flexible. Moreover, we estimate that increasing
the vector size to 512 bits will enable higher performance in ARM CPUs compared
to their x86-64 counterparts while maintaining their price advantage.

### 4. [Mainframe-style channel controllers for modern disaggregated memory systems](http://arxiv.org/pdf/2506.09758v1)

Authors: Zikai Liu, Jasmin Schult, Pengcheng Xu, Timothy Roscoe

Despite the promise of alleviating the main memory bottleneck, and the
existence of commercial hardware implementations, techniques for Near-Data
Processing have seen relatively little real-world deployment. The idea has
received renewed interest with the appearance of disaggregated or "far" memory,
for example in the use of CXL memory pools.
  However, we argue that the lack of a clear OS-centric abstraction of
Near-Data Processing is a major barrier to adoption of the technology. Inspired
by the channel controllers which interface the CPU to disk drives in mainframe
systems, we propose memory channel controllers as a convenient, portable, and
virtualizable abstraction of Near-Data Processing for modern disaggregated
memory systems.
  In addition to providing a clean abstraction that enables OS integration
while requiring no changes to CPU architecture, memory channel controllers
incorporate another key innovation: they exploit the cache coherence provided
by emerging interconnects to provide a much richer programming model, with more
fine-grained interaction, than has been possible with existing designs.

### Graphics

### 1. [TransGI: Real-Time Dynamic Global Illumination With Object-Centric Neural Transfer Model](http://arxiv.org/pdf/2506.09909v1)

Authors: Yijie Deng, Lei Han, Lu Fang

Neural rendering algorithms have revolutionized computer graphics, yet their
impact on real-time rendering under arbitrary lighting conditions remains
limited due to strict latency constraints in practical applications. The key
challenge lies in formulating a compact yet expressive material representation.
To address this, we propose TransGI, a novel neural rendering method for
real-time, high-fidelity global illumination. It comprises an object-centric
neural transfer model for material representation and a radiance-sharing
lighting system for efficient illumination. Traditional BSDF representations
and spatial neural material representations lack expressiveness, requiring
thousands of ray evaluations to converge to noise-free colors. Conversely,
real-time methods trade quality for efficiency by supporting only diffuse
materials. In contrast, our object-centric neural transfer model achieves
compactness and expressiveness through an MLP-based decoder and vertex-attached
latent features, supporting glossy effects with low memory overhead. For
dynamic, varying lighting conditions, we introduce local light probes capturing
scene radiance, coupled with an across-probe radiance-sharing strategy for
efficient probe generation. We implemented our method in a real-time rendering
engine, combining compute shaders and CUDA-based neural networks. Experimental
results demonstrate that our method achieves real-time performance of less than
10 ms to render a frame and significantly improved rendering quality compared
to baseline methods.

### 2. [VideoMat: Extracting PBR Materials from Video Diffusion Models](http://arxiv.org/pdf/2506.09665v1)

Authors: Jacob Munkberg, Zian Wang, Ruofan Liang, Tianchang Shen, Jon Hasselgren

We leverage finetuned video diffusion models, intrinsic decomposition of
videos, and physically-based differentiable rendering to generate high quality
materials for 3D models given a text prompt or a single image. We condition a
video diffusion model to respect the input geometry and lighting condition.
This model produces multiple views of a given 3D model with coherent material
properties. Secondly, we use a recent model to extract intrinsics (base color,
roughness, metallic) from the generated video. Finally, we use the intrinsics
alongside the generated video in a differentiable path tracer to robustly
extract PBR materials directly compatible with common content creation tools.

### 3. [Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation](http://arxiv.org/pdf/2506.09485v1)

Authors: Yuxin Liu, Zhenghao Peng, Xuanhao Cui, Bolei Zhou

Scenario-based testing is essential for validating the performance of
autonomous driving (AD) systems. However, such testing is limited by the
scarcity of long-tailed, safety-critical scenarios in existing datasets
collected in the real world. To tackle the data issue, we propose the Adv-BMT
framework, which augments real-world scenarios with diverse and realistic
adversarial interactions. The core component of Adv-BMT is a bidirectional
motion transformer (BMT) model to perform inverse traffic motion predictions,
which takes agent information in the last time step of the scenario as input,
and reconstruct the traffic in the inverse of chronological order until the
initial time step. The Adv-BMT framework is a two-staged pipeline: it first
conducts adversarial initializations and then inverse motion predictions.
Different from previous work, we do not need any collision data for
pretraining, and are able to generate realistic and diverse collision
interactions. Our experimental results validate the quality of generated
collision scenarios by Adv-BMT: training in our augmented dataset would reduce
episode collision rates by 20\% compared to previous work.

### 4. [DGS-LRM: Real-Time Deformable 3D Gaussian Reconstruction From Monocular Videos](http://arxiv.org/pdf/2506.09997v1)

Authors: Chieh Hubert Lin, Zhaoyang Lv, Songyin Wu, Zhen Xu, Thu Nguyen-Phuoc, Hung-Yu Tseng, Julian Straub, Numair Khan, Lei Xiao, Ming-Hsuan Yang, Yuheng Ren, Richard Newcombe, Zhao Dong, Zhengqin Li

We introduce the Deformable Gaussian Splats Large Reconstruction Model
(DGS-LRM), the first feed-forward method predicting deformable 3D Gaussian
splats from a monocular posed video of any dynamic scene. Feed-forward scene
reconstruction has gained significant attention for its ability to rapidly
create digital replicas of real-world environments. However, most existing
models are limited to static scenes and fail to reconstruct the motion of
moving objects. Developing a feed-forward model for dynamic scene
reconstruction poses significant challenges, including the scarcity of training
data and the need for appropriate 3D representations and training paradigms. To
address these challenges, we introduce several key technical contributions: an
enhanced large-scale synthetic dataset with ground-truth multi-view videos and
dense 3D scene flow supervision; a per-pixel deformable 3D Gaussian
representation that is easy to learn, supports high-quality dynamic view
synthesis, and enables long-range 3D tracking; and a large transformer network
that achieves real-time, generalizable dynamic scene reconstruction. Extensive
qualitative and quantitative experiments demonstrate that DGS-LRM achieves
dynamic scene reconstruction quality comparable to optimization-based methods,
while significantly outperforming the state-of-the-art predictive dynamic
reconstruction method on real-world examples. Its predicted physically grounded
3D deformation is accurate and can readily adapt for long-range 3D tracking
tasks, achieving performance on par with state-of-the-art monocular video 3D
tracking methods.

### Computer Science and Game Theory

### 1. [Beyond Nash Equilibrium: Bounded Rationality of LLMs and humans in Strategic Decision-making](http://arxiv.org/pdf/2506.09390v1)

Authors: Kehan Zheng, Jinfeng Zhou, Hongning Wang

Large language models are increasingly used in strategic decision-making
settings, yet evidence shows that, like humans, they often deviate from full
rationality. In this study, we compare LLMs and humans using experimental
paradigms directly adapted from behavioral game-theory research. We focus on
two well-studied strategic games, Rock-Paper-Scissors and the Prisoner's
Dilemma, which are well known for revealing systematic departures from rational
play in human subjects. By placing LLMs in identical experimental conditions,
we evaluate whether their behaviors exhibit the bounded rationality
characteristic of humans. Our findings show that LLMs reproduce familiar human
heuristics, such as outcome-based strategy switching and increased cooperation
when future interaction is possible, but they apply these rules more rigidly
and demonstrate weaker sensitivity to the dynamic changes in the game
environment. Model-level analyses reveal distinctive architectural signatures
in strategic behavior, and even reasoning models sometimes struggle to find
effective strategies in adaptive situations. These results indicate that
current LLMs capture only a partial form of human-like bounded rationality and
highlight the need for training methods that encourage flexible opponent
modeling and stronger context awareness.

### 2. [Metritocracy: Representative Metrics for Lite Benchmarks](http://arxiv.org/pdf/2506.09813v1)

Authors: Ariel Procaccia, Benjamin Schiffer, Serena Wang, Shirley Zhang

A common problem in LLM evaluation is how to choose a subset of metrics from
a full suite of possible metrics. Subset selection is usually done for
efficiency or interpretability reasons, and the goal is often to select a
``representative'' subset of metrics. However, ``representative'' is rarely
clearly defined. In this work, we use ideas from social choice theory to
formalize two notions of representation for the selection of a subset of
evaluation metrics. We first introduce positional representation, which
guarantees every alternative is sufficiently represented at every position
cutoff. We then introduce positional proportionality, which guarantees no
alternative is proportionally over- or under-represented by more than a small
error at any position. We prove upper and lower bounds on the smallest number
of metrics needed to guarantee either of these properties in the worst case. We
also study a generalized form of each property that allows for additional input
on groups of metrics that must be represented. Finally, we tie theory to
practice through real-world case studies on both LLM evaluation and hospital
quality evaluation.

### 3. [Enhanced V2X Communication Using Game-Theory Based Adaptive MAC Protocols](http://arxiv.org/pdf/2506.09817v1)

Authors: Dhrumil Bhatt, Nirbhay Singhal

This paper presents an enhanced Vehicle-to-Everything (V2X) communication
system featuring adaptive Medium Access Control (MAC) using game theory. Our
approach integrates dynamic transmission power control, dynamic beacon rates,
contention window adaptation, and implicit acknowledgment mechanisms within a
Manhattan-like grid-based mobility scenario. Simulations are conducted in a
circular coverage area, incorporating refined signal propagation models and
probabilistic vehicle mobility with boundary reflection. The results
demonstrate effective beacon delivery with average delays under 0.35 s and
packet loss rates less than 1% in high-density conditions specifically, with up
to 80 vehicles operating within a 250 m radius. Key innovations include game
theory-based environment-aware transmission parameter adaptation and a scalable
design suited for interference-prone V2X deployments.

### Human-Computer Interaction

### 1. [Patterns of Patterns III](http://arxiv.org/pdf/2506.09696v1)

Authors: Joseph Corneli, Charles J. Danoff, Raymond S. Puzio, Sridevi Ayloo, Serge Belich, Mary Tedeschi

Building on earlier installments, this paper re-examines the PLACARD pattern.
We report on a series of workshops where PLACARD was used to scaffold
collaborative reflection, speculative inquiry, and stimulate design pattern
generation. These accounts are enriched by a comparison case: virtual workshops
carried out with simple AI-based chatbots. We discuss limitations and lessons
learned from both the human and multi-agent settings. We conclude by outlining
a future development strategy at the intersection of AI agents, design
patterns, and institutional governance.

### 2. [Investigating the Perception of Translational Shape-Changing Haptic Interfaces](http://arxiv.org/pdf/2506.09801v1)

Authors: Qihan Yang, Xin Zhou, Adam J. Spiers

Shape-changing haptic interfaces (SCHIs) are a promising and emerging field.
However, compared to more established stimulus modalities, such as vibration,
there is sparse literature on the perception of dynamic shapes. Furthermore,
the influence of properties such as grasp types and displacement
magnitude/direction has not been formally evaluated. This work attempts to
initiate a formal perceptual evaluation of SCHIs via a psychophysical user
study involving a 1-DOF translational shape-changing interface that can move
its body with 1.25-micrometer resolution. Participants completed a Method of
Constant Stimulus study while holding the device with three different grasps.
Stimuli direction occurred both toward and away from the thumb, while the
standard stimuli varied between small (0.48 mm) and large (6 mm). Our results
indicate that translational SCHIs should maximize the translation magnitude
rather than the number of fingers in contact. We also demonstrated how to apply
our findings to real-world applications via a simple 'paddle game', where we
compared conventional linear mapping with non-linear mapping derived from our
perceptual experiment outcomes between the device position and its represented
value. Results indicate that the non-linear mapping was more effective, with
improved error distribution. We hope this work inspires further formal
perceptual investigation into other SCHI morphologies.

### 3. [SRLAgent: Enhancing Self-Regulated Learning Skills through Gamification and LLM Assistance](http://arxiv.org/pdf/2506.09968v1)

Authors: Wentao Ge, Yuqing Sun, Ziyan Wang, Haoyue Zheng, Weiyang He, Piaohong Wang, Qianyu Zhu, Benyou Wang

Self-regulated learning (SRL) is crucial for college students navigating
increased academic demands and independence. Insufficient SRL skills can lead
to disorganized study habits, low motivation, and poor time management,
undermining learners ability to thrive in challenging environments. Through a
formative study involving 59 college students, we identified key challenges
students face in developing SRL skills, including difficulties with
goal-setting, time management, and reflective learning. To address these
challenges, we introduce SRLAgent, an LLM-assisted system that fosters SRL
skills through gamification and adaptive support from large language models
(LLMs). Grounded in Zimmermans three-phase SRL framework, SRLAgent enables
students to engage in goal-setting, strategy execution, and self-reflection
within an interactive game-based environment. The system offers real-time
feedback and scaffolding powered by LLMs to support students independent study
efforts. We evaluated SRLAgent using a between-subjects design, comparing it to
a baseline system (SRL without Agent features) and a traditional multimedia
learning condition. Results showed significant improvements in SRL skills
within the SRLAgent group (p < .001, Cohens d = 0.234) and higher engagement
compared to the baselines. This work highlights the value of embedding SRL
scaffolding and real-time AI support within gamified environments, offering
design implications for educational technologies that aim to promote deeper
learning and metacognitive skill development.

### 4. ["Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions](http://arxiv.org/pdf/2506.09354v1)

Authors: Kellie Yu Hui Sim, Roy Ka-Wei Lee, Kenny Tsu Wei Choo

Mental health is a growing global concern, prompting interest in AI-driven
solutions to expand access to psychosocial support. Peer support, grounded in
lived experience, offers a valuable complement to professional care. However,
variability in training, effectiveness, and definitions raises concerns about
quality, consistency, and safety. Large Language Models (LLMs) present new
opportunities to enhance peer support interactions, particularly in real-time,
text-based interactions. We present and evaluate an AI-supported system with an
LLM-simulated distressed client, context-sensitive LLM-generated suggestions,
and real-time emotion visualisations. 2 mixed-methods studies with 12 peer
supporters and 5 mental health professionals (i.e., experts) examined the
system's effectiveness and implications for practice. Both groups recognised
its potential to enhance training and improve interaction quality. However, we
found a key tension emerged: while peer supporters engaged meaningfully,
experts consistently flagged critical issues in peer supporter responses, such
as missed distress cues and premature advice-giving. This misalignment
highlights potential limitations in current peer support training, especially
in emotionally charged contexts where safety and fidelity to best practices are
essential. Our findings underscore the need for standardised, psychologically
grounded training, especially as peer support scales globally. They also
demonstrate how LLM-supported systems can scaffold this development--if
designed with care and guided by expert oversight. This work contributes to
emerging conversations on responsible AI integration in mental health and the
evolving role of LLMs in augmenting peer-delivered care.

### 5. ["I Said Things I Needed to Hear Myself": Peer Support as an Emotional, Organisational, and Sociotechnical Practice in Singapore](http://arxiv.org/pdf/2506.09362v1)

Authors: Kellie Yu Hui Sim, Kenny Tsu Wei Choo

Peer support plays a vital role in expanding access to mental health care by
providing empathetic, community-based support outside formal clinical systems.
As digital platforms increasingly mediate such support, the design and impact
of these technologies remain under-examined, particularly in Asian contexts.
This paper presents findings from an interview study with 20 peer supporters in
Singapore, who operate across diverse online, offline, and hybrid environments.
Through a thematic analysis, we unpack how participants start, conduct, and
sustain peer support, highlighting their motivations, emotional labour, and the
sociocultural dimensions shaping their practices. Building on this grounded
understanding, we surface design directions for culturally responsive digital
tools that scaffold rather than supplant relational care. Drawing insights from
qualitative accounts, we offer a situated perspective on how AI might
responsibly augment peer support. This research contributes to human-centred
computing by articulating the lived realities of peer supporters and proposing
design implications for trustworthy and context-sensitive AI in mental health.

### 6. [Fine-Tuning Large Audio-Language Models with LoRA for Precise Temporal Localization of Prolonged Exposure Therapy Elements](http://arxiv.org/pdf/2506.09707v1)

Authors: Suhas BN, Andrew M. Sherrill, Jyoti Alaparthi, Dominik Mattioli, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah

Prolonged Exposure (PE) therapy is an effective treatment for post-traumatic
stress disorder (PTSD), but evaluating therapist fidelity remains
labor-intensive due to the need for manual review of session recordings. We
present a method for the automatic temporal localization of key PE fidelity
elements -- identifying their start and stop times -- directly from session
audio and transcripts. Our approach fine-tunes a large pre-trained
audio-language model, Qwen2-Audio, using Low-Rank Adaptation (LoRA) to process
focused 30-second windows of audio-transcript input. Fidelity labels for three
core protocol phases -- therapist orientation (P1), imaginal exposure (P2), and
post-imaginal processing (P3) -- are generated via LLM-based prompting and
verified by trained raters. The model is trained to predict normalized boundary
offsets using soft supervision guided by task-specific prompts. On a dataset of
313 real PE sessions, our best configuration (LoRA rank 8, 30s windows)
achieves a mean absolute error (MAE) of 5.3 seconds across tasks. We further
analyze the effects of window size and LoRA rank, highlighting the importance
of context granularity and model adaptation. This work introduces a scalable
framework for fidelity tracking in PE therapy, with potential to support
clinician training, supervision, and quality assurance.

### 7. [Stakeholder Participation for Responsible AI Development: Disconnects Between Guidance and Current Practice](http://arxiv.org/pdf/2506.09873v1)

Authors: Emma Kallina, Thomas Bohné, Jat Singh

Responsible AI (rAI) guidance increasingly promotes stakeholder involvement
(SHI) during AI development. At the same time, SHI is already common in
commercial software development, but with potentially different foci. This
study clarifies the extent to which established SHI practices are able to
contribute to rAI efforts as well as potential disconnects -- essential
insights to inform and tailor future interventions that further shift industry
practice towards rAI efforts. First, we analysed 56 rAI guidance documents to
identify why SHI is recommended (i.e. its expected benefits for rAI) and
uncovered goals such as redistributing power, improving socio-technical
understandings, anticipating risks, and enhancing public oversight. To
understand why and how SHI is currently practised in commercial settings, we
then conducted an online survey (n=130) and semi-structured interviews (n=10)
with AI practitioners. Our findings reveal that SHI in practice is primarily
driven by commercial priorities (e.g. customer value, compliance) and several
factors currently discourage more rAI-aligned SHI practices. This suggests that
established SHI practices are largely not contributing to rAI efforts. To
address this disconnect, we propose interventions and research opportunities to
advance rAI development in practice.

### 8. [A Call for Collaborative Intelligence: Why Human-Agent Systems Should Precede AI Autonomy](http://arxiv.org/pdf/2506.09420v1)

Authors: Henry Peng Zou, Wei-Chieh Huang, Yaozu Wu, Chunyu Miao, Dongyuan Li, Aiwei Liu, Yue Zhou, Yankai Chen, Weizhi Zhang, Yangning Li, Liancheng Fang, Renhe Jiang, Philip S. Yu

Recent improvements in large language models (LLMs) have led many researchers
to focus on building fully autonomous AI agents. This position paper questions
whether this approach is the right path forward, as these autonomous systems
still have problems with reliability, transparency, and understanding the
actual requirements of human. We suggest a different approach: LLM-based
Human-Agent Systems (LLM-HAS), where AI works with humans rather than replacing
them. By keeping human involved to provide guidance, answer questions, and
maintain control, these systems can be more trustworthy and adaptable. Looking
at examples from healthcare, finance, and software development, we show how
human-AI teamwork can handle complex tasks better than AI working alone. We
also discuss the challenges of building these collaborative systems and offer
practical solutions. This paper argues that progress in AI should not be
measured by how independent systems become, but by how well they can work with
humans. The most promising future for AI is not in systems that take over human
roles, but in those that enhance human capabilities through meaningful
partnership.

### Information Retrieval

### 1. [MAGMaR Shared Task System Description: Video Retrieval with OmniEmbed](http://arxiv.org/pdf/2506.09409v1)

Authors: Jiaqi Samantha Zhan, Crystina Zhang, Shengyao Zhuang, Xueguang Ma, Jimmy Lin

Effective video retrieval remains challenging due to the complexity of
integrating visual, auditory, and textual modalities. In this paper, we explore
unified retrieval methods using OmniEmbed, a powerful multimodal embedding
model from the Tevatron 2.0 toolkit, in the context of the MAGMaR shared task.
Evaluated on the comprehensive MultiVENT 2.0 dataset, OmniEmbed generates
unified embeddings for text, images, audio, and video, enabling robust
multimodal retrieval. By finetuning OmniEmbed with the combined multimodal
data--visual frames, audio tracks, and textual descriptions provided in
MultiVENT 2.0, we achieve substantial improvements in complex, multilingual
video retrieval tasks. Our submission achieved the highest score on the MAGMaR
shared task leaderboard among public submissions as of May 20th, 2025,
highlighting the practical effectiveness of our unified multimodal retrieval
approach. Model checkpoint in this work is opensourced.

### 2. [Discrete Scale-invariant Metric Learning for Efficient Collaborative Filtering](http://arxiv.org/pdf/2506.09898v1)

Authors: Yan Zhang, Li Deng, Lixin Duan, Sami Azam

Metric learning has attracted extensive interest for its ability to provide
personalized recommendations based on the importance of observed user-item
interactions. Current metric learning methods aim to push negative items away
from the corresponding users and positive items by an absolute geometrical
distance margin. However, items may come from imbalanced categories with
different intra-class variations. Thus, the absolute distance margin may not be
ideal for estimating the difference between user preferences over imbalanced
items. To this end, we propose a new method, named discrete scale-invariant
metric learning (DSIML), by adding binary constraints to users and items, which
maps users and items into binary codes of a shared Hamming subspace to speed up
the online recommendation. Specifically, we firstly propose a scale-invariant
margin based on angles at the negative item points in the shared Hamming
subspace. Then, we derive a scale-invariant triple hinge loss based on the
margin. To capture more preference difference information, we integrate a
pairwise ranking loss into the scale-invariant loss in the proposed model. Due
to the difficulty of directly optimizing the mixed integer optimization problem
formulated with \textit{log-sum-exp} functions, we seek to optimize its
variational quadratic upper bound and learn hash codes with an alternating
optimization strategy. Experiments on benchmark datasets clearly show that our
proposed method is superior to competitive metric learning and hashing-based
baselines for recommender systems. The implementation code is available at
https://github.com/AnonyFeb/dsml.

### 3. [PGDA-KGQA: A Prompt-Guided Generative Framework with Multiple Data Augmentation Strategies for Knowledge Graph Question Answering](http://arxiv.org/pdf/2506.09414v1)

Authors: Xiujun Zhou, Pingjian Zhang, Deyou Tang

Knowledge Graph Question Answering (KGQA) is a crucial task in natural
language processing that requires reasoning over knowledge graphs (KGs) to
answer natural language questions. Recent methods utilizing large language
models (LLMs) have shown remarkable semantic parsing capabilities but are
limited by the scarcity of diverse annotated data and multi-hop reasoning
samples. Traditional data augmentation approaches are focus mainly on
single-hop questions and prone to semantic distortion, while LLM-based methods
primarily address semantic distortion but usually neglect multi-hop reasoning,
thus limiting data diversity. The scarcity of multi-hop samples further weakens
models' generalization. To address these issues, we propose PGDA-KGQA, a
prompt-guided generative framework with multiple data augmentation strategies
for KGQA. At its core, PGDA-KGQA employs a unified prompt-design paradigm: by
crafting meticulously engineered prompts that integrate the provided textual
content, it leverages LLMs to generate large-scale (question, logical form)
pairs for model training. Specifically, PGDA-KGQA enriches its training set by:
(1) generating single-hop pseudo questions to improve the alignment of question
semantics with KG relations; (2) applying semantic-preserving question
rewriting to improve robustness against linguistic variations; (3) employing
answer-guided reverse path exploration to create realistic multi-hop questions.
By adopting an augment-generate-retrieve semantic parsing pipeline, PGDA-KGQA
utilizes the augmented data to enhance the accuracy of logical form generation
and thus improve answer retrieval performance. Experiments demonstrate that
outperforms state-of-the-art methods on standard KGQA datasets, achieving
improvements on WebQSP by 2.8%, 1.2%, and 3.1% and on ComplexWebQuestions by
1.8%, 1.1%, and 2.4% in F1, Hits@1, and Accuracy, respectively.

### 4. [Learning Efficient and Generalizable Graph Retriever for Knowledge-Graph Question Answering](http://arxiv.org/pdf/2506.09645v1)

Authors: Tianjun Yao, Haoxuan Li, Zhiqiang Shen, Pan Li, Tongliang Liu, Kun Zhang

Large Language Models (LLMs) have shown strong inductive reasoning ability
across various domains, but their reliability is hindered by the outdated
knowledge and hallucinations. Retrieval-Augmented Generation mitigates these
issues by grounding LLMs with external knowledge; however, most existing RAG
pipelines rely on unstructured text, limiting interpretability and structured
reasoning. Knowledge graphs, which represent facts as relational triples, offer
a more structured and compact alternative. Recent studies have explored
integrating knowledge graphs with LLMs for knowledge graph question answering
(KGQA), with a significant proportion adopting the retrieve-then-reasoning
paradigm. In this framework, graph-based retrievers have demonstrated strong
empirical performance, yet they still face challenges in generalization
ability. In this work, we propose RAPL, a novel framework for efficient and
effective graph retrieval in KGQA. RAPL addresses these limitations through
three aspects: (1) a two-stage labeling strategy that combines heuristic
signals with parametric models to provide causally grounded supervision; (2) a
model-agnostic graph transformation approach to capture both intra- and
inter-triple interactions, thereby enhancing representational capacity; and (3)
a path-based reasoning strategy that facilitates learning from the injected
rational knowledge, and supports downstream reasoner through structured inputs.
Empirically, RAPL outperforms state-of-the-art methods by $2.66\%-20.34\%$, and
significantly reduces the performance gap between smaller and more powerful
LLM-based reasoners, as well as the gap under cross-dataset settings,
highlighting its superior retrieval capability and generalizability. Codes are
available at: https://github.com/tianyao-aka/RAPL.

### Machine Learning

### 1. [On-the-Fly Adaptive Distillation of Transformer to Dual-State Linear Attention](http://arxiv.org/pdf/2506.09316v1)

Authors: Yeonju Ro, Zhenyu Zhang, Souvik Kundu, Zhangyang Wang, Aditya Akella

Large language models (LLMs) excel at capturing global token dependencies via
self-attention but face prohibitive compute and memory costs on lengthy inputs.
While sub-quadratic methods (e.g., linear attention) can reduce these costs,
they often degrade accuracy due to overemphasizing recent tokens. In this work,
we first propose \textit{dual-state linear attention} (\textbf{\dsla}), a novel
design that maintains two specialized hidden states-one for preserving
historical context and one for tracking recency-thereby mitigating the
short-range bias typical of linear-attention architectures. To further balance
efficiency and accuracy under dynamic workload conditions, we introduce
\textbf{\serve}, an online \textit{adaptive distillation} framework that
progressively replaces Transformer layers with DSLA layers at inference time,
guided by a sensitivity-based layer ordering. \serve\ uses a chained
fine-tuning strategy to ensure that each newly converted DSLA layer remains
consistent with previously replaced layers, preserving the overall quality.
Extensive evaluations on commonsense reasoning, long-context QA, and text
summarization demonstrate that \serve\ yields \textbf{2.3x} faster inference
than Llama2-7B and \textbf{3.0x} faster than the hybrid Zamba-7B, while
retaining comparable performance across downstream tasks. Our ablation studies
show that DSLA's dual states capture both global and local dependencies,
addressing the historical-token underrepresentation seen in prior linear
attentions. Codes are available at https://github.com/utnslab/DSLA-Serve.

### 2. [Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation](http://arxiv.org/pdf/2506.09376v1)

Authors: Bowen Zheng, Tianming Yang

Diffusion distillation is a widely used technique to reduce the sampling cost
of diffusion models, yet it often requires extensive training, and the student
performance tends to be degraded. Recent studies show that incorporating a GAN
objective may alleviate these issues, yet the underlying mechanism remains
unclear. In this work, we first identify a key limitation of distillation:
mismatched step sizes and parameter numbers between the teacher and the student
model lead them to converge to different local minima, rendering direct
imitation suboptimal. We further demonstrate that a standalone GAN objective,
without relying a distillation loss, overcomes this limitation and is
sufficient to convert diffusion models into efficient one-step generators.
Based on this finding, we propose that diffusion training may be viewed as a
form of generative pre-training, equipping models with capabilities that can be
unlocked through lightweight GAN fine-tuning. Supporting this view, we create a
one-step generation model by fine-tuning a pre-trained model with 85% of
parameters frozen, achieving strong performance with only 0.2M images and
near-SOTA results with 5M images. We further present a frequency-domain
analysis that may explain the one-step generative capability gained in
diffusion training. Overall, our work provides a new perspective for diffusion
training, highlighting its role as a powerful generative pre-training process,
which can be the basis for building efficient one-step generation models.

### 3. [Mitigating Spurious Correlations in LLMs via Causality-Aware Post-Training](http://arxiv.org/pdf/2506.09433v1)

Authors: Shurui Gui, Shuiwang Ji

While large language models (LLMs) have demonstrated remarkable capabilities
in language modeling, recent studies reveal that they often fail on
out-of-distribution (OOD) samples due to spurious correlations acquired during
pre-training. Here, we aim to mitigate such spurious correlations through
causality-aware post-training (CAPT). By decomposing a biased prediction into
two unbiased steps, known as \textit{event estimation} and \textit{event
intervention}, we reduce LLMs' pre-training biases without incurring additional
fine-tuning biases, thus enhancing the model's generalization ability.
Experiments on the formal causal inference benchmark CLadder and the logical
reasoning dataset PrOntoQA show that 3B-scale language models fine-tuned with
CAPT can outperform both traditional SFT and larger LLMs on in-distribution
(ID) and OOD tasks using only 100 ID fine-tuning samples, demonstrating the
effectiveness and sample efficiency of CAPT.

### 4. [NDCG-Consistent Softmax Approximation with Accelerated Convergence](http://arxiv.org/pdf/2506.09454v1)

Authors: Yuanhao Pu, Defu Lian, Xiaolong Chen, Xu Huang, Jin Chen, Enhong Chen

Ranking tasks constitute fundamental components of extreme similarity
learning frameworks, where extremely large corpora of objects are modeled
through relative similarity relationships adhering to predefined ordinal
structures. Among various ranking surrogates, Softmax (SM) Loss has been widely
adopted due to its natural capability to handle listwise ranking via global
negative comparisons, along with its flexibility across diverse application
scenarios. However, despite its effectiveness, SM Loss often suffers from
significant computational overhead and scalability limitations when applied to
large-scale object spaces. To address this challenge, we propose novel loss
formulations that align directly with ranking metrics: the
Ranking-Generalizable \textbf{squared} (RG$^2$) Loss and the
Ranking-Generalizable interactive (RG$^\times$) Loss, both derived through
Taylor expansions of the SM Loss. Notably, RG$^2$ reveals the intrinsic
mechanisms underlying weighted squared losses (WSL) in ranking methods and
uncovers fundamental connections between sampling-based and non-sampling-based
loss paradigms. Furthermore, we integrate the proposed RG losses with the
highly efficient Alternating Least Squares (ALS) optimization method, providing
both generalization guarantees and convergence rate analyses. Empirical
evaluations on real-world datasets demonstrate that our approach achieves
comparable or superior ranking performance relative to SM Loss, while
significantly accelerating convergence. This framework offers the similarity
learning community both theoretical insights and practically efficient tools,
with methodologies applicable to a broad range of tasks where balancing ranking
quality and computational efficiency is essential.

### 5. [On a few pitfalls in KL divergence gradient estimation for RL](http://arxiv.org/pdf/2506.09477v1)

Authors: Yunhao Tang, Rémi Munos

We point out a few pitfalls in implementing gradient estimation for KL
divergence in RL training for LLM, as seen in a number of open source projects
and papers. The first major pitfall is to differentiate through the KL estimate
as loss functions to minimize KL divergence. We show that such implementations
are generally incorrect and do not produce the desired KL gradient. Secondly,
we show that some implementations do not account for the sequential nature of
the estimation problem and produce a partial gradient at best. We demonstrate
the impact of such issues with illustrative tabular and LLM experiments, and
show the correct way to implement the KL gradient.

### 6. [STOAT: Spatial-Temporal Probabilistic Causal Inference Network](http://arxiv.org/pdf/2506.09544v1)

Authors: Yang Yang, Du Yin, Hao Xue, Flora Salim

Spatial-temporal causal time series (STC-TS) involve region-specific temporal
observations driven by causally relevant covariates and interconnected across
geographic or network-based spaces. Existing methods often model spatial and
temporal dynamics independently and overlook causality-driven probabilistic
forecasting, limiting their predictive power. To address this, we propose STOAT
(Spatial-Temporal Probabilistic Causal Inference Network), a novel framework
for probabilistic forecasting in STC-TS. The proposed method extends a causal
inference approach by incorporating a spatial relation matrix that encodes
interregional dependencies (e.g. proximity or connectivity), enabling spatially
informed causal effect estimation. The resulting latent series are processed by
deep probabilistic models to estimate the parameters of the distributions,
enabling calibrated uncertainty modeling. We further explore multiple output
distributions (e.g., Gaussian, Student's-$t$, Laplace) to capture
region-specific variability. Experiments on COVID-19 data across six countries
demonstrate that STOAT outperforms state-of-the-art probabilistic forecasting
models (DeepAR, DeepVAR, Deep State Space Model, etc.) in key metrics,
particularly in regions with strong spatial dependencies. By bridging causal
inference and geospatial probabilistic forecasting, STOAT offers a
generalizable framework for complex spatial-temporal tasks, such as epidemic
management.

### 7. [MOORL: A Framework for Integrating Offline-Online Reinforcement Learning](http://arxiv.org/pdf/2506.09574v1)

Authors: Gaurav Chaudhary, Wassim Uddin Mondal, Laxmidhar Behera

Sample efficiency and exploration remain critical challenges in Deep
Reinforcement Learning (DRL), particularly in complex domains. Offline RL,
which enables agents to learn optimal policies from static, pre-collected
datasets, has emerged as a promising alternative. However, offline RL is
constrained by issues such as out-of-distribution (OOD) actions that limit
policy performance and generalization. To overcome these limitations, we
propose Meta Offline-Online Reinforcement Learning (MOORL), a hybrid framework
that unifies offline and online RL for efficient and scalable learning. While
previous hybrid methods rely on extensive design components and added
computational complexity to utilize offline data effectively, MOORL introduces
a meta-policy that seamlessly adapts across offline and online trajectories.
This enables the agent to leverage offline data for robust initialization while
utilizing online interactions to drive efficient exploration. Our theoretical
analysis demonstrates that the hybrid approach enhances exploration by
effectively combining the complementary strengths of offline and online data.
Furthermore, we demonstrate that MOORL learns a stable Q-function without added
complexity. Extensive experiments on 28 tasks from the D4RL and V-D4RL
benchmarks validate its effectiveness, showing consistent improvements over
state-of-the-art offline and hybrid RL baselines. With minimal computational
overhead, MOORL achieves strong performance, underscoring its potential for
practical applications in real-world scenarios.

### 8. [Beyond Overconfidence: Foundation Models Redefine Calibration in Deep Neural Networks](http://arxiv.org/pdf/2506.09593v1)

Authors: Achim Hekler, Lukas Kuhn, Florian Buettner

Reliable uncertainty calibration is essential for safely deploying deep
neural networks in high-stakes applications. Deep neural networks are known to
exhibit systematic overconfidence, especially under distribution shifts.
Although foundation models such as ConvNeXt, EVA and BEiT have demonstrated
significant improvements in predictive performance, their calibration
properties remain underexplored. This paper presents a comprehensive
investigation into the calibration behavior of foundation models, revealing
insights that challenge established paradigms. Our empirical analysis shows
that these models tend to be underconfident in in-distribution predictions,
resulting in higher calibration errors, while demonstrating improved
calibration under distribution shifts. Furthermore, we demonstrate that
foundation models are highly responsive to post-hoc calibration techniques in
the in-distribution setting, enabling practitioners to effectively mitigate
underconfidence bias. However, these methods become progressively less reliable
under severe distribution shifts and can occasionally produce counterproductive
results. Our findings highlight the complex, non-monotonic effects of
architectural and training innovations on calibration, challenging established
narratives of continuous improvement.

### 9. [Accelerating Large-Scale Regularized High-Order Tensor Recovery](http://arxiv.org/pdf/2506.09594v1)

Authors: Wenjin Qin, Hailin Wang, Jingyao Hou, Jianjun Wang

Currently, existing tensor recovery methods fail to recognize the impact of
tensor scale variations on their structural characteristics. Furthermore,
existing studies face prohibitive computational costs when dealing with
large-scale high-order tensor data. To alleviate these issue, assisted by the
Krylov subspace iteration, block Lanczos bidiagonalization process, and random
projection strategies, this article first devises two fast and accurate
randomized algorithms for low-rank tensor approximation (LRTA) problem.
Theoretical bounds on the accuracy of the approximation error estimate are
established. Next, we develop a novel generalized nonconvex modeling framework
tailored to large-scale tensor recovery, in which a new regularization paradigm
is exploited to achieve insightful prior representation for large-scale
tensors. On the basis of the above, we further investigate new unified
nonconvex models and efficient optimization algorithms, respectively, for
several typical high-order tensor recovery tasks in unquantized and quantized
situations. To render the proposed algorithms practical and efficient for
large-scale tensor data, the proposed randomized LRTA schemes are integrated
into their central and time-intensive computations. Finally, we conduct
extensive experiments on various large-scale tensors, whose results demonstrate
the practicability, effectiveness and superiority of the proposed method in
comparison with some state-of-the-art approaches.

### 10. [SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot](http://arxiv.org/pdf/2506.09613v1)

Authors: Kaiwen Tuo, Huan Wang

State-space language models such as Mamba match Transformer quality while
permitting linear complexity inference, yet still comprise billions of
parameters that hinder deployment. Existing one-shot pruning methods are
tailored to attention blocks and fail to account for the time-shared and
discretized state-transition matrix at the heart of the selective state-space
module (SSM). In this paper, we introduce SparseSSM, the first training-free
pruning framework that extends the classic optimal brain surgeon (OBS)
framework to state space architectures. Our layer-wise algorithm (i) derives an
approximate second-order saliency score that aggregates Hessian-trace
information across time steps, (ii) incorporates a component sensitivity
analysis to guide feed-forward network (FFN) pruning, which also sheds light on
where redundancy resides in mamba architecture, (iii) can be easily extended to
semi-structured and structured sparsity. Empirically, we prune 50% of SSM
weights without fine-tuning and observe no zero-shot accuracy loss, achieving
the current state-of-the-art pruning algorithm for Mamba-based LLMs.

### Neural and Evolutionary Computing

### 1. [Energy Aware Development of Neuromorphic Implantables: From Metrics to Action](http://arxiv.org/pdf/2506.09599v1)

Authors: Enrique Barba Roque, Luis Cruz

Spiking Neural Networks (SNNs) and neuromorphic computing present a promising
alternative to traditional Artificial Neural Networks (ANNs) by significantly
improving energy efficiency, particularly in edge and implantable devices.
However, assessing the energy performance of SNN models remains a challenge due
to the lack of standardized and actionable metrics and the difficulty of
measuring energy consumption in experimental neuromorphic hardware. In this
paper, we conduct a preliminary exploratory study of energy efficiency metrics
proposed in the SNN benchmarking literature. We classify 13 commonly used
metrics based on four key properties: Accessibility, Fidelity, Actionability,
and Trend-Based analysis. Our findings indicate that while many existing
metrics provide useful comparisons between architectures, they often lack
practical insights for SNN developers. Notably, we identify a gap between
accessible and high-fidelity metrics, limiting early-stage energy assessment.
Additionally, we emphasize the lack of metrics that provide practitioners with
actionable insights, making it difficult to guide energy-efficient SNN
development. To address these challenges, we outline research directions for
bridging accessibility and fidelity and finding new Actionable metrics for
implantable neuromorphic devices, introducing more Trend-Based metrics, metrics
that reflect changes in power requirements, battery-aware metrics, and
improving energy-performance tradeoff assessments. The results from this paper
pave the way for future research on enhancing energy metrics and their
Actionability for SNNs.

### 2. [Synergizing Reinforcement Learning and Genetic Algorithms for Neural Combinatorial Optimization](http://arxiv.org/pdf/2506.09404v1)

Authors: Shengda Gu, Kai Li, Junliang Xing, Yifan Zhang, Jian Cheng

Combinatorial optimization problems are notoriously challenging due to their
discrete structure and exponentially large solution space. Recent advances in
deep reinforcement learning (DRL) have enabled the learning heuristics directly
from data. However, DRL methods often suffer from limited exploration and
susceptibility to local optima. On the other hand, evolutionary algorithms such
as Genetic Algorithms (GAs) exhibit strong global exploration capabilities but
are typically sample inefficient and computationally intensive. In this work,
we propose the Evolutionary Augmentation Mechanism (EAM), a general and
plug-and-play framework that synergizes the learning efficiency of DRL with the
global search power of GAs. EAM operates by generating solutions from a learned
policy and refining them through domain-specific genetic operations such as
crossover and mutation. These evolved solutions are then selectively reinjected
into the policy training loop, thereby enhancing exploration and accelerating
convergence. We further provide a theoretical analysis that establishes an
upper bound on the KL divergence between the evolved solution distribution and
the policy distribution, ensuring stable and effective policy updates. EAM is
model-agnostic and can be seamlessly integrated with state-of-the-art DRL
solvers such as the Attention Model, POMO, and SymNCO. Extensive results on
benchmark problems (e.g., TSP, CVRP, PCTSP, and OP) demonstrate that EAM
significantly improves both solution quality and training efficiency over
competitive baselines.

### Networking and Internet Architecture

### 1. [Multi-Level Damage-Aware Graph Learning for Resilient UAV Swarm Networks](http://arxiv.org/pdf/2506.09703v1)

Authors: Huan Lin, Chenguang Zhu, Lianghui Ding, Feng Yang

Unmanned aerial vehicle (UAV) swarm networks leverage resilient algorithms to
address communication network split issues and restore connectivity. However,
existing graph learning-based resilient algorithms face over-aggregation and
non-convergence problems caused by uneven and sparse topology under massive
damage scenarios. To alleviate these problems, we propose a novel Multi-Level
Damage-Aware Graph Learning (ML-DAGL) algorithm, which generates recovery
trajectories by mining information from destroyed UAVs. We first introduce a
Multi-Branch Damage Attention (MBDA) module, which forms a sequence of
multi-hop Damage Attentive Graphs (mDAG) with different ranges of receptive
fields. Each mDAG links only remaining and damaged nodes to ensure a more even
degree distribution for mitigating over-aggregation, and utilizes multi-hop
dilation to establish more links for sparse topology enhancement. To resort to
the mDAG, we propose a Dilated Graph Convolution Network (DGCN), which
generates the optimal recovery trajectories with theoretically proven
convergence under massive damage cases. Simulation results show that the
proposed algorithm can guarantee the connectivity restoration under large swarm
and damage scales, while significantly expediting the recovery time by 75.94%
and improving the topology uniformity after recovery.

### 2. [Virtualizing RAN: Science, Strategy, and Architecture of Software-Defined Mobile Networks](http://arxiv.org/pdf/2506.09878v1)

Authors: Ryan Barker

Virtualising the Radio Access Network (RAN) is widely touted as the
corner-stone of affordable 5G and a prerequisite for AI-native 6G. Yet current
discourse often isolates spectrum policy, cloud engineering and organisational
readiness into silos. This paper delivers an integrated analysis that spans
science, technology, business strategy and culture. I first review
spectrum-auction economics and show-via a comparative study of T-Mobile US and
Verizon-that mid-band contiguity leveraged through software-defined carrier
aggregation outperforms mmWave-centric deployments in both coverage and churn
metrics. I then formalise the technical foundations of virtualised and open
RAN, deriving capacity limits from contiguous and dis-contiguous spectrum maths
and quantifying hardware ceilings for 400 MHz mmWave channels. Edge compute
platforms (NVIDIA EGX, Samsung vRAN 3.0) and SDN-controlled RAN Intelligent
Controllers are examined alongside AI ML pipelines that enable
digital-twin-driven optimisation. A security cost model extends recent O-RAN
measurements to show how 256-bit cipher enforcement adds 35-60 us latency
unless mitigated by inline crypto off-load. Finally, a national automation case
study of live vRAN sites -- demonstrates an 81 to 13 day cycle-time reduction
once cultural change errors are corrected. I conclude with open research
challenges for sub-THz 6G, energy-neutral AI accelerators and zero-trust
orchestration, offering actionable recommendations for operators, vendors and
researchers.

### 3. [Securing Open RAN: A Survey of Cryptographic Challenges and Emerging Solutions for 5G](http://arxiv.org/pdf/2506.09418v1)

Authors: Ryan Barker, Fatemeh Afghah

The advent of Open Radio Access Networks (O-RAN) introduces modularity and
flexibility into 5G deployments but also surfaces novel security challenges
across disaggregated interfaces. This literature review synthesizes recent
research across thirteen academic and industry sources, examining
vulnerabilities such as cipher bidding-down attacks, partial encryption
exposure on control/user planes, and performance trade-offs in securing O-RAN
interfaces like E2 and O1. The paper surveys key cryptographic tools -- SNOW-V,
AES-256, and ZUC-256 -- evaluating their throughput, side-channel resilience,
and adaptability to heterogeneous slices (eMBB, URLLC, mMTC). Emphasis is
placed on emerging testbeds and AI-driven controllers that facilitate dynamic
orchestration, anomaly detection, and secure configuration. We conclude by
outlining future research directions, including hardware offloading,
cross-layer cipher adaptation, and alignment with 3GPP TS 33.501 and O-RAN
Alliance security mandates, all of which point toward the need for integrated,
zero-trust architectures in 6G.

### 4. [Real-Time Network Traffic Forecasting with Missing Data: A Generative Model Approach](http://arxiv.org/pdf/2506.09647v1)

Authors: Lei Deng, Wenhan Xu, Jingwei Li, Danny H. K. Tsang

Real-time network traffic forecasting is crucial for network management and
early resource allocation. Existing network traffic forecasting approaches
operate under the assumption that the network traffic data is fully observed.
However, in practical scenarios, the collected data are often incomplete due to
various human and natural factors. In this paper, we propose a generative model
approach for real-time network traffic forecasting with missing data. Firstly,
we model the network traffic forecasting task as a tensor completion problem.
Secondly, we incorporate a pre-trained generative model to achieve the low-rank
structure commonly associated with tensor completion. The generative model
effectively captures the intrinsic low-rank structure of network traffic data
during pre-training and enables the mapping from a compact latent
representation to the tensor space. Thirdly, rather than directly optimizing
the high-dimensional tensor, we optimize its latent representation, which
simplifies the optimization process and enables real-time forecasting. We also
establish a theoretical recovery guarantee that quantifies the error bound of
the proposed approach. Experiments on real-world datasets demonstrate that our
approach achieves accurate network traffic forecasting within 100 ms, with a
mean absolute error (MAE) below 0.002, as validated on the Abilene dataset.

### 5. [SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving](http://arxiv.org/pdf/2506.09397v1)

Authors: Xiangchen Li, Dimitrios Spatharakis, Saeid Ghafouri, Jiakun Fan, Dimitrios Nikolopoulos

Regardless the advancements in device capabilities, efficient inferencing
advanced large language models (LLMs) at the edge remains challenging due to
limited device memory and power constraints. Existing strategies, such as
aggressive quantization, pruning, or remote inference, trade accuracy for
efficiency or lead to substantial cost burdens. This position paper introduces
a new approach that leverages speculative decoding, previously viewed primarily
as a decoding acceleration technique for autoregressive generation of LLMs, as
a promising approach specifically adapted for edge computing by orchestrating
computation across heterogeneous devices. We propose SLED, a method that allows
lightweight edge devices to draft multiple candidate tokens locally using
diverse draft models, while a single, shared edge server efficiently batches
and verifies the tokens utilizing a more precise target model. This approach
supports device heterogeneity and reduces server-side memory footprint by
avoiding the need to deploy multiple target models. Our initial experiments
with Jetson Orin Nano, Raspberry Pi 5, and an RTX 6000 edge server indicate
substantial benefits: significantly reduced latency, improved energy
efficiency, and increased concurrent inference sessions, all without
sacrificing model accuracy.

### Robotics

### 1. [Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation](http://arxiv.org/pdf/2506.09384v1)

Authors: Chendong Xin, Mingrui Yu, Yongpeng Jiang, Zhefeng Zhang, Xiang Li

Kinematic retargeting from human hands to robot hands is essential for
transferring dexterity from humans to robots in manipulation teleoperation and
imitation learning. However, due to mechanical differences between human and
robot hands, completely reproducing human motions on robot hands is impossible.
Existing works on retargeting incorporate various optimization objectives,
focusing on different aspects of hand configuration. However, the lack of
experimental comparative studies leaves the significance and effectiveness of
these objectives unclear. This work aims to analyze these retargeting
objectives for dexterous manipulation through extensive real-world comparative
experiments. Specifically, we propose a comprehensive retargeting objective
formulation that integrates intuitively crucial factors appearing in recent
approaches. The significance of each factor is evaluated through experimental
ablation studies on the full objective in kinematic posture retargeting and
real-world teleoperated manipulation tasks. Experimental results and
conclusions provide valuable insights for designing more accurate and effective
retargeting algorithms for real-world dexterous manipulation.

### 2. [Design of an innovative robotic surgical instrument for circular stapling](http://arxiv.org/pdf/2506.09444v1)

Authors: Paul Tucan, Nadim Al Hajjar, Calin Vaida, Alexandru Pusca, Tiberiu Antal, Corina Radu, Daniel Jucan, Adrian Pisla, Damien Chablat, Doina Pisla

Esophageal cancer remains a highly aggressive malignancy with low survival
rates, requiring advanced surgical interventions like esophagectomy.
Traditional manual techniques, including circular staplers, face challenges
such as limited precision, prolonged recovery times, and complications like
leaks and tissue misalignment. This paper presents a novel robotic circular
stapler designed to enhance the dexterity in confined spaces, improve tissue
alignment, and reduce post-operative risks. Integrated with a cognitive robot
that serves as a surgeon's assistant, the surgical stapler uses three actuators
to perform anvil motion, cutter/stapler motion and allows a 75-degree bending
of the cartridge (distal tip). Kinematic analysis is used to compute the
stapler tip's position, ensuring synchronization with a robotic system.

### 3. [Advances on Affordable Hardware Platforms for Human Demonstration Acquisition in Agricultural Applications](http://arxiv.org/pdf/2506.09494v1)

Authors: Alberto San-Miguel-Tello, Gennaro Scarati, Alejandro Hernández, Mario Cavero-Vidal, Aakash Maroti, Néstor García

This paper presents advances on the Universal Manipulation Interface (UMI), a
low-cost hand-held gripper for robot Learning from Demonstration (LfD), for
complex in-the-wild scenarios found in agricultural settings. The focus is on
improving the acquisition of suitable samples with minimal additional setup.
Firstly, idle times and user's cognitive load are reduced through the
extraction of individual samples from a continuous demonstration considering
task events. Secondly, reliability on the generation of task sample's
trajectories is increased through the combination on-board inertial
measurements and external visual marker localization usage using Extended
Kalman Filtering (EKF). Results are presented for a fruit harvesting task,
outperforming the default pipeline.

### 4. [Integrating Quantized LLMs into Robotics Systems as Edge AI to Leverage their Natural Language Processing Capabilities](http://arxiv.org/pdf/2506.09581v1)

Authors: Miguel Á. González-Santamarta, Francisco J. Rodríguez-Lera, David Sobrín-Hidalgo, Ángel Manuel Guerrero-Higueras, Vicente MatellÁn-Olivera

Large Language Models (LLMs) have experienced great advancements in the last
year resulting in an increase of these models in several fields to face natural
language tasks. The integration of these models in robotics can also help to
improve several aspects such as human-robot interaction, navigation, planning
and decision-making. Therefore, this paper introduces llama\_ros, a tool
designed to integrate quantized Large Language Models (LLMs) into robotic
systems using ROS 2. Leveraging llama.cpp, a highly optimized runtime engine,
llama\_ros enables the efficient execution of quantized LLMs as edge artificial
intelligence (AI) in robotics systems with resource-constrained environments,
addressing the challenges of computational efficiency and memory limitations.
By deploying quantized LLMs, llama\_ros empowers robots to leverage the natural
language understanding and generation for enhanced decision-making and
interaction which can be paired with prompt engineering, knowledge graphs,
ontologies or other tools to improve the capabilities of autonomous robots.
Additionally, this paper provides insights into some use cases of using
llama\_ros for planning and explainability in robotics.

### 5. [VAULT: A Mobile Mapping System for ROS 2-based Autonomous Robots](http://arxiv.org/pdf/2506.09583v1)

Authors: Miguel Á. González-Santamarta, Francisco J. Rodríguez-Lera, Vicente Matellán-Olivera

Localization plays a crucial role in the navigation capabilities of
autonomous robots, and while indoor environments can rely on wheel odometry and
2D LiDAR-based mapping, outdoor settings such as agriculture and forestry,
present unique challenges that necessitate real-time localization and
consistent mapping. Addressing this need, this paper introduces the VAULT
prototype, a ROS 2-based mobile mapping system (MMS) that combines various
sensors to enable robust outdoor and indoor localization. The proposed solution
harnesses the power of Global Navigation Satellite System (GNSS) data,
visual-inertial odometry (VIO), inertial measurement unit (IMU) data, and the
Extended Kalman Filter (EKF) to generate reliable 3D odometry. To further
enhance the localization accuracy, Visual SLAM (VSLAM) is employed, resulting
in the creation of a comprehensive 3D point cloud map. By leveraging these
sensor technologies and advanced algorithms, the prototype offers a
comprehensive solution for outdoor localization in autonomous mobile robots,
enabling them to navigate and map their surroundings with confidence and
precision.

### 6. [Attention-Based Map Encoding for Learning Generalized Legged Locomotion](http://arxiv.org/pdf/2506.09588v1)

Authors: Junzhe He, Chong Zhang, Fabian Jenelten, Ruben Grandia, Moritz BÄcher, Marco Hutter

Dynamic locomotion of legged robots is a critical yet challenging topic in
expanding the operational range of mobile robots. It requires precise planning
when possible footholds are sparse, robustness against uncertainties and
disturbances, and generalizability across diverse terrains. While traditional
model-based controllers excel at planning on complex terrains, they struggle
with real-world uncertainties. Learning-based controllers offer robustness to
such uncertainties but often lack precision on terrains with sparse steppable
areas. Hybrid methods achieve enhanced robustness on sparse terrains by
combining both methods but are computationally demanding and constrained by the
inherent limitations of model-based planners. To achieve generalized legged
locomotion on diverse terrains while preserving the robustness of
learning-based controllers, this paper proposes to learn an attention-based map
encoding conditioned on robot proprioception, which is trained as part of the
end-to-end controller using reinforcement learning. We show that the network
learns to focus on steppable areas for future footholds when the robot
dynamically navigates diverse and challenging terrains. We synthesize behaviors
that exhibit robustness against uncertainties while enabling precise and agile
traversal of sparse terrains. Additionally, our method offers a way to
interpret the topographical perception of a neural network. We have trained two
controllers for a 12-DoF quadrupedal robot and a 23-DoF humanoid robot
respectively and tested the resulting controllers in the real world under
various challenging indoor and outdoor scenarios, including ones unseen during
training.

### 7. [Analytic Task Scheduler: Recursive Least Squares Based Method for Continual Learning in Embodied Foundation Models](http://arxiv.org/pdf/2506.09623v1)

Authors: Lipei Xie, Yingxin Li, Huiping Zhuang

Embodied foundation models are crucial for Artificial Intelligence (AI)
interacting with the physical world by integrating multi-modal inputs, such as
proprioception, vision and language, to understand human intentions and
generate actions to control robots. While these models demonstrate strong
generalization and few-shot learning capabilities, they face significant
challenges in continually acquiring new skills without forgetting previously
learned skills, a problem known as catastrophic forgetting. To address this
issue, we propose the Analytic Task Scheduler (ATS), a novel framework for
continual learning in embodied foundation models. ATS consists of a
task-specific model library, where each model is fine-tuned independently on a
single task, and an analytic scheduler trained using recursive least squares
(RLS) to learn the mapping between language instructions and task-specific
models. This architecture enables accurate task recognition and dynamic model
selection while fundamentally avoiding parameter interference across tasks. The
scheduler updates its parameters incrementally using only statistics
(autocorrelation and cross-correlation matrices), enabling forgetting-resistant
learning without the need to revisit historical data. We validate ATS on a
real-world robot platform (RM65B), demonstrating superior resistance to
forgetting and strong adaptability to task variations. The results highlight
ATS as an effective, scalable, and deployable solution for continual learning
in embodied foundation models operating in complex, dynamic environments. Our
code will be available at
https://github.com/MIAA-Embodied-AI/AnalyticTaskScheduler

### 8. [R-CARLA: High-Fidelity Sensor Simulations with Interchangeable Dynamics for Autonomous Racing](http://arxiv.org/pdf/2506.09629v1)

Authors: Maurice Brunner, Edoardo Ghignone, Nicolas Baumann, Michele Magno

Autonomous racing has emerged as a crucial testbed for autonomous driving
algorithms, necessitating a simulation environment for both vehicle dynamics
and sensor behavior. Striking the right balance between vehicle dynamics and
sensor accuracy is crucial for pushing vehicles to their performance limits.
However, autonomous racing developers often face a trade-off between accurate
vehicle dynamics and high-fidelity sensor simulations. This paper introduces
R-CARLA, an enhancement of the CARLA simulator that supports holistic
full-stack testing, from perception to control, using a single system. By
seamlessly integrating accurate vehicle dynamics with sensor simulations,
opponents simulation as NPCs, and a pipeline for creating digital twins from
real-world robotic data, R-CARLA empowers researchers to push the boundaries of
autonomous racing development. Furthermore, it is developed using CARLA's rich
suite of sensor simulations. Our results indicate that incorporating the
proposed digital-twin framework into R-CARLA enables more realistic full-stack
testing, demonstrating a significant reduction in the Sim-to-Real gap of car
dynamics simulation by 42% and by 82% in the case of sensor simulation across
various testing scenarios.

### 9. [Human-robot collaborative transport personalization via Dynamic Movement Primitives and velocity scaling](http://arxiv.org/pdf/2506.09697v1)

Authors: Paolo Franceschi, Andrea Bussolan, Vincenzo Pomponi, Oliver Avram, Stefano Baraldo, Anna Valente

Nowadays, industries are showing a growing interest in human-robot
collaboration, particularly for shared tasks. This requires intelligent
strategies to plan a robot's motions, considering both task constraints and
human-specific factors such as height and movement preferences. This work
introduces a novel approach to generate personalized trajectories using Dynamic
Movement Primitives (DMPs), enhanced with real-time velocity scaling based on
human feedback. The method was rigorously tested in industrial-grade
experiments, focusing on the collaborative transport of an engine cowl lip
section. Comparative analysis between DMP-generated trajectories and a
state-of-the-art motion planner (BiTRRT) highlights their adaptability combined
with velocity scaling. Subjective user feedback further demonstrates a clear
preference for DMP- based interactions. Objective evaluations, including
physiological measurements from brain and skin activity, reinforce these
findings, showcasing the advantages of DMPs in enhancing human-robot
interaction and improving user experience.

### 10. [Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving](http://arxiv.org/pdf/2506.09800v1)

Authors: Haochen Liu, Tianyu Li, Haohan Yang, Li Chen, Caojun Wang, Ke Guo, Haochen Tian, Hongchen Li, Hongyang Li, Chen Lv

End-to-end autonomous driving has emerged as a promising paradigm for
directly mapping sensor inputs to planning maneuvers using learning-based
modular integrations. However, existing imitation learning (IL)-based models
suffer from generalization to hard cases, and a lack of corrective feedback
loop under post-deployment. While reinforcement learning (RL) offers a
potential solution to tackle hard cases with optimality, it is often hindered
by overfitting to specific driving cases, resulting in catastrophic forgetting
of generalizable knowledge and sample inefficiency. To overcome these
challenges, we propose Reinforced Refinement with Self-aware Expansion (R2SE),
a novel learning pipeline that constantly refines hard domain while keeping
generalizable driving policy for model-agnostic end-to-end driving systems.
Through reinforcement fine-tuning and policy expansion that facilitates
continuous improvement, R2SE features three key components: 1) Generalist
Pretraining with hard-case allocation trains a generalist imitation learning
(IL) driving system while dynamically identifying failure-prone cases for
targeted refinement; 2) Residual Reinforced Specialist Fine-tuning optimizes
residual corrections using reinforcement learning (RL) to improve performance
in hard case domain while preserving global driving knowledge; 3) Self-aware
Adapter Expansion dynamically integrates specialist policies back into the
generalist model, enhancing continuous performance improvement. Experimental
results in closed-loop simulation and real-world datasets demonstrate
improvements in generalization, safety, and long-horizon policy robustness over
state-of-the-art E2E systems, highlighting the effectiveness of reinforce
refinement for scalable autonomous driving.

### Software Engineering

### 1. [Assessing the Impact of Refactoring Energy-Inefficient Code Patterns on Software Sustainability: An Industry Case Study](http://arxiv.org/pdf/2506.09370v1)

Authors: Rohit Mehra, Priyavanshi Pathania, Vibhu Saujanya Sharma, Vikrant Kaulgud, Sanjay Podder, Adam P. Burden

Advances in technologies like artificial intelligence and metaverse have led
to a proliferation of software systems in business and everyday life. With this
widespread penetration, the carbon emissions of software are rapidly growing as
well, thereby negatively impacting the long-term sustainability of our
environment. Hence, optimizing software from a sustainability standpoint
becomes more crucial than ever. We believe that the adoption of automated tools
that can identify energy-inefficient patterns in the code and guide appropriate
refactoring can significantly assist in this optimization. In this extended
abstract, we present an industry case study that evaluates the sustainability
impact of refactoring energy-inefficient code patterns identified by automated
software sustainability assessment tools for a large application. Preliminary
results highlight a positive impact on the application's sustainability
post-refactoring, leading to a 29% decrease in per-user per-month energy
consumption.

### 2. [Automated Synthesis of Formally Verified Multi-Abstraction Function Summaries](http://arxiv.org/pdf/2506.09550v1)

Authors: Fanpeng Yang, Xu Ma, Shuling Wang, Xiong Xu, Qinxiang Cao, Naijun Zhan, Xiaofeng Li, Bin Gu

Function summaries, which characterize the behavior of code segments
(typically functions) through preconditions and postconditions, are essential
for understanding, reusing, and verifying software, particularly in
safety-critical domains like aerospace embedded systems. However, these
mission-critical legacy code serving as a valuable reused asset often lacks
formal specifications. It is challenging to automatically generate function
summaries for C programs, due to the existence of complex features such as
loops, nested function calls, pointer aliasing, and so on. Moreover, function
summaries should support multiple abstraction levels to meet diverse
requirements, e.g. precise summaries capturing full functionality for formal
verification and intuitive summaries for human understanding.
  To address these challenges, we first propose a novel framework that combines
symbolic execution, large language models (LLMs), and formal verification to
generate Relatively Strongest Postconditions (RSPs) and build function
summaries that fully capture program behavior. Our approach leverages VST-A's
symbolic execution to precisely track program execution paths and state
transitions, employs LLMs to infer loop invariants based on predefined
templates, and uses Frama-C to guarantee soundness of generated summaries in an
iterative refinement loop. Furthermore, from generated RSPs, we automatically
synthesize strongest non-redundant postconditions expressed within given domain
specific language. We compare our approach with existing work through extensive
experiments.

### 3. [ASTAGEN: Empirical Evaluation of Automated SATD Taxonomy Generation with LLMs](http://arxiv.org/pdf/2506.09601v1)

Authors: Sota Nakashima, Yuta Ishimoto, Masanari Kondo, Tao Xiao, Yasutaka Kamei

Technical debt refers to suboptimal code that degrades software quality. When
developers intentionally introduce such debt, it is called self-admitted
technical debt (SATD). Since SATD hinders maintenance, identifying its
categories is key to uncovering quality issues. Traditionally, constructing
such taxonomies requires manually inspecting SATD comments and surrounding
code, which is time-consuming, labor-intensive, and often inconsistent due to
annotator subjectivity. This study presents ASTAGEN, an initial step toward
automating SATD taxonomy generation using large language models (LLMs). Given a
comment and its surrounding code, ASTAGEN first generates a concise explanation
for each SATD comment, then incrementally generates and updates categories to
construct a taxonomy. We evaluate ASTAGEN on SATD datasets from three domains:
quantum software, smart contracts, and machine learning. It successfully
recovers domain-specific categories reported in prior work, such as Layer
Configuration in machine learning. Compared to a naive use of an LLM, ASTAGEN
produces more consistent category assignments due to its explanation-driven,
iterative design. It also completes taxonomy generation in under two hours and
for less than one USD, even on the largest dataset. These results suggest that
while full automation remains challenging, ASTAGEN is able to support
semi-automated taxonomy construction. Furthermore, our work opens up avenues
for future work, such as automatic taxonomy generation in other areas.

### 4. [Translating a VDM Model of a Medical Device into Kapture](http://arxiv.org/pdf/2506.09636v1)

Authors: Joe Hare, Leo Freitas, Ken Pierce

As the complexity of safety-critical medical devices increases, so does the
need for clear, verifiable, software requirements. This paper explores the use
of Kapture, a formal modelling tool developed by D-RisQ, to translate an
existing formal VDM model of a medical implant for treating focal epilepsy
called CANDO. The work was undertaken without prior experience in formal
methods. The paper assess Kapture's usability, the challenges of formal
modelling, and the effectiveness of the translated model. The result is a model
in Kapture which covers over 90% of the original VDM model, and produces
matching traces of results. While several issues were encountered during design
and implementation, mainly due to the initial learning curve, this paper
demonstrates that complex systems can be effectively modelled in Kapture by
inexperienced users and highlights some difficulties in translating VDM
specifications to Kapture.

### 5. [A First Look at Bugs in LLM Inference Engines](http://arxiv.org/pdf/2506.09713v1)

Authors: Mugeng Liu, Siqi Zhong, Weichen Bi, Yixuan Zhang, Zhiyang Chen, Zhenpeng Chen, Xuanzhe Liu, Yun Ma

Large language model-specific inference engines (in short as \emph{LLM
inference engines}) have become a fundamental component of modern AI
infrastructure, enabling the deployment of LLM-powered applications (LLM apps)
across cloud and local devices. Despite their critical role, LLM inference
engines are prone to bugs due to the immense resource demands of LLMs and the
complexities of cross-platform compatibility. However, a systematic
understanding of these bugs remains lacking. To bridge this gap, we present the
first empirical study on bugs in LLM inference engines. We mine official
repositories of 5 widely adopted LLM inference engines, constructing a
comprehensive dataset of 929 real-world bugs. Through a rigorous open coding
process, we analyze these bugs to uncover their symptoms, root causes, and
commonality. Our findings reveal six major bug symptoms and a taxonomy of 28
root causes, shedding light on the key challenges in bug detection and location
within LLM inference engines. Based on these insights, we propose a series of
actionable implications for researchers, inference engine vendors, and LLM app
developers.

### 6. [Towards Bridging Formal Methods and Human Interpretability](http://arxiv.org/pdf/2506.09759v1)

Authors: Abhijit Paul, Proma Chowdhury, Kazi Sakib

Labeled Transition Systems (LTS) are integral to model checking and design
repair tools. System engineers frequently examine LTS designs during model
checking or design repair to debug, identify inconsistencies, and validate
system behavior. Despite LTS's significance, no prior research has examined
human comprehension of these designs. To address this, we draw on traditional
software engineering and graph theory, identifying 7 key metrics: cyclomatic
complexity, state space size, average branching factor, maximum depth, Albin
complexity, modularity, and redundancy. We created a dataset of 148 LTS
designs, sampling 48 for 324 paired comparisons, and ranked them using the
Bradley-Terry model. Through Kendall's Tau correlation analysis, we found that
Albin complexity ($\tau = 0.444$), state space size ($\tau = 0.420$),
cyclomatic complexity ($\tau = 0.366$), and redundancy ($\tau = 0.315$) most
accurately reflect human comprehension of LTS designs. To showcase the metrics'
utility, we applied the Albin complexity metric within the Fortis design repair
tool, ranking system redesigns. This ranking reduced annotators' comprehension
time by 39\%, suggesting that metrics emphasizing human factors can enhance
formal design interpretability.

### 7. [variability.dev: Towards an Online Toolbox for Feature Modeling](http://arxiv.org/pdf/2506.09845v1)

Authors: Tobias Heß, Lukas Ostheimer, Tobias Betz, Simon Karrer, Tim Jannik Schmidt, Pierre Coquet, Sean Semmler, Thomas Thüm

The emergence of feature models as the default to model the variability in
configurable systems fosters a rich diversity in applications, application
domains, and perspectives. Independent of their domain, modelers require to
open, view, edit, transform, save, and configure models as well as to
collaborate with others. However, at the time of writing, the top five results
when googling ``Online Editor Feature Model'' point to editors that either have
minimal functionality, are unmaintained or defunct, or require an offline
installation, such as FeatureIDE. In this work we present a preview of our
in-development online toolbox for feature modeling, variability.dev. In
particular, we showcase our collaborative feature-model editor and our online
configurator both of which are built on top of the FeatureIDE library.

### 8. [Reasoning as a Resource: Optimizing Fast and Slow Thinking in Code Generation Models](http://arxiv.org/pdf/2506.09396v1)

Authors: Zongjie Li, Shuai Wang

This position paper proposes a fundamental shift in designing code generation
models: treating reasoning depth as a controllable resource. Rather than being
an incidental byproduct of prompting, we argue that the trade-off between
rapid, direct answers ("fast thinking") and elaborate, chain-of-thought
deliberation ("slow thinking") must be explicitly managed. We contend that
optimizing reasoning budgets across the entire model lifecycle - from synthetic
data creation and benchmarking to real-world deploymen - can unlock superior
trade-offs among accuracy, latency, and cost. This paper outlines how adaptive
control over reasoning can enrich supervision signals, motivate new
multi-dimensional benchmarks, and inform cost-aware, security-conscious
deployment policies. By viewing fast and slow thinking as complementary modes
to be scheduled, we envision coding agents that think deep when necessary and
act fast when possible.

### 9. [Calculating Software's Energy Use and Carbon Emissions: A Survey of the State of Art, Challenges, and the Way Ahead](http://arxiv.org/pdf/2506.09683v1)

Authors: Priyavanshi Pathania, Nikhil Bamby, Rohit Mehra, Samarth Sikand, Vibhu Saujanya Sharma, Vikrant Kaulgud, Sanjay Podder, Adam P. Burden

The proliferation of software and AI comes with a hidden risk: its growing
energy and carbon footprint. As concerns regarding environmental sustainability
come to the forefront, understanding and optimizing how software impacts the
environment becomes paramount. In this paper, we present a state-of-the-art
review of methods and tools that enable the measurement of software and
AI-related energy and/or carbon emissions. We introduce a taxonomy to
categorize the existing work as Monitoring, Estimation, or Black-Box
approaches. We delve deeper into the tools and compare them across different
dimensions and granularity - for example, whether their measurement encompasses
energy and carbon emissions and the components considered (like CPU, GPU, RAM,
etc.). We present our observations on the practical use (component wise
consolidation of approaches) as well as the challenges that we have identified
across the current state-of-the-art. As we start an initiative to address these
challenges, we emphasize active collaboration across the community in this
important field.

### 10. [Mapping NVD Records to Their VFCs: How Hard is it?](http://arxiv.org/pdf/2506.09702v1)

Authors: Huu Hung Nguyen, Duc Manh Tran, Yiran Cheng, Thanh Le-Cong, Hong Jin Kang, Ratnadira Widyasari, Shar Lwin Khin, Ouh Eng Lieh, Ting Zhang, David Lo

Mapping National Vulnerability Database (NVD) records to vulnerability-fixing
commits (VFCs) is crucial for vulnerability analysis but challenging due to
sparse explicit links in NVD references.This study explores this mapping's
feasibility through an empirical approach. Manual analysis of NVD references
showed Git references enable over 86% success, while non-Git references achieve
under 14%. Using these findings, we built an automated pipeline extracting
31,942 VFCs from 20,360 NVD records (8.7% of 235,341) with 87% precision,
mainly from Git references. To fill gaps, we mined six external security
databases, yielding 29,254 VFCs for 18,985 records (8.1%) at 88.4% precision,
and GitHub repositories, adding 3,686 VFCs for 2,795 records (1.2%) at 73%
precision. Combining these, we mapped 26,710 unique records (11.3% coverage)
from 7,634 projects, with overlap between NVD and external databases, plus
unique GitHub contributions. Despite success with Git references, 88.7% of
records remain unmapped, highlighting the difficulty without Git links. This
study offers insights for enhancing vulnerability datasets and guiding future
automated security research.

### Social and Information Networks

### 1. [ELRUHNA: Elimination Rule-basedHypergraph Alignment](http://arxiv.org/pdf/2506.09866v1)

Authors: Cameron Ibrahim, S M Ferdous, Ilya Safro, Marco Minutoli, Mahantesh Halappanavar

Hypergraph alignment is a well-known NP-hard problem with numerous practical
applications across domains such as bioinformatics, social network analysis,
and computer vision. Despite its computational complexity, practical and
scalable solutions are urgently needed to enable pattern discovery and entity
correspondence in high-order relational data. The problem remains understudied
in contrast to its graph based counterpart. In this paper, we propose ELRUHNA,
an elimination rule-based framework for unsupervised hypergraph alignment that
operates on the bipartite representation of hypergraphs. We introduce the
incidence alignment formulation, a binary quadratic optimization approach that
jointly aligns vertices and hyperedges. ELRUHNA employs a novel similarity
propagation scheme using local matching and cooling rules, supported by an
initialization strategy based on generalized eigenvector centrality for
incidence matrices. Through extensive experiments on real-world datasets, we
demonstrate that ELRUHNA achieves higher alignment accuracy compared to
state-of-the-art algorithms, while scaling effectively to large hypergraphs.

### 2. [Alice and the Caterpillar: A more descriptive null model for assessing data mining results](http://arxiv.org/pdf/2506.09764v1)

Authors: Giulia Preti, Gianmarco De Francisci Morales, Matteo Riondato

We introduce novel null models for assessing the results obtained from
observed binary transactional and sequence datasets, using statistical
hypothesis testing. Our null models maintain more properties of the observed
dataset than existing ones. Specifically, they preserve the Bipartite Joint
Degree Matrix of the bipartite (multi-)graph corresponding to the dataset,
which ensures that the number of caterpillars, i.e., paths of length three, is
preserved, in addition to other properties considered by other models. We
describe Alice, a suite of Markov chain Monte Carlo algorithms for sampling
datasets from our null models, based on a carefully defined set of states and
efficient operations to move between them. The results of our experimental
evaluation show that Alice mixes fast and scales well, and that our null model
finds different significant results than ones previously considered in the
literature.

### 3. [KI4Demokratie: An AI-Based Platform for Monitoring and Fostering Democratic Discourse](http://arxiv.org/pdf/2506.09947v1)

Authors: Rudy Alexandro Garrido Veliz, Till Nikolaus Schaland, Simon Bergmoser, Florian Horwege, Somya Bansal, Ritesh Nahar, Martin Semmann, Jörg Forthmann, Seid Muhie Yimam

Social media increasingly fuel extremism, especially right-wing extremism,
and enable the rapid spread of antidemocratic narratives. Although AI and data
science are often leveraged to manipulate political opinion, there is a
critical need for tools that support effective monitoring without infringing on
freedom of expression. We present KI4Demokratie, an AI-based platform that
assists journalists, researchers, and policymakers in monitoring right-wing
discourse that may undermine democratic values. KI4Demokratie applies machine
learning models to a large-scale German online data gathered on a daily basis,
providing a comprehensive view of trends in the German digital sphere. Early
analysis reveals both the complexity of tracking organized extremist behavior
and the promise of our integrated approach, especially during key events.

### 4. [Don't be Afraid of Cell Complexes! An Introduction from an Applied Perspective](http://arxiv.org/pdf/2506.09726v1)

Authors: Josef Hoppe, Vincent P. Grande, Michael T. Schaub

Cell complexes (CCs) are a higher-order network model deeply rooted in
algebraic topology that has gained interest in signal processing and network
science recently. However, while the processing of signals supported on CCs can
be described in terms of easily-accessible algebraic or combinatorial notions,
the commonly presented definition of CCs is grounded in abstract concepts from
topology and remains disconnected from the signal processing methods developed
for CCs. In this paper, we aim to bridge this gap by providing a simplified
definition of CCs that is accessible to a wider audience and can be used in
practical applications. Specifically, we first introduce a simplified notion of
abstract regular cell complexes (ARCCs). These ARCCs only rely on notions from
algebra and can be shown to be equivalent to regular cell complexes for most
practical applications. Second, using this new definition we provide an
accessible introduction to (abstract) cell complexes from a perspective of
network science and signal processing. Furthermore, as many practical
applications work with CCs of dimension 2 and below, we provide an even simpler
definition for this case that significantly simplifies understanding and
working with CCs in practice.

### Systems and Control

### 1. [Integer-Clustering Optimization of Hydrogen and Battery EV Fleets Considering DERs](http://arxiv.org/pdf/2506.09388v1)

Authors: Sijia Geng, Thomas Lee, Dharik Mallapragada, Audun Botterud

Electrified transportation leads to a tighter integration between
transportation and energy distribution systems. In this work, we develop
scalable optimization models to co-design hydrogen and battery electric vehicle
(EV) fleets, distributed energy resources, and fast-charging and
hydrogen-fueling infrastructure to efficiently meet transportation demands. A
novel integer-clustering formulation is used for optimizing fleet-level EV
operation while maintaining accurate individual vehicle dispatch, which
significantly improves the computation efficiency with guaranteed performance.
We apply the optimization model to Boston's public transit bus network using
real geospatial data and cost parameters. Realistic insights are provided into
the future evolution of coupled electricity-transportation-hydrogen systems,
including the effects of electricity price structure, hydrogen fuel cost,
carbon emission constraint, temperature effects on EV range, and distribution
system upgrade cost.

### 2. [Voltage-Controlled Oscillator and Memristor-Based Analog Computing for Solving Systems of Linear Equations](http://arxiv.org/pdf/2506.09392v1)

Authors: Hao Li, Rizwan S. Peerla, Frank Barrows, Francesco Caravelli, Bibhu Datta Sahoo

Matrix computations have become increasingly significant in many data-driven
applications. However, Moores law for digital computers has been gradually
approaching its limit in recent years. Moreover, digital computers encounter
substantial complexity when performing matrix computations and need a long time
to finish the computations, and existing analog matrix computation schemes
require a large chip area and power consumption. This paper proposes a linear
algebra system of equations based on integrators, which features low power
consumption, compact area, and fast computation time. Due to the simple
structure of the ring oscillator, the ring oscillator-based integrator exhibits
a compact area and low power consumption. Therefore, ring oscillator-based
integrators are introduced into the linear algebra system of equations, and
this system can be used to compute the linear algebra equations of the matrix
with either positive or negative values. This paper provides a detailed
analysis and verification of the proposed circuit structure. Compared to
similar circuits, this work has significant advantages in terms of area, power
consumption, and computation speed.

### 3. [Large-scale LH2 pipeline infrastructure concept for airports](http://arxiv.org/pdf/2506.09410v1)

Authors: H. A. Krog, Y. Jooss, H. Fyhn, P. Nekså, I. Hjorth

Infrastructure and processes for handling of liquid hydrogen (LH2) is needed
to enable large-scale decarbonization of aviation with hydrogen aircraft. At
large airports, pipeline and hydrant systems will be important for a mature
hydrogen-powered air travel market. As the vaporization of LH2 is a challenge
in fuel handling, the pipeline infrastructure must be designed and operated
such that the fuel is subcooled. Through modelling and simulation of aircraft
tanks refuelling by a pipeline infrastructure concept, it is found that
continuous recycling of LH2 within the system is needed to maintain subcooling,
and the pump operation is important for preventing flashing. With the proposed
concept, some hydrogen vapor is formed in the aircraft tank, but the vapor can
be utilised by hydrogen-powered ground support equipment.

### 4. [Optimization and Control Technologies for Renewable-Dominated Hydrogen-Blended Integrated Gas-Electricity System: A Review](http://arxiv.org/pdf/2506.09447v1)

Authors: Wenxin Liu, Jiakun Fang, Shichang Cui, Iskandar Abdullaev, Suyang Zhou, Xiaomeng Ai, Jinyu Wen

The growing coupling among electricity, gas, and hydrogen systems is driven
by green hydrogen blending into existing natural gas pipelines, paving the way
toward a renewable-dominated energy future. However, the integration poses
significant challenges, particularly ensuring efficient and safe operation
under varying hydrogen penetration and infrastructure adaptability. This paper
reviews progress in optimization and control technologies for hydrogen-blended
integrated gas-electricity system. First, key technologies and international
demonstration projects are introduced to provide an overview of current
developments. Besides, advances in gas-electricity system integration,
including modeling, scheduling, planning and market design, are reviewed
respectively. Then, the potential for cross-system fault propagation is
highlighted, and practical methods for safety analysis and control are
proposed. Finally, several possible research directions are introduced, aiming
to ensure efficient renewable integration and reliable operation.

### 5. [Bridging Continuous-time LQR and Reinforcement Learning via Gradient Flow of the Bellman Error](http://arxiv.org/pdf/2506.09685v1)

Authors: Armin Gießler, Albertus Johannes Malan, Sören Hohmann

In this paper, we present a novel method for computing the optimal feedback
gain of the infinite-horizon Linear Quadratic Regulator (LQR) problem via an
ordinary differential equation. We introduce a novel continuous-time Bellman
error, derived from the Hamilton-Jacobi-Bellman (HJB) equation, which
quantifies the suboptimality of stabilizing policies and is parametrized in
terms of the feedback gain. We analyze its properties, including its effective
domain, smoothness, coerciveness and show the existence of a unique stationary
point within the stability region. Furthermore, we derive a closed-form
gradient expression of the Bellman error that induces a gradient flow. This
converges to the optimal feedback and generates a unique trajectory which
exclusively comprises stabilizing feedback policies. Additionally, this work
advances interesting connections between LQR theory and Reinforcement Learning
(RL) by redefining suboptimality of the Algebraic Riccati Equation (ARE) as a
Bellman error, adapting a state-independent formulation, and leveraging
Lyapunov equations to overcome the infinite-horizon challenge. We validate our
method in a simulation and compare it to the state of the art.

### 6. [Bipedal Balance Control with Whole-body Musculoskeletal Standing and Falling Simulations](http://arxiv.org/pdf/2506.09383v1)

Authors: Chengtian Ma, Yunyue Wei, Chenhui Zuo, Chen Zhang, Yanan Sui

Balance control is important for human and bipedal robotic systems. While
dynamic balance during locomotion has received considerable attention,
quantitative understanding of static balance and falling remains limited. This
work presents a hierarchical control pipeline for simulating human balance via
a comprehensive whole-body musculoskeletal system. We identified spatiotemporal
dynamics of balancing during stable standing, revealed the impact of muscle
injury on balancing behavior, and generated fall contact patterns that aligned
with clinical data. Furthermore, our simulated hip exoskeleton assistance
demonstrated improvement in balance maintenance and reduced muscle effort under
perturbation. This work offers unique muscle-level insights into human balance
dynamics that are challenging to capture experimentally. It could provide a
foundation for developing targeted interventions for individuals with balance
impairments and support the advancement of humanoid robotic systems.

### 7. [A Survey on the Role of Artificial Intelligence and Machine Learning in 6G-V2X Applications](http://arxiv.org/pdf/2506.09512v1)

Authors: Donglin Wang, Anjie Qiu, Qiuheng Zhou, Hans D. Schotten

The rapid advancement of Vehicle-to-Everything (V2X) communication is
transforming Intelligent Transportation Systems (ITS), with 6G networks
expected to provide ultra-reliable, low-latency, and high-capacity connectivity
for Connected and Autonomous Vehicles (CAVs). Artificial Intelligence (AI) and
Machine Learning (ML) have emerged as key enablers in optimizing V2X
communication by enhancing network management, predictive analytics, security,
and cooperative driving due to their outstanding performance across various
domains, such as natural language processing and computer vision. This survey
comprehensively reviews recent advances in AI and ML models applied to 6G-V2X
communication. It focuses on state-of-the-art techniques, including Deep
Learning (DL), Reinforcement Learning (RL), Generative Learning (GL), and
Federated Learning (FL), with particular emphasis on developments from the past
two years. Notably, AI, especially GL, has shown remarkable progress and
emerging potential in enhancing the performance, adaptability, and intelligence
of 6G-V2X systems. Despite these advances, a systematic summary of recent
research efforts in this area remains lacking, which this survey aims to
address. We analyze their roles in 6G-V2X applications, such as intelligent
resource allocation, beamforming, intelligent traffic management, and security
management. Furthermore, we explore the technical challenges, including
computational complexity, data privacy, and real-time decision-making
constraints, while identifying future research directions for AI-driven 6G-V2X
development. This study aims to provide valuable insights for researchers,
engineers, and policymakers working towards realizing intelligent, AI-powered
V2X ecosystems in 6G communication.

### 8. [Adaptive event-triggered robust tracking control of soft robots](http://arxiv.org/pdf/2506.09523v1)

Authors: Renjie Ma, Ziyao Qu, Zhijian Hu, Dong Zhao, Marios M. Polycarpou

Soft robots manufactured with flexible materials can be highly compliant and
adaptive to their surroundings, which facilitates their application in areas
such as dexterous manipulation and environmental exploration. This paper aims
at investigating the tracking control problem for soft robots under uncertainty
such as unmodeled dynamics and external disturbance. First, we establish a
novel switching function and design the compensated tracking error dynamics by
virtue of the command filter. Then, based on the backstepping methodology, the
virtual controllers and the adaptive logic estimating the supremum of
uncertainty impacts are developed for synthesizing an event-triggered control
strategy. In addition, the uniformed finite-time stability certification is
derived for different scenarios of the switching function. Finally, we perform
a case study of a soft robot to illustrate the effectiveness of the proposed
control algorithm.

### 9. [Probability-One Optimization of Generalized Rayleigh Quotient Sum For Multi-Source Generalized Total Least-Squares](http://arxiv.org/pdf/2506.09573v1)

Authors: Dominik Friml, Pavel Václavek

This paper addresses the global optimization of the sum of the Rayleigh
quotient and the generalized Rayleigh quotient on the unit sphere. While
various methods have been proposed for this problem, they do not guarantee
convergence to the global maximizer. To overcome this limitation, we introduce
a probability-one homotopy optimization method that, under certain conditions,
guarantees convergence to the global maximizer. The proposed method is analyzed
alongside state-of-the-art approaches through numerical experiments, evaluating
their performance in terms of convergence speed and ability to reach the global
maximizer. Furthermore, we demonstrate how this ties in with the multi-source
Bayesian Generalized Total Least-Squares (B-GTLS) problem, illustrating its
applicability.

### 10. [Vulnerability-Based Optimal Grid Defense Strategies for Enhancing Cyber-Physical Energy System Resilience](http://arxiv.org/pdf/2506.09766v1)

Authors: Eric Tönges, Philipp Härtel, Martin Braun

An approach is proposed to identify optimal asset protection strategies based
on vulnerability assessment outcomes. Traditional bilevel attacker-defender
models emphasize worstcase scenarios but offer limited defensive guidance. In
contrast, trilevel models introduce high computational complexity and rely on
fixed network configurations. The proposed critical-components method leverages
vulnerability assessment results to determine protection strategies,
effectively outsourcing the upper-level defense decision. This enables
adaptability to diverse network topologies, assessment techniques, and
cyber-physical energy systems without the overhead of multi-level optimization.
Case studies demonstrate the potential for improved system resilience across
varying operational conditions.

### Machine Learning (Statistics Category)

### 1. [Attention-Bayesian Hybrid Approach to Modular Multiple Particle Tracking](http://arxiv.org/pdf/2506.09441v1)

Authors: Piyush Mishra, Philippe Roudot

Tracking multiple particles in noisy and cluttered scenes remains challenging
due to a combinatorial explosion of trajectory hypotheses, which scales
super-exponentially with the number of particles and frames. The transformer
architecture has shown a significant improvement in robustness against this
high combinatorial load. However, its performance still falls short of the
conventional Bayesian filtering approaches in scenarios presenting a reduced
set of trajectory hypothesis. This suggests that while transformers excel at
narrowing down possible associations, they may not be able to reach the
optimality of the Bayesian approach in locally sparse scenario. Hence, we
introduce a hybrid tracking framework that combines the ability of
self-attention to learn the underlying representation of particle behavior with
the reliability and interpretability of Bayesian filtering. We perform
trajectory-to-detection association by solving a label prediction problem,
using a transformer encoder to infer soft associations between detections
across frames. This prunes the hypothesis set, enabling efficient
multiple-particle tracking in Bayesian filtering framework. Our approach
demonstrates improved tracking accuracy and robustness against spurious
detections, offering a solution for high clutter multiple particle tracking
scenarios.

### 2. [Safe Screening Rules for Group SLOPE](http://arxiv.org/pdf/2506.09451v1)

Authors: Runxue Bao, Quanchao Lu, Yanfu Zhang

Variable selection is a challenging problem in high-dimensional sparse
learning, especially when group structures exist. Group SLOPE performs well for
the adaptive selection of groups of predictors. However, the block
non-separable group effects in Group SLOPE make existing methods either invalid
or inefficient. Consequently, Group SLOPE tends to incur significant
computational costs and memory usage in practical high-dimensional scenarios.
To overcome this issue, we introduce a safe screening rule tailored for the
Group SLOPE model, which efficiently identifies inactive groups with zero
coefficients by addressing the block non-separable group effects. By excluding
these inactive groups during training, we achieve considerable gains in
computational efficiency and memory usage. Importantly, the proposed screening
rule can be seamlessly integrated into existing solvers for both batch and
stochastic algorithms. Theoretically, we establish that our screening rule can
be safely employed with existing optimization algorithms, ensuring the same
results as the original approaches. Experimental results confirm that our
method effectively detects inactive feature groups and significantly boosts
computational efficiency without compromising accuracy.

### 3. [Evasion Attacks Against Bayesian Predictive Models](http://arxiv.org/pdf/2506.09640v1)

Authors: Pablo G. Arce, Roi Naveiro, David Ríos Insua

There is an increasing interest in analyzing the behavior of machine learning
systems against adversarial attacks. However, most of the research in
adversarial machine learning has focused on studying weaknesses against evasion
or poisoning attacks to predictive models in classical setups, with the
susceptibility of Bayesian predictive models to attacks remaining
underexplored. This paper introduces a general methodology for designing
optimal evasion attacks against such models. We investigate two adversarial
objectives: perturbing specific point predictions and altering the entire
posterior predictive distribution. For both scenarios, we propose novel
gradient-based attacks and study their implementation and properties in various
computational setups.

### 4. [Scaling Laws for Uncertainty in Deep Learning](http://arxiv.org/pdf/2506.09648v1)

Authors: Mattia Rosso, Simone Rossi, Giulio Franzese, Markus Heinonen, Maurizio Filippone

Deep learning has recently revealed the existence of scaling laws,
demonstrating that model performance follows predictable trends based on
dataset and model sizes. Inspired by these findings and fascinating phenomena
emerging in the over-parameterized regime, we examine a parallel direction: do
similar scaling laws govern predictive uncertainties in deep learning? In
identifiable parametric models, such scaling laws can be derived in a
straightforward manner by treating model parameters in a Bayesian way. In this
case, for example, we obtain $O(1/N)$ contraction rates for epistemic
uncertainty with respect to the number of data $N$. However, in
over-parameterized models, these guarantees do not hold, leading to largely
unexplored behaviors. In this work, we empirically show the existence of
scaling laws associated with various measures of predictive uncertainty with
respect to dataset and model sizes. Through experiments on vision and language
tasks, we observe such scaling laws for in- and out-of-distribution predictive
uncertainty estimated through popular approximate Bayesian inference and
ensemble methods. Besides the elegance of scaling laws and the practical
utility of extrapolating uncertainties to larger data or models, this work
provides strong evidence to dispel recurring skepticism against Bayesian
approaches: "In many applications of deep learning we have so much data
available: what do we need Bayes for?". Our findings show that "so much data"
is typically not enough to make epistemic uncertainty negligible.

### 5. [Knockoffs Inference under Privacy Constraints](http://arxiv.org/pdf/2506.09690v1)

Authors: Zhanrui Cai, Yingying Fan, Lan Gao

Model-X knockoff framework offers a model-free variable selection method that
ensures finite sample false discovery rate (FDR) control. However, the
complexity of generating knockoff variables, coupled with the model-free
assumption, presents significant challenges for protecting data privacy in this
context. In this paper, we propose a comprehensive framework for knockoff
inference within the differential privacy paradigm. Our proposed method
guarantees robust privacy protection while preserving the exact FDR control
entailed by the original model-X knockoff procedure. We further conduct power
analysis and establish sufficient conditions under which the noise added for
privacy preservation does not asymptotically compromise power. Through various
applications, we demonstrate that the differential privacy knockoff
(DP-knockoff) method can be effectively utilized to safeguard privacy during
variable selection with FDR control in both low and high dimensional settings.

### 6. [On the Similarities of Embeddings in Contrastive Learning](http://arxiv.org/pdf/2506.09781v1)

Authors: Chungpa Lee, Sehee Lim, Kibok Lee, Jy-yong Sohn

Contrastive learning (CL) operates on a simple yet effective principle:
embeddings of positive pairs are pulled together, while those of negative pairs
are pushed apart. Although various forms of contrastive loss have been proposed
and analyzed from different perspectives, prior works lack a comprehensive
framework that systematically explains a broad class of these objectives. In
this paper, we present a unified framework for understanding CL, which is based
on analyzing the cosine similarity between embeddings of positive and negative
pairs. In full-batch settings, we show that perfect alignment of positive pairs
is unattainable when similarities of negative pairs fall below a certain
threshold, and that this misalignment can be alleviated by incorporating
within-view negative pairs. In mini-batch settings, we demonstrate that smaller
batch sizes incur stronger separation among negative pairs within batches,
which leads to higher variance in similarities of negative pairs. To address
this limitation of mini-batch CL, we introduce an auxiliary loss term that
reduces the variance of similarities of negative pairs in CL. Empirical results
demonstrate that incorporating the proposed loss consistently improves the
performance of CL methods in small-batch training.

### 7. [A Deep Generative Model for the Simulation of Discrete Karst Networks](http://arxiv.org/pdf/2506.09832v1)

Authors: Dany Lauzon, Julien Straubhaar, Philippe Renard

The simulation of discrete karst networks presents a significant challenge
due to the complexity of the physicochemical processes occurring within various
geological and hydrogeological contexts over extended periods. This complex
interplay leads to a wide variety of karst network patterns, each intricately
linked to specific hydrogeological conditions. We explore a novel approach that
represents karst networks as graphs and applies graph generative models (deep
learning techniques) to capture the intricate nature of karst environments. In
this representation, nodes retain spatial information and properties, while
edges signify connections between nodes. Our generative process consists of two
main steps. First, we utilize graph recurrent neural networks (GraphRNN) to
learn the topological distribution of karst networks. GraphRNN decomposes the
graph simulation into a sequential generation of nodes and edges, informed by
previously generated structures. Second, we employ denoising diffusion
probabilistic models on graphs (G-DDPM) to learn node features (spatial
coordinates and other properties). G-DDPMs enable the generation of nodes
features on the graphs produced by the GraphRNN that adhere to the learned
statistical properties by sampling from the derived probability distribution,
ensuring that the generated graphs are realistic and capture the essential
features of the original data. We test our approach using real-world karst
networks and compare generated subgraphs with actual subgraphs from the
database, by using geometry and topology metrics. Our methodology allows
stochastic simulation of discrete karst networks across various types of
formations, a useful tool for studying the behavior of physical processes such
as flow and transport.

### 8. [Bayesian Probabilistic Matrix Factorization](http://arxiv.org/pdf/2506.09928v1)

Authors: Ruixuan Xu, Xiangxiang Weng

Matrix factorization is a widely used technique in recommendation systems.
Probabilistic Matrix Factorization (PMF) [1] extends traditional matrix
factorization by incorporating probability distributions over latent factors,
allowing for uncertainty quantification. However, computing the posterior
distribution is intractable due to the high-dimensional integral. To address
this, we employ two Bayesian inference methods: Markov Chain Monte Carlo (MCMC)
[2] and Variational Inference (VI) [3] to approximate the posterior. We
evaluate their performance on MovieLens dataset and compare their convergence
speed, predictive accuracy, and computational efficiency. Experimental results
demonstrate that VI offers faster convergence, while MCMC provides more
accurate posterior estimates.

### 9. [Know What You Don't Know: Uncertainty Calibration of Process Reward Models](http://arxiv.org/pdf/2506.09338v1)

Authors: Young-Jin Park, Kristjan Greenewald, Kaveh Alim, Hao Wang, Navid Azizan

Process reward models (PRMs) play a central role in guiding inference-time
scaling algorithms for large language models (LLMs). However, we observe that
even state-of-the-art PRMs can be poorly calibrated and often overestimate
success probabilities. To address this, we present a calibration approach,
performed via quantile regression, that adjusts PRM outputs to better align
with true success probabilities. Leveraging these calibrated success estimates
and their associated confidence bounds, we introduce an \emph{instance-adaptive
scaling} (IAS) framework that dynamically adjusts the inference budget based on
the estimated likelihood that a partial reasoning trajectory will yield a
correct final answer. Unlike conventional methods that allocate a fixed number
of reasoning trajectories per query, this approach successfully adapts to each
instance and reasoning step when using our calibrated PRMs. Experiments on
mathematical reasoning benchmarks show that (i) our PRM calibration method
successfully achieves small calibration error, outperforming the baseline
methods, (ii) calibration is crucial for enabling effective adaptive scaling,
and (iii) the proposed IAS strategy reduces inference costs while maintaining
final answer accuracy, utilizing less compute on more confident problems as
desired.

### 10. [Adversarial Surrogate Risk Bounds for Binary Classification](http://arxiv.org/pdf/2506.09348v1)

Authors: Natalie S. Frank

A central concern in classification is the vulnerability of machine learning
models to adversarial attacks. Adversarial training is one of the most popular
techniques for training robust classifiers, which involves minimizing an
adversarial surrogate risk. Recent work characterized when a minimizing
sequence of an adversarial surrogate risk is also a minimizing sequence of the
adversarial classification risk for binary classification -- a property known
as adversarial consistency. However, these results do not address the rate at
which the adversarial classification risk converges to its optimal value for
such a sequence of functions that minimize the adversarial surrogate. This
paper provides surrogate risk bounds that quantify that convergence rate.
Additionally, we derive distribution-dependent surrogate risk bounds in the
standard (non-adversarial) learning setting, that may be of independent
interest.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-06-12 PST.

### 1. [Consider the energy consumption of your quantum circuits](https://www.nature.com/articles/s42254-025-00846-0)

Authors: Coral Calero et al.

### 2. [How you breathe is like a fingerprint that can identify you](https://www.nature.com/articles/d41586-025-01835-0)

Authors: Humberto Basilio

