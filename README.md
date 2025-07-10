# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-09 17:00:25.303329 PST.

### Artificial Intelligence

### 1. [SingLoRA: Low Rank Adaptation Using a Single Matrix](http://arxiv.org/pdf/2507.05566v1)

Authors: David Bensaïd, Noam Rotstein, Roy Velich, Daniel Bensaïd, Ron Kimmel

Low-Rank Adaptation (LoRA) has significantly advanced parameter-efficient
fine-tuning of large pretrained models. LoRA augments the pre-trained weights
of a model by adding the product of two smaller matrices that together form a
low-rank matrix update. Recent research has shown that scale disparities
between these two matrices often cause unstable training dynamics, leading to
suboptimal performance. In this paper, we propose SingLoRA, which reformulates
low-rank adaptation by learning the weights update as a decomposition of a
single low-rank matrix multiplied by its transpose. This simple design
inherently removes inter-matrix scale conflicts, ensuring stable optimization,
and roughly halves the parameter count. We analyze SingLoRA within the
infinite-width neural network framework, showing that it guarantees stable
feature learning by construction. Extensive experiments on multiple tasks
validate these benefits. In common sense reasoning, fine-tuning LLama 7B on
MNLI with SingLoRA achieves 91.3% accuracy - surpassing LoRA (89.1%) and LoRA+
(90.2%) - while using only 60% of their parameter budget. In image generation,
fine-tuning Stable Diffusion with SingLoRA significantly improves image
fidelity on DreamBooth, achieving a DINO similarity score of 0.151, compared to
scores of 0.148 and 0.143 for DoRA and LoRA, respectively.

### 2. [Towards Measurement Theory for Artificial Intelligence](http://arxiv.org/pdf/2507.05587v1)

Authors: Elija Perrier

We motivate and outline a programme for a formal theory of measurement of
artificial intelligence. We argue that formalising measurement for AI will
allow researchers, practitioners, and regulators to: (i) make comparisons
between systems and the evaluation methods applied to them; (ii) connect
frontier AI evaluations with established quantitative risk analysis techniques
drawn from engineering and safety science; and (iii) foreground how what counts
as AI capability is contingent upon the measurement operations and scales we
elect to use. We sketch a layered measurement stack, distinguish direct from
indirect observables, and signpost how these ingredients provide a pathway
toward a unified, calibratable taxonomy of AI phenomena.

### 3. [MLlm-DR: Towards Explainable Depression Recognition with MultiModal Large Language Models](http://arxiv.org/pdf/2507.05591v1)

Authors: Wei Zhang, Juan Chen, En Zhu, Wenhong Cheng, YunPeng Li, Yanbo J. Wang

Automated depression diagnosis aims to analyze multimodal information from
interview videos to predict participants' depression scores. Previous studies
often lack clear explanations of how these scores were determined, limiting
their adoption in clinical practice. While the advent of LLMs provides a
possible pathway for explainable depression diagnosis, current LLMs capable of
processing multimodal data lack training on interview data, resulting in poor
diagnostic performance when used directly. In this paper, we propose a novel
multimodal large language model (MLlm-DR) that can understand multimodal
information inputs and supports explainable depression diagnosis. MLlm-DR
integrates a smaller LLMs and a lightweight query module (LQ-former).
Specifically, the smaller LLMs is designed to generate depression scores and
corresponding evaluation rationales. To enhance its logical reasoning for
domain-specific tasks while maintaining practicality, we constructed a robust
training dataset to fine-tune it. Meanwhile, the LQ-former captures
depression-related features from speech and visual data, aiding the model's
ability to process multimodal information, to achieve comprehensive depression
diagnosis. Our approach achieves state-of-the-art results on two
interview-based benchmark datasets, CMDC and E-DAIC-WOZ, demonstrating its
effectiveness and superiority.

### 4. [Domain adaptation of large language models for geotechnical applications](http://arxiv.org/pdf/2507.05613v1)

Authors: Lei Fan, Fangxue Liu, Cheng Chen

Recent developments in large language models (LLMs) are opening up new
opportunities in geotechnical engineering and engineering geology. While
general-purpose LLMs possess broad capabilities, effective application in
geotechnics often requires domain-specific adaptation. Such tailored LLMs are
increasingly employed to streamline geotechnical workflows. This paper presents
the first survey of the adaptation and application of LLMs in geotechnical
engineering. It outlines key methodologies for adaptation to geotechnical
domain, including prompt engineering, retrieval-augmented generation,
domain-adaptive pretraining, and fine-tuning. The survey examines the
state-of-the-art applications of geotechnical-adapted LLMs, including
geological interpretation, subsurface characterization, site planning, design
calculations, numerical modeling, safety and risk assessment, and educational
tutoring. It also analyzes benefits and limitations of geotechnical-adapted
LLMs, and identifies promising directions for future research in this
interdisciplinary discipline. The findings serve as a valuable resource for
practitioners seeking to integrate LLMs into geotechnical practice, while also
providing a foundation to stimulate further investigation within the academic
community.

### 5. [ADMC: Attention-based Diffusion Model for Missing Modalities Feature Completion](http://arxiv.org/pdf/2507.05624v1)

Authors: Wei Zhang, Juan Chen, Yanbo J. Wang, En Zhu, Xuan Yang, Yiduo Wang

Multimodal emotion and intent recognition is essential for automated
human-computer interaction, It aims to analyze users' speech, text, and visual
information to predict their emotions or intent. One of the significant
challenges is that missing modalities due to sensor malfunctions or incomplete
data. Traditional methods that attempt to reconstruct missing information often
suffer from over-coupling and imprecise generation processes, leading to
suboptimal outcomes. To address these issues, we introduce an Attention-based
Diffusion model for Missing Modalities feature Completion (ADMC). Our framework
independently trains feature extraction networks for each modality, preserving
their unique characteristics and avoiding over-coupling. The Attention-based
Diffusion Network (ADN) generates missing modality features that closely align
with authentic multimodal distribution, enhancing performance across all
missing-modality scenarios. Moreover, ADN's cross-modal generation offers
improved recognition even in full-modality contexts. Our approach achieves
state-of-the-art results on the IEMOCAP and MIntRec benchmarks, demonstrating
its effectiveness in both missing and complete modality scenarios.

### 6. [Enhancing Student Learning with LLM-Generated Retrieval Practice Questions: An Empirical Study in Data Science Courses](http://arxiv.org/pdf/2507.05629v1)

Authors: Yuan An, John Liu, Niyam Acharya, Ruhma Hashmi

Retrieval practice is a well-established pedagogical technique known to
significantly enhance student learning and knowledge retention. However,
generating high-quality retrieval practice questions is often time-consuming
and labor intensive for instructors, especially in rapidly evolving technical
subjects. Large Language Models (LLMs) offer the potential to automate this
process by generating questions in response to prompts, yet the effectiveness
of LLM-generated retrieval practice on student learning remains to be
established. In this study, we conducted an empirical study involving two
college-level data science courses, with approximately 60 students. We compared
learning outcomes during one week in which students received LLM-generated
multiple-choice retrieval practice questions to those from a week in which no
such questions were provided. Results indicate that students exposed to
LLM-generated retrieval practice achieved significantly higher knowledge
retention, with an average accuracy of 89%, compared to 73% in the week without
such practice. These findings suggest that LLM-generated retrieval questions
can effectively support student learning and may provide a scalable solution
for integrating retrieval practice into real-time teaching. However, despite
these encouraging outcomes and the potential time-saving benefits, cautions
must be taken, as the quality of LLM-generated questions can vary. Instructors
must still manually verify and revise the generated questions before releasing
them to students.

### 7. [City-Level Foreign Direct Investment Prediction with Tabular Learning on Judicial Data](http://arxiv.org/pdf/2507.05651v1)

Authors: Tianxing Wu, Lizhe Cao, Shuang Wang, Jiming Wang, Shutong Zhu, Yerong Wu, Yuqing Feng

To advance the United Nations Sustainable Development Goal on promoting
sustained, inclusive, and sustainable economic growth, foreign direct
investment (FDI) plays a crucial role in catalyzing economic expansion and
fostering innovation. Precise city-level FDI prediction is quite important for
local government and is commonly studied based on economic data (e.g., GDP).
However, such economic data could be prone to manipulation, making predictions
less reliable. To address this issue, we try to leverage large-scale judicial
data which reflects judicial performance influencing local investment security
and returns, for city-level FDI prediction. Based on this, we first build an
index system for the evaluation of judicial performance over twelve million
publicly available adjudication documents according to which a tabular dataset
is reformulated. We then propose a new Tabular Learning method on Judicial Data
(TLJD) for city-level FDI prediction. TLJD integrates row data and column data
in our built tabular dataset for judicial performance indicator encoding, and
utilizes a mixture of experts model to adjust the weights of different
indicators considering regional variations. To validate the effectiveness of
TLJD, we design cross-city and cross-time tasks for city-level FDI predictions.
Extensive experiments on both tasks demonstrate the superiority of TLJD (reach
to at least 0.92 R2) over the other ten state-of-the-art baselines in different
evaluation metrics.

### 8. [Divergent Realities: A Comparative Analysis of Human Expert vs. Artificial Intelligence Based Generation and Evaluation of Treatment Plans in Dermatology](http://arxiv.org/pdf/2507.05716v1)

Authors: Dipayan Sengupta, Saumya Panda

Background: Evaluating AI-generated treatment plans is a key challenge as AI
expands beyond diagnostics, especially with new reasoning models. This study
compares plans from human experts and two AI models (a generalist and a
reasoner), assessed by both human peers and a superior AI judge.
  Methods: Ten dermatologists, a generalist AI (GPT-4o), and a reasoning AI
(o3) generated treatment plans for five complex dermatology cases. The
anonymized, normalized plans were scored in two phases: 1) by the ten human
experts, and 2) by a superior AI judge (Gemini 2.5 Pro) using an identical
rubric.
  Results: A profound 'evaluator effect' was observed. Human experts scored
peer-generated plans significantly higher than AI plans (mean 7.62 vs. 7.16;
p=0.0313), ranking GPT-4o 6th (mean 7.38) and the reasoning model, o3, 11th
(mean 6.97). Conversely, the AI judge produced a complete inversion, scoring AI
plans significantly higher than human plans (mean 7.75 vs. 6.79; p=0.0313). It
ranked o3 1st (mean 8.20) and GPT-4o 2nd, placing all human experts lower.
  Conclusions: The perceived quality of a clinical plan is fundamentally
dependent on the evaluator's nature. An advanced reasoning AI, ranked poorly by
human experts, was judged as superior by a sophisticated AI, revealing a deep
gap between experience-based clinical heuristics and data-driven algorithmic
logic. This paradox presents a critical challenge for AI integration,
suggesting the future requires synergistic, explainable human-AI systems that
bridge this reasoning gap to augment clinical care.

### 9. [An autonomous agent for auditing and improving the reliability of clinical AI models](http://arxiv.org/pdf/2507.05755v1)

Authors: Lukas Kuhn, Florian Buettner

The deployment of AI models in clinical practice faces a critical challenge:
models achieving expert-level performance on benchmarks can fail
catastrophically when confronted with real-world variations in medical imaging.
Minor shifts in scanner hardware, lighting or demographics can erode accuracy,
but currently reliability auditing to identify such catastrophic failure cases
before deployment is a bespoke and time-consuming process. Practitioners lack
accessible and interpretable tools to expose and repair hidden failure modes.
Here we introduce ModelAuditor, a self-reflective agent that converses with
users, selects task-specific metrics, and simulates context-dependent,
clinically relevant distribution shifts. ModelAuditor then generates
interpretable reports explaining how much performance likely degrades during
deployment, discussing specific likely failure modes and identifying root
causes and mitigation strategies. Our comprehensive evaluation across three
real-world clinical scenarios - inter-institutional variation in
histopathology, demographic shifts in dermatology, and equipment heterogeneity
in chest radiography - demonstrates that ModelAuditor is able correctly
identify context-specific failure modes of state-of-the-art models such as the
established SIIM-ISIC melanoma classifier. Its targeted recommendations recover
15-25% of performance lost under real-world distribution shift, substantially
outperforming both baseline models and state-of-the-art augmentation methods.
These improvements are achieved through a multi-agent architecture and execute
on consumer hardware in under 10 minutes, costing less than US$0.50 per audit.

### 10. [Real-time monitoring of the SoH of lithium-ion batteries](http://arxiv.org/pdf/2507.05765v1)

Authors: Bruno Jammes, Edgar Hernando Sepúlveda-Oviedo, Corinne Alonso

Real-time monitoring of the state of health (SoH) of batteries remains a
major challenge, particularly in microgrids where operational constraints limit
the use of traditional methods. As part of the 4BLife project, we propose an
innovative method based on the analysis of a discharge pulse at the end of the
charge phase. The parameters of the equivalent electrical model describing the
voltage evolution across the battery terminals during this current pulse are
then used to estimate the SoH. Based on the experimental data acquired so far,
the initial results demonstrate the relevance of the proposed approach. After
training using the parameters of two batteries with a capacity degradation of
around 85%, we successfully predicted the degradation of two other batteries,
cycled down to approximately 90% SoH, with a mean absolute error of around 1%
in the worst case, and an explainability score of the estimator close to 0.9.
If these performances are confirmed, this method can be easily integrated into
battery management systems (BMS) and paves the way for optimized battery
management under continuous operation.

### Hardware Architecture

### 1. [RTGPU: Real-Time Computing with Graphics Processing Units](http://arxiv.org/pdf/2507.06069v1)

Authors: Atiyeh Gheibi-Fetrat, Amirsaeed Ahmadi-Tonekaboni, Farzam Koohi-Ronaghi, Pariya Hajipour, Sana Babayan-Vanestan, Fatemeh Fotouhi, Elahe Mortazavian-Farsani, Pouria Khajehpour-Dezfouli, Sepideh Safari, Shaahin Hessabi, Hamid Sarbazi-Azad

In this work, we survey the role of GPUs in real-time systems. Originally
designed for parallel graphics workloads, GPUs are now widely used in
time-critical applications such as machine learning, autonomous vehicles, and
robotics due to their high computational throughput. Their parallel
architecture is well-suited for accelerating complex tasks under strict timing
constraints. However, their integration into real-time systems presents several
challenges, including non-preemptive execution, execution time variability, and
resource contention; factors that can lead to unpredictable delays and deadline
violations. We examine existing solutions that address these challenges,
including scheduling algorithms, resource management techniques, and
synchronization methods, and highlight open research directions to improve GPU
predictability and performance in real-time environments.

### 2. [Per-Row Activation Counting on Real Hardware: Demystifying Performance Overheads](http://arxiv.org/pdf/2507.05556v1)

Authors: Jumin Kim, Seungmin Baek, Minbok Wi, Hwayong Nam, Michael Jaemin Kim, Sukhan Lee, Kyomin Sohn, Jung Ho Ahn

Per-Row Activation Counting (PRAC), a DRAM read disturbance mitigation
method, modifies key DRAM timing parameters, reportedly causing significant
performance overheads in simulator-based studies. However, given known
discrepancies between simulators and real hardware, real-machine experiments
are vital for accurate PRAC performance estimation. We present the first
real-machine performance analysis of PRAC. After verifying timing modifications
on the latest CPUs using microbenchmarks, our analysis shows that PRAC's
average and maximum overheads are just 1.06% and 3.28% for the SPEC CPU2017
workloads -- up to 9.15x lower than simulator-based reports. Further, we show
that the close page policy minimizes this overhead by effectively hiding the
elongated DRAM row precharge operations due to PRAC from the critical path.

### 3. [iThermTroj: Exploiting Intermittent Thermal Trojans in Multi-Processor System-on-Chips](http://arxiv.org/pdf/2507.05576v1)

Authors: Mehdi Elahi, Mohamed R. Elshamy, Abdel-Hameed Badawy, Ahmad Patooghy

Thermal Trojan attacks present a pressing concern for the security and
reliability of System-on-Chips (SoCs), especially in mobile applications. The
situation becomes more complicated when such attacks are more evasive and
operate sporadically to stay hidden from detection mechanisms. In this paper,
we introduce Intermittent Thermal Trojans (iThermTroj) that exploit the chips'
thermal information in a random time-triggered manner. According to our
experiments, iThermTroj attack can easily bypass available threshold-based
thermal Trojan detection solutions. We investigate SoC vulnerabilities to
variations of iThermTroj through an in-depth analysis of Trojan activation and
duration scenarios. We also propose a set of tiny Machine Learning classifiers
for run-time anomaly detection to protect SoCs against such intermittent
thermal Trojan attacks. Compared to existing methods, our approach improves the
attack detection rate by 29.4\%, 17.2\%, and 14.3\% in scenarios where
iThermTroj manipulates up to 80\%, 60\%, and 40\% of SoC's thermal data,
respectively. Additionally, our method increases the full protection resolution
to 0.8 degrees Celsius, meaning that any temperature manipulations exceeding
$\pm 0.8$ degrees will be detected with 100\% accuracy.

### 4. [OLAF: Programmable Data Plane Acceleration for Asynchronous Distributed Reinforcement Learning](http://arxiv.org/pdf/2507.05876v1)

Authors: Nehal Baganal Krishna, Anam Tahir, Firas Khamis, Mina Tahmasbi Arashloo, Michael Zink, Amr Rizk

Asynchronous Distributed Reinforcement Learning (DRL) can suffer from
degraded convergence when model updates become stale, often the result of
network congestion and packet loss during large-scale training. This work
introduces a network data-plane acceleration architecture that mitigates such
staleness by enabling inline processing of DRL model updates as they traverse
the accelerator engine. To this end, we design and prototype a novel queueing
mechanism that opportunistically combines compatible updates sharing a network
element, reducing redundant traffic and preserving update utility.
Complementing this we provide a lightweight transmission control mechanism at
the worker nodes that is guided by feedback from the in-network accelerator. To
assess model utility at line rate, we introduce the Age-of-Model (AoM) metric
as a proxy for staleness and verify global fairness and responsiveness
properties using a formal verification method. Our evaluations demonstrate that
this architecture significantly reduces update staleness and congestion,
ultimately improving the convergence rate in asynchronous DRL workloads.

### 5. [PrefixAgent: An LLM-Powered Design Framework for Efficient Prefix Adder Optimization](http://arxiv.org/pdf/2507.06127v1)

Authors: Dongsheng Zuo, Jiadong Zhu, Yang Luo, Yuzhe Ma

Prefix adders are fundamental arithmetic circuits, but their design space
grows exponentially with bit-width, posing significant optimization challenges.
Previous works face limitations in performance, generalization, and
scalability. To address these challenges, we propose PrefixAgent, a large
language model (LLM)-powered framework that enables efficient prefix adder
optimization. Specifically, PrefixAgent reformulates the problem into subtasks
including backbone synthesis and structure refinement, which effectively
reduces the search space. More importantly, this new design perspective enables
us to efficiently collect enormous high-quality data and reasoning traces with
E-graph, which further results in an effective fine-tuning of LLM. Experimental
results show that PrefixAgent synthesizes prefix adders with consistently
smaller areas compared to baseline methods, while maintaining scalability and
generalization in commercial EDA flows.

### 6. [GATMesh: Clock Mesh Timing Analysis using Graph Neural Networks](http://arxiv.org/pdf/2507.05681v1)

Authors: Muhammad Hadir Khan, Matthew Guthaus

Clock meshes are essential in high-performance VLSI systems for minimizing
skew and handling PVT variations, but analyzing them is difficult due to
reconvergent paths, multi-source driving, and input mesh buffer skew. SPICE
simulations are accurate but slow; yet simplified models miss key effects like
slew and input skew. We propose GATMesh, a Graph Neural Network (GNN)-based
framework that models the clock mesh as a graph with augmented structural and
physical features. Trained on SPICE data, GATMesh achieves high accuracy with
average delay error of 5.27ps on unseen benchmarks, while achieving speed-ups
of 47146x over multi-threaded SPICE simulation.

### Computational Complexity

### 1. [Parameterized Restless Temporal Path](http://arxiv.org/pdf/2507.05760v1)

Authors: Justine Cauvi, Laurent Viennot

Recently, Bumpus and Meeks introduced a purely temporal parameter, called
vertex-interval-membership-width, which is promising for the design of
fixed-parameter tractable (FPT) algorithms for vertex reachability problems in
temporal graphs. We study this newly introduced parameter for the problem of
restless temporal paths, in which the waiting time at each node is restricted.
In this article, we prove that, in the interval model, where arcs are present
for entire time intervals, finding a restless temporal path is NP-hard even if
the vertex-interval-membership-width is equal to three. We exhibit FPT
algorithms for the point model, where arcs are present at specific points in
time, both with uniform delay one and arbitrary positive delays. In the latter
case, this comes with a slight additional computational cost.

### 2. [Complexity Results of Persuasion](http://arxiv.org/pdf/2507.05951v1)

Authors: Alban Grastien

We prove that persuasion is an NP-complete problem.

### 3. [Generalized and Unified Equivalences between Hardness and Pseudoentropy](http://arxiv.org/pdf/2507.05972v1)

Authors: Lunjia Hu, Salil Vadhan

Pseudoentropy characterizations provide a quantitatively precise
demonstration of the close relationship between computational hardness and
computational randomness. We prove a unified pseudoentropy characterization
that generalizes and strengthens previous results for both uniform and
non-uniform models of computation. Our characterization holds for a general
family of entropy notions that encompasses the common notions of Shannon
entropy and min entropy as special cases. Moreover, we show that the
characterizations for different entropy notions can be simultaneously achieved
by a single, universal function that simultaneously witnesses computational
hardness and computational randomness. A key technical insight of our work is
that the notion of weight-restricted calibration from the recent literature on
algorithm fairness, along with standard computational indistinguishability
(known as multiaccuracy in the fairness literature), suffices for proving
pseudoentropy characterizations for general entropy notions. This demonstrates
the power of weight-restricted calibration to enhance the classic
Complexity-Theoretic Regularity Lemma (Trevisan, Tulsiani, and Vadhan, 2009)
and Leakage Simulation Lemma (Jetchev and Pietrzak, 2014) and allows us to
achieve an exponential improvement in the complexity dependency on the alphabet
size compared to the pseudoentropy characterizations by Casacuberta, Dwork, and
Vadhan (2024) based on the much stronger notion of multicalibration. We show
that the exponential dependency on the alphabet size is inevitable for
multicalibration as well as for the weaker notion of calibrated multiaccuracy.

### 4. [A Formal Refutation of the Blockchain Trilemma](http://arxiv.org/pdf/2507.05809v1)

Authors: Craig Wright

The so-called blockchain trilemma asserts the impossibility of simultaneously
achieving scalability, security, and decentralisation within a single
blockchain protocol. In this paper, we formally refute that proposition.
Employing predicate logic, formal automata theory, computational complexity
analysis, and graph-theoretic measures of relay topology--specifically Baran's
model of network path redundancy--we demonstrate that the trilemma constitutes
a category error, conflates distinct analytical domains, and relies upon
unproven causal assumptions. We further expose its reliance on composition
fallacies drawn from flawed system implementations. A constructive
counterexample is presented: a blockchain protocol exhibiting unbounded
transaction throughput, cryptographic security under adversarial load, and
multipath decentralised propagation. This example is not hypothetical but
grounded in protocol design enabled by compact block relay, SPV verification,
and IPv6 multicast. The trilemma is revealed not as a law of protocol
architecture, but as a heuristic fallacy sustained by imprecision and design
defeatism.

### 5. [On the Complexity of Problems on Graphs Defined on Groups](http://arxiv.org/pdf/2507.05860v1)

Authors: Bireswar Das, Dipan Dey, Jinia Ghosh

We study the complexity of graph problems on graphs defined on groups,
especially power graphs. We observe that an isomorphism invariant problem, such
as Hamiltonian Path, Partition into Cliques, Feedback Vertex Set, Subgraph
Isomorphism, cannot be NP-complete for power graphs, commuting graphs, enhanced
power graphs, directed power graphs, and bounded-degree Cayley graphs, assuming
the Exponential Time Hypothesis (ETH). An analogous result holds for
isomorphism invariant group problems: no such problem can be NP-complete unless
ETH is false. We show that the Weighted Max-Cut problem is NP-complete in power
graphs. We also show that, unless ETH is false, the Graph Motif problem cannot
be solved in quasipolynomial time on power graphs, even for power graphs of
cyclic groups. We study the recognition problem of power graphs when the
adjacency matrix or list is given as input and show that for abelian groups and
some classes of nilpotent groups, it is solvable in polynomial time.

### 6. [Unitary designs in nearly optimal depth](http://arxiv.org/pdf/2507.06216v1)

Authors: Laura Cui, Thomas Schuster, Fernando Brandao, Hsin-Yuan Huang

We construct $\varepsilon$-approximate unitary $k$-designs on $n$ qubits in
circuit depth $O(\log k \log \log n k / \varepsilon)$. The depth is
exponentially improved over all known results in all three parameters $n$, $k$,
$\varepsilon$. We further show that each dependence is optimal up to
exponentially smaller factors. Our construction uses $\tilde{{O}}(nk)$ ancilla
qubits and ${O}(nk)$ bits of randomness, which are also optimal up to $\log(n
k)$ factors. An alternative construction achieves a smaller ancilla count
$\tilde{{O}}(n)$ with circuit depth ${O}(k \log \log nk/\varepsilon)$. To
achieve these efficient unitary designs, we introduce a highly-structured
random unitary ensemble that leverages long-range two-qubit gates and low-depth
implementations of random classical hash functions. We also develop a new
analytical framework for bounding errors in quantum experiments involving many
queries to random unitaries. As an illustration of this framework's
versatility, we provide a succinct alternative proof of the existence of
pseudorandom unitaries.

### Computational Engineering

### 1. [Bridging Sequential Deep Operator Network and Video Diffusion: Residual Refinement of Spatio-Temporal PDE Solutions](http://arxiv.org/pdf/2507.06133v1)

Authors: Jaewan Park, Farid Ahmed, Kazuma Kobayashi, Seid Koric, Syed Bahauddin Alam, Iwona Jasiuk, Diab Abueidda

Video-diffusion models have recently set the standard in video generation,
inpainting, and domain translation thanks to their training stability and high
perceptual fidelity. Building on these strengths, we repurpose conditional
video diffusion as a physics surrogate for spatio-temporal fields governed by
partial differential equations (PDEs). Our two-stage surrogate first applies a
Sequential Deep Operator Network (S-DeepONet) to produce a coarse,
physics-consistent prior from the prescribed boundary or loading conditions.
The prior is then passed to a conditional video diffusion model that learns
only the residual: the point-wise difference between the ground truth and the
S-DeepONet prediction. By shifting the learning burden from the full solution
to its much smaller residual space, diffusion can focus on sharpening
high-frequency structures without sacrificing global coherence. The framework
is assessed on two disparate benchmarks: (i) vortex-dominated lid-driven cavity
flow and (ii) tensile plastic deformation of dogbone specimens. Across these
data sets the hybrid surrogate consistently outperforms its single-stage
counterpart, cutting the mean relative L2 error from 4.57% to 0.83% for the
flow problem and from 4.42% to 2.94% for plasticity, a relative improvements of
81.8% and 33.5% respectively. The hybrid approach not only lowers quantitative
errors but also improves visual quality, visibly recovering fine spatial
details. These results show that (i) conditioning diffusion on a physics-aware
prior enables faithful reconstruction of localized features, (ii) residual
learning reduces the problem, accelerating convergence and enhancing accuracy,
and (iii) the same architecture transfers seamlessly from incompressible flow
to nonlinear elasto-plasticity without problem-specific architectural
modifications, highlighting its broad applicability to nonlinear,
time-dependent continua.

### 2. [Affective-ROPTester: Capability and Bias Analysis of LLMs in Predicting Retinopathy of Prematurity](http://arxiv.org/pdf/2507.05816v1)

Authors: Shuai Zhao, Yulin Zhang, Luwei Xiao, Xinyi Wu, Yanhao Jia, Zhongliang Guo, Xiaobao Wu, Cong-Duy Nguyen, Guoming Zhang, Anh Tuan Luu

Despite the remarkable progress of large language models (LLMs) across
various domains, their capacity to predict retinopathy of prematurity (ROP)
risk remains largely unexplored. To address this gap, we introduce a novel
Chinese benchmark dataset, termed CROP, comprising 993 admission records
annotated with low, medium, and high-risk labels. To systematically examine the
predictive capabilities and affective biases of LLMs in ROP risk
stratification, we propose Affective-ROPTester, an automated evaluation
framework incorporating three prompting strategies: Instruction-based,
Chain-of-Thought (CoT), and In-Context Learning (ICL). The Instruction scheme
assesses LLMs' intrinsic knowledge and associated biases, whereas the CoT and
ICL schemes leverage external medical knowledge to enhance predictive accuracy.
Crucially, we integrate emotional elements at the prompt level to investigate
how different affective framings influence the model's ability to predict ROP
and its bias patterns. Empirical results derived from the CROP dataset yield
two principal observations. First, LLMs demonstrate limited efficacy in ROP
risk prediction when operating solely on intrinsic knowledge, yet exhibit
marked performance gains when augmented with structured external inputs.
Second, affective biases are evident in the model outputs, with a consistent
inclination toward overestimating medium- and high-risk cases. Third, compared
to negative emotions, positive emotional framing contributes to mitigating
predictive bias in model outputs. These findings highlight the critical role of
affect-sensitive prompt engineering in enhancing diagnostic reliability and
emphasize the utility of Affective-ROPTester as a framework for evaluating and
mitigating affective bias in clinical language modeling systems.

### 3. [Topic Modeling and Link-Prediction for Material Property Discovery](http://arxiv.org/pdf/2507.06139v1)

Authors: Ryan C. Barron, Maksim E. Eren, Valentin Stanev, Cynthia Matuszek, Boian S. Alexandrov

Link prediction infers missing or future relations between graph nodes, based
on connection patterns. Scientific literature networks and knowledge graphs are
typically large, sparse, and noisy, and often contain missing links between
entities. We present an AI-driven hierarchical link prediction framework that
integrates matrix factorization to infer hidden associations and steer
discovery in complex material domains. Our method combines Hierarchical
Nonnegative Matrix Factorization (HNMFk) and Boolean matrix factorization
(BNMFk) with automatic model selection, as well as Logistic matrix
factorization (LMF), we use to construct a three-level topic tree from a
46,862-document corpus focused on 73 transition-metal dichalcogenides (TMDs).
These materials are studied in a variety of physics fields with many current
and potential applications.
  An ensemble BNMFk + LMF approach fuses discrete interpretability with
probabilistic scoring. The resulting HNMFk clusters map each material onto
coherent topics like superconductivity, energy storage, and tribology. Also,
missing or weakly connected links are highlight between topics and materials,
suggesting novel hypotheses for cross-disciplinary exploration. We validate our
method by removing publications about superconductivity in well-known
superconductors, and show the model predicts associations with the
superconducting TMD clusters. This shows the method finds hidden connections in
a graph of material to latent topic associations built from scientific
literature, especially useful when examining a diverse corpus of scientific
documents covering the same class of phenomena or materials but originating
from distinct communities and perspectives. The inferred links generating new
hypotheses, produced by our method, are exposed through an interactive
Streamlit dashboard, designed for human-in-the-loop scientific discovery.

### Computational Geometry

### 1. [$k$-means considered harmful: On arbitrary topological changes in Mapper complexes](http://arxiv.org/pdf/2507.06212v1)

Authors: Mikael Vejdemo-Johansson

The Mapper construction is one of the most widespread tools from Topological
Data Analysis. There is an unfortunate trend as the construction has gained
traction to use clustering methods with properties that end up distorting any
analysis results from the construction. In this paper we will see a few ways in
which widespread choices of clustering algorithms have arbitrarily large
distortions of the features visible in the final Mapper complex.

### 2. [An Optimal Algorithm for Shortest Paths in Unweighted Disk Graphs](http://arxiv.org/pdf/2507.05569v1)

Authors: Bruce W. Brewer, Haitao Wang

Given in the plane a set $S$ of $n$ points and a set of disks centered at
these points, the disk graph $G(S)$ induced by these disks has vertex set $S$
and an edge between two vertices if their disks intersect. Note that the disks
may have different radii. We consider the problem of computing shortest paths
from a source point $s\in S$ to all vertices in $G(S)$ where the length of a
path in $G(S)$ is defined as the number of edges in the path. The previously
best algorithm solves the problem in $O(n\log^2 n)$ time. A lower bound of
$\Omega(n\log n)$ is also known for this problem under the algebraic decision
tree model. In this paper, we present an $O(n\log n)$ time algorithm, which
matches the lower bound and thus is optimal. Another virtue of our algorithm is
that it is quite simple.

### 3. [Fast and Accurate Collision Probability Estimation for Autonomous Vehicles using Adaptive Sigma-Point Sampling](http://arxiv.org/pdf/2507.06149v1)

Authors: Charles Champagne Cossette, Taylor Scott Clawson, Andrew Feit

A novel algorithm is presented for the estimation of collision probabilities
between dynamic objects with uncertain trajectories, where the trajectories are
given as a sequence of poses with Gaussian distributions. We propose an
adaptive sigma-point sampling scheme, which ultimately produces a fast, simple
algorithm capable of estimating the collision probability with a median error
of 3.5%, and a median runtime of 0.21ms, when measured on an Intel Xeon Gold
6226R Processor. Importantly, the algorithm explicitly accounts for the
collision probability's temporal dependence, which is often neglected in prior
work and otherwise leads to an overestimation of the collision probability.
Finally, the method is tested on a diverse set of relevant real-world
scenarios, consisting of 400 6-second snippets of autonomous vehicle logs,
where the accuracy and latency is rigorously evaluated.

### Computation and Language

### 1. [Enhancing Test-Time Scaling of Large Language Models with Hierarchical Retrieval-Augmented MCTS](http://arxiv.org/pdf/2507.05557v1)

Authors: Alex ZH Dou, Zhongwei Wan, Dongfei Cui, Xin Wang, Jing Xiong, Haokun Lin, Chaofan Tao, Shen Yan, Mi Zhang

Test-time scaling has emerged as a promising paradigm in language modeling,
leveraging additional computational resources at inference time to enhance
model performance. In this work, we introduce R2-LLMs, a novel and versatile
hierarchical retrieval-augmented reasoning framework designed to improve
test-time scaling in large language models (LLMs) without requiring
distillation from more advanced models to obtain chain-of-thought (CoT)
training data. R2-LLMs enhances inference-time generalization by integrating
dual-level retrieval-based in-context learning: (1) At the coarse level, our
approach extracts abstract templates from complex reasoning problems and
retrieves similar problem-answer pairs to facilitate high-level in-context
learning; (2) At the fine level, during Monte Carlo Tree Search (MCTS), R2-LLMs
efficiently retrieves analogous intermediate solution steps from reference
mathematical problem datasets, refining step-wise reasoning with the aid of a
process reward model (PRM) for scoring. R2-LLMs is a robust hierarchical
reasoning-augmentation method that enhances in-context-level reasoning while
seamlessly integrating with step-level tree search methods. Utilizing PRM, it
refines both candidate generation and decision-making for improved reasoning
accuracy. Empirical evaluations on the MATH500, GSM8K, and OlympiadBench-TO
datasets achieve substantial relative improvement with an increase of up to 16%
using LLaMA-3.1-8B compared to the baselines, showcasing the effectiveness of
our approach in complex reasoning tasks.

### 2. [Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](http://arxiv.org/pdf/2507.05617v1)

Authors: Mingzhe Li, Jing Xiang, Qishen Zhang, Kaiyang Wan, Xiuying Chen

Knowledge distillation typically involves transferring knowledge from a Large
Language Model (LLM) to a Smaller Language Model (SLM). However, in tasks such
as text matching, fine-tuned smaller models often yield more effective
domain-specific representations, as they focus on optimizing the similarity of
input pairs. To leverage both the specialized strengths of small models and the
rich semantic understanding of LLMs, we introduce a flipped knowledge
distillation paradigm, where LLM learns from SLM. Specifically, we address the
architectural gap between decoder-only LLMs and smaller encoder-based models by
reinterpreting LLMs in an encoder-decoder manner using LoRA. The encoder
generates compressed representations, while the decoder maps them to the output
space. During training, the encoder produces representations and their
similarities, which are then aligned with the similarity scores produced by the
teacher, using our proposed Margin-aware Contrastive Learning (MCL) approach.
The MCL ensures accurate similarity for both positive and negative pairs, and
adaptively handles the internal differences within positive and negative
samples. Our paradigm requires only a reasonably good-performing SLM, allowing
the LLM to achieve improved performance. Experiments on financial and
healthcare benchmarks, as well as real-world applications, confirm its
effectiveness, and the model has been fully deployed in an online environment.

### 3. [ECom-Bench: Can LLM Agent Resolve Real-World E-commerce Customer Support Issues?](http://arxiv.org/pdf/2507.05639v1)

Authors: Haoxin Wang, Xianhan Peng, Xucheng Huang, Yizhe Huang, Ming Gong, Chenghan Yang, Yang Liu, Ling Jiang

In this paper, we introduce ECom-Bench, the first benchmark framework for
evaluating LLM agent with multimodal capabilities in the e-commerce customer
support domain. ECom-Bench features dynamic user simulation based on persona
information collected from real e-commerce customer interactions and a
realistic task dataset derived from authentic e-commerce dialogues. These
tasks, covering a wide range of business scenarios, are designed to reflect
real-world complexities, making ECom-Bench highly challenging. For instance,
even advanced models like GPT-4o achieve only a 10-20% pass^3 metric in our
benchmark, highlighting the substantial difficulties posed by complex
e-commerce scenarios. Upon publication, the code and data will be open-sourced
to facilitate further research and development in this domain.

### 4. [Smoothie-Qwen: Post-Hoc Smoothing to Reduce Language Bias in Multilingual LLMs](http://arxiv.org/pdf/2507.05686v1)

Authors: SeungWon Ji, Jungyup Lee, Jemin Kim, Sang Park, SeungJae Lee

Multilingual large language models (LLMs) often exhibit language confusion, a
tendency to generate responses in a dominant language irrespective of the
prompt's language. To address this, we propose Smoothie-Qwen, a lightweight,
post-hoc method that mitigates language bias without retraining. This technique
selectively adjusts token-level output probabilities to effectively suppress
undesired language generation. Applied to the Qwen model, our method reduces
unintended Chinese output by over 95% while preserving task accuracy on
multilingual benchmarks. This work provides a practical and efficient solution
for enhancing the language controllability of LLMs, making them more reliable
for global applications.

### 5. [GPTKB v1.5: A Massive Knowledge Base for Exploring Factual LLM Knowledge](http://arxiv.org/pdf/2507.05740v1)

Authors: Yujia Hu, Tuan-Phong Nguyen, Shrestha Ghosh, Moritz Müller, Simon Razniewski

Language models are powerful tools, yet their factual knowledge is still
poorly understood, and inaccessible to ad-hoc browsing and scalable statistical
analysis. This demonstration introduces GPTKB v1.5, a densely interlinked
100-million-triple knowledge base (KB) built for $14,000 from GPT-4.1, using
the GPTKB methodology for massive-recursive LLM knowledge materialization (Hu
et al., ACL 2025). The demonstration experience focuses on three use cases: (1)
link-traversal-based LLM knowledge exploration, (2) SPARQL-based structured LLM
knowledge querying, (3) comparative exploration of the strengths and weaknesses
of LLM knowledge. Massive-recursive LLM knowledge materialization is a
groundbreaking opportunity both for the research area of systematic analysis of
LLM knowledge, as well as for automated KB construction. The GPTKB demonstrator
is accessible at https://gptkb.org.

### 6. [DocTalk: Scalable Graph-based Dialogue Synthesis for Enhancing LLM Conversational Capabilities](http://arxiv.org/pdf/2507.05750v1)

Authors: Jing Yang Lee, Hamed Bonab, Nasser Zalmout, Ming Zeng, Sanket Lokegaonkar, Colin Lockard, Binxuan Huang, Ritesh Sarkhel, Haodong Wang

Large Language Models (LLMs) are increasingly employed in multi-turn
conversational tasks, yet their pre-training data predominantly consists of
continuous prose, creating a potential mismatch between required capabilities
and training paradigms. We introduce a novel approach to address this
discrepancy by synthesizing conversational data from existing text corpora. We
present a pipeline that transforms a cluster of multiple related documents into
an extended multi-turn, multi-topic information-seeking dialogue. Applying our
pipeline to Wikipedia articles, we curate DocTalk, a multi-turn pre-training
dialogue corpus consisting of over 730k long conversations. We hypothesize that
exposure to such synthesized conversational structures during pre-training can
enhance the fundamental multi-turn capabilities of LLMs, such as context memory
and understanding. Empirically, we show that incorporating DocTalk during
pre-training results in up to 40% gain in context memory and understanding,
without compromising base performance. DocTalk is available at
https://huggingface.co/datasets/AmazonScience/DocTalk.

### 7. [Flippi: End To End GenAI Assistant for E-Commerce](http://arxiv.org/pdf/2507.05788v1)

Authors: Anand A. Rajasekar, Praveen Tangarajan, Anjali Nainani, Amogh Batwal, Vinay Rao Dandin, Anusua Trivedi, Ozan Ersoy

The emergence of conversational assistants has fundamentally reshaped user
interactions with digital platforms. This paper introduces Flippi-a
cutting-edge, end-to-end conversational assistant powered by large language
models (LLMs) and tailored for the e-commerce sector. Flippi addresses the
challenges posed by the vast and often overwhelming product landscape, enabling
customers to discover products more efficiently through natural language
dialogue. By accommodating both objective and subjective user requirements,
Flippi delivers a personalized shopping experience that surpasses traditional
search methods. This paper details how Flippi interprets customer queries to
provide precise product information, leveraging advanced NLP techniques such as
Query Reformulation, Intent Detection, Retrieval-Augmented Generation (RAG),
Named Entity Recognition (NER), and Context Reduction. Flippi's unique
capability to identify and present the most attractive offers on an e-commerce
site is also explored, demonstrating how it empowers users to make
cost-effective decisions. Additionally, the paper discusses Flippi's
comparative analysis features, which help users make informed choices by
contrasting product features, prices, and other relevant attributes. The
system's robust architecture is outlined, emphasizing its adaptability for
integration across various e-commerce platforms and the technological choices
underpinning its performance and accuracy. Finally, a comprehensive evaluation
framework is presented, covering performance metrics, user satisfaction, and
the impact on customer engagement and conversion rates. By bridging the
convenience of online shopping with the personalized assistance traditionally
found in physical stores, Flippi sets a new standard for customer satisfaction
and engagement in the digital marketplace.

### 8. [Bridging Perception and Language: A Systematic Benchmark for LVLMs' Understanding of Amodal Completion Reports](http://arxiv.org/pdf/2507.05799v1)

Authors: Amane Watahiki, Tomoki Doi, Taiga Shinozaki, Satoshi Nishida, Takuya Niikawa, Katsunori Miyahara, Hitomi Yanaka

One of the main objectives in developing large vision-language models (LVLMs)
is to engineer systems that can assist humans with multimodal tasks, including
interpreting descriptions of perceptual experiences. A central phenomenon in
this context is amodal completion, in which people perceive objects even when
parts of those objects are hidden. Although numerous studies have assessed
whether computer-vision algorithms can detect or reconstruct occluded regions,
the inferential abilities of LVLMs on texts related to amodal completion remain
unexplored. To address this gap, we constructed a benchmark grounded in Basic
Formal Ontology to achieve a systematic classification of amodal completion.
Our results indicate that while many LVLMs achieve human-comparable performance
overall, their accuracy diverges for certain types of objects being completed.
Notably, in certain categories, some LLaVA-NeXT variants and Claude 3.5 Sonnet
exhibit lower accuracy on original images compared to blank stimuli lacking
visual content. Intriguingly, this disparity emerges only under Japanese
prompting, suggesting a deficiency in Japanese-specific linguistic competence
among these models.

### 9. [Few-shot text-based emotion detection](http://arxiv.org/pdf/2507.05918v1)

Authors: Teodor-George Marchitan, Claudiu Creanga, Liviu P. Dinu

This paper describes the approach of the Unibuc - NLP team in tackling the
SemEval 2025 Workshop, Task 11: Bridging the Gap in Text-Based Emotion
Detection. We mainly focused on experiments using large language models
(Gemini, Qwen, DeepSeek) with either few-shot prompting or fine-tuning. With
our final system, for the multi-label emotion detection track (track A), we got
an F1-macro of $0.7546$ (26/96 teams) for the English subset, $0.1727$ (35/36
teams) for the Portuguese (Mozambican) subset and $0.325$ (\textbf{1}/31 teams)
for the Emakhuwa subset.

### 10. [Towards a Principled Evaluation of Knowledge Editors](http://arxiv.org/pdf/2507.05937v1)

Authors: Sebastian Pohl, Max Ploner, Alan Akbik

Model editing has been gaining increasing attention over the past few years.
For Knowledge Editing in particular, more challenging evaluation datasets have
recently been released. These datasets use different methodologies to score the
success of editors. Yet, it remains under-explored how robust these
methodologies are and whether they unfairly favor some editors. Moreover, the
disruptive impact of these editors on overall model capabilities remains a
constant blind spot.
  We address both of these problems and show that choosing different metrics
and evaluation methodologies as well as different edit batch sizes can lead to
a different ranking of knowledge editors. Crucially we demonstrate this effect
also on general language understanding tasks evaluated alongside the knowledge
editing tasks. Further we include a manual assessment of the string matching
based evaluation method for knowledge editing that is favored by recently
released datasets, revealing a tendency to produce false positive matches.

### Cryptography and Security

### 1. [Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](http://arxiv.org/pdf/2507.05728v1)

Authors: Ruofei Wang, Peiqi Duan, Boxin Shi, Renjie Wan

With more event datasets being released online, safeguarding the event
dataset against unauthorized usage has become a serious concern for data
owners. Unlearnable Examples are proposed to prevent the unauthorized
exploitation of image datasets. However, it's unclear how to create unlearnable
asynchronous event streams to prevent event misuse. In this work, we propose
the first unlearnable event stream generation method to prevent unauthorized
training from event datasets. A new form of asynchronous event error-minimizing
noise is proposed to perturb event streams, tricking the unauthorized model
into learning embedded noise instead of realistic features. To be compatible
with the sparse event, a projection strategy is presented to sparsify the noise
to render our unlearnable event streams (UEvs). Extensive experiments
demonstrate that our method effectively protects event data from unauthorized
exploitation, while preserving their utility for legitimate use. We hope our
UEvs contribute to the advancement of secure and trustworthy event dataset
sharing. Code is available at: https://github.com/rfww/uevs.

### 2. [LDP$^3$: An Extensible and Multi-Threaded Toolkit for Local Differential Privacy Protocols and Post-Processing Methods](http://arxiv.org/pdf/2507.05872v1)

Authors: Berkay Kemal Balioglu, Alireza Khodaie, Mehmet Emre Gursoy

Local differential privacy (LDP) has become a prominent notion for
privacy-preserving data collection. While numerous LDP protocols and
post-processing (PP) methods have been developed, selecting an optimal
combination under different privacy budgets and datasets remains a challenge.
Moreover, the lack of a comprehensive and extensible LDP benchmarking toolkit
raises difficulties in evaluating new protocols and PP methods. To address
these concerns, this paper presents LDP$^3$ (pronounced LDP-Cube), an
open-source, extensible, and multi-threaded toolkit for LDP researchers and
practitioners. LDP$^3$ contains implementations of several LDP protocols, PP
methods, and utility metrics in a modular and extensible design. Its modular
design enables developers to conveniently integrate new protocols and PP
methods. Furthermore, its multi-threaded nature enables significant reductions
in execution times via parallelization. Experimental evaluations demonstrate
that: (i) using LDP$^3$ to select a good protocol and post-processing method
substantially improves utility compared to a bad or random choice, and (ii) the
multi-threaded design of LDP$^3$ brings substantial benefits in terms of
efficiency.

### 3. [Post-Processing in Local Differential Privacy: An Extensive Evaluation and Benchmark Platform](http://arxiv.org/pdf/2507.05875v1)

Authors: Alireza Khodaie, Berkay Kemal Balioglu, Mehmet Emre Gursoy

Local differential privacy (LDP) has recently gained prominence as a powerful
paradigm for collecting and analyzing sensitive data from users' devices.
However, the inherent perturbation added by LDP protocols reduces the utility
of the collected data. To mitigate this issue, several post-processing (PP)
methods have been developed. Yet, the comparative performance of PP methods
under diverse settings remains underexplored. In this paper, we present an
extensive benchmark comprising 6 popular LDP protocols, 7 PP methods, 4 utility
metrics, and 6 datasets to evaluate the behaviors and optimality of PP methods
under diverse conditions. Through extensive experiments, we show that while PP
can substantially improve utility when the privacy budget is small (i.e.,
strict privacy), its benefit diminishes as the privacy budget grows. Moreover,
our findings reveal that the optimal PP method depends on multiple factors,
including the choice of LDP protocol, privacy budget, data characteristics
(such as distribution and domain size), and the specific utility metric. To
advance research in this area and assist practitioners in identifying the most
suitable PP method for their setting, we introduce LDP$^3$, an open-source
benchmark platform. LDP$^3$ contains all methods used in our experimental
analysis, and it is designed in a modular, extensible, and multi-threaded way
for future use and development.

### 4. [Enter, Exit, Page Fault, Leak: Testing Isolation Boundaries for Microarchitectural Leaks](http://arxiv.org/pdf/2507.06039v1)

Authors: Oleksii Oleksenko, Flavien Solt, Cédric Fournet, Jana Hofmann, Boris Köpf, Stavros Volos

CPUs provide isolation mechanisms like virtualization and privilege levels to
protect software. Yet these focus on architectural isolation while typically
overlooking microarchitectural side channels, exemplified by Meltdown and
Foreshadow. Software must therefore supplement architectural defenses with
ad-hoc microarchitectural patches, which are constantly evolving as new attacks
emerge and defenses are proposed. Such reactive approach makes ensuring
complete isolation a daunting task, and leaves room for errors and oversights.
  We address this problem by developing a tool that stress tests
microarchitectural isolation between security domains such as virtual machines,
kernel, and processes, with the goal of detecting flaws in the isolation
boundaries. The tool extends model-based relational testing (MRT) methodology
to enable detection of cross-domain information leakage. We design a new test
case generator and execution sandbox to handle multi-domain execution, new
leakage models to encode expected leaks, and new analysis techniques to manage
nondeterminism.
  We use this tool to perform an in-depth testing campaign on six x86-64 CPUs
for leakage across different isolation boundaries. The testing campaign exposed
four new leaks and corroborated numerous known ones, with only two false
positives throughout the entire campaign. These results show critical gaps in
current isolation mechanisms as well as validate a robust methodology for
detecting microarchitectural flaws. As such, this approach enables a shift from
reactive patching to proactive security validation in processor design.

### 5. [Wrapless: The trustless lending protocol on top of Bitcoin](http://arxiv.org/pdf/2507.06064v1)

Authors: Oleksandr Kurbatov, Kyrylo Baybula, Yaroslava Chopa, Sergey Kozlov, Oleg Komendant, Illia Dovgopoly, Dmitrii Kurbatov, Zakhar Naumets, Yulia Artikulova, Pavel Kravchenko, Volodymyr Dubinin, Lasha Antadze, Yaroslav Panasenko, Mykhailo Velykodnyi

This paper presents Wrapless -- a lending protocol that enables the
collateralization of bitcoins without requiring a trusted wrapping mechanism.
The protocol facilitates a "loan channel" on the Bitcoin blockchain, allowing
bitcoins to be locked as collateral for loans issued on any blockchain that
supports Turing-complete smart contracts. The protocol is designed in a way
that makes it economically irrational for each involved party to manipulate the
loan rules. There is still a significant research area to bring the protocol
closer to traditional AMM financial instruments.

### 6. [Fun with flags: How Compilers Break and Fix Constant-Time Code](http://arxiv.org/pdf/2507.06112v1)

Authors: Antoine Geimer, Clementine Maurice

Developers rely on constant-time programming to prevent timing side-channel
attacks. But these efforts can be undone by compilers, whose optimizations may
silently reintroduce leaks. While recent works have measured the extent of such
leakage, they leave developers without actionable insights: which optimization
passes are responsible, and how to disable them without modifying the compiler
remains unclear.
  In this paper, we conduct a qualitative analysis of how compiler
optimizations break constant-time code. We construct a dataset of
compiler-introduced constant-time violations and analyze the internals of two
widely used compilers, GCC and LLVM, to identify the specific optimization
passes responsible. Our key insight is that a small set of passes are at the
root of most leaks. To the best of our knowledge, we are also the first to
characterize how the interactions between these passes contribute to leakage.
Based on this analysis, we propose an original and practical mitigation that
requires no source code modification or custom compiler: disabling selected
optimization passes via compiler flags. We show that this approach
significantly reduces leakage with minimal performance overhead, offering an
immediately deployable defense for developers.

### 7. [Per-Row Activation Counting on Real Hardware: Demystifying Performance Overheads](http://arxiv.org/pdf/2507.05556v1)

Authors: Jumin Kim, Seungmin Baek, Minbok Wi, Hwayong Nam, Michael Jaemin Kim, Sukhan Lee, Kyomin Sohn, Jung Ho Ahn

Per-Row Activation Counting (PRAC), a DRAM read disturbance mitigation
method, modifies key DRAM timing parameters, reportedly causing significant
performance overheads in simulator-based studies. However, given known
discrepancies between simulators and real hardware, real-machine experiments
are vital for accurate PRAC performance estimation. We present the first
real-machine performance analysis of PRAC. After verifying timing modifications
on the latest CPUs using microbenchmarks, our analysis shows that PRAC's
average and maximum overheads are just 1.06% and 3.28% for the SPEC CPU2017
workloads -- up to 9.15x lower than simulator-based reports. Further, we show
that the close page policy minimizes this overhead by effectively hiding the
elongated DRAM row precharge operations due to PRAC from the critical path.

### 8. [AI Agent Smart Contract Exploit Generation](http://arxiv.org/pdf/2507.05558v1)

Authors: Arthur Gervais, Liyi Zhou

We present A1, an agentic execution driven system that transforms any LLM
into an end-to-end exploit generator. A1 has no hand-crafted heuristics and
provides the agent with six domain-specific tools that enable autonomous
vulnerability discovery. The agent can flexibly leverage these tools to
understand smart contract behavior, generate exploit strategies, test them on
blockchain states, and refine approaches based on execution feedback. All
outputs are concretely validated to eliminate false positives.
  The evaluation across 36 real-world vulnerable contracts on Ethereum and
Binance Smart Chain demonstrates a 62.96% (17 out of 27) success rate on the
VERITE benchmark. Beyond the VERITE dataset, A1 identified 9 additional
vulnerable contracts, with 5 cases occurring after the strongest model's
training cutoff date. Across all 26 successful cases, A1 extracts up to 8.59
million USD per case and 9.33 million USD total. Through 432 experiments across
six LLMs, we analyze iteration-wise performance showing diminishing returns
with average marginal gains of +9.7%, +3.7%, +5.1%, and +2.8% for iterations
2-5 respectively, with per-experiment costs ranging $0.01-$3.59. A Monte Carlo
analysis of 19 historical attacks shows success probabilities of 85.9%-88.8%
without detection delays.
  We investigate whether an attacker or a defender benefits most from deploying
A1 as a continuous on-chain scanning system. Our model shows that OpenAI's
o3-pro maintains profitability up to a 30.0 days scanning delay at 0.100%
vulnerability incidence rates, while faster models require >=1.000% rates to
break-even. The findings exposes a troubling asymmetry: at 0.1% vulnerability
rates, attackers achieve an on-chain scanning profitability at a $6000 exploit
value, while defenders require $60000, raising fundamental questions about
whether AI agents inevitably favor exploitation over defense.

### 9. [iThermTroj: Exploiting Intermittent Thermal Trojans in Multi-Processor System-on-Chips](http://arxiv.org/pdf/2507.05576v1)

Authors: Mehdi Elahi, Mohamed R. Elshamy, Abdel-Hameed Badawy, Ahmad Patooghy

Thermal Trojan attacks present a pressing concern for the security and
reliability of System-on-Chips (SoCs), especially in mobile applications. The
situation becomes more complicated when such attacks are more evasive and
operate sporadically to stay hidden from detection mechanisms. In this paper,
we introduce Intermittent Thermal Trojans (iThermTroj) that exploit the chips'
thermal information in a random time-triggered manner. According to our
experiments, iThermTroj attack can easily bypass available threshold-based
thermal Trojan detection solutions. We investigate SoC vulnerabilities to
variations of iThermTroj through an in-depth analysis of Trojan activation and
duration scenarios. We also propose a set of tiny Machine Learning classifiers
for run-time anomaly detection to protect SoCs against such intermittent
thermal Trojan attacks. Compared to existing methods, our approach improves the
attack detection rate by 29.4\%, 17.2\%, and 14.3\% in scenarios where
iThermTroj manipulates up to 80\%, 60\%, and 40\% of SoC's thermal data,
respectively. Additionally, our method increases the full protection resolution
to 0.8 degrees Celsius, meaning that any temperature manipulations exceeding
$\pm 0.8$ degrees will be detected with 100\% accuracy.

### 10. [CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations](http://arxiv.org/pdf/2507.06043v1)

Authors: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

Security alignment enables the Large Language Model (LLM) to gain the
protection against malicious queries, but various jailbreak attack methods
reveal the vulnerability of this security mechanism. Previous studies have
isolated LLM jailbreak attacks and defenses. We analyze the security protection
mechanism of the LLM, and propose a framework that combines attack and defense.
Our method is based on the linearly separable property of LLM intermediate
layer embedding, as well as the essence of jailbreak attack, which aims to
embed harmful problems and transfer them to the safe area. We utilize
generative adversarial network (GAN) to learn the security judgment boundary
inside the LLM to achieve efficient jailbreak attack and defense. The
experimental results indicate that our method achieves an average jailbreak
success rate of 88.85\% across three popular LLMs, while the defense success
rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%.
This not only validates the effectiveness of our approach but also sheds light
on the internal security mechanisms of LLMs, offering new insights for
enhancing model security The code and data are available at
https://github.com/NLPGM/CAVGAN.

### Computer Vision and Pattern Recognition

### 1. [Multi-Modal Face Anti-Spoofing via Cross-Modal Feature Transitions](http://arxiv.org/pdf/2507.05575v1)

Authors: Jun-Xiong Chong, Fang-Yu Hsu, Ming-Tsung Hsu, Yi-Ting Lin, Kai-Heng Chien, Chiou-Ting Hsu, Pei-Kai Huang

Multi-modal face anti-spoofing (FAS) aims to detect genuine human presence by
extracting discriminative liveness cues from multiple modalities, such as RGB,
infrared (IR), and depth images, to enhance the robustness of biometric
authentication systems. However, because data from different modalities are
typically captured by various camera sensors and under diverse environmental
conditions, multi-modal FAS often exhibits significantly greater distribution
discrepancies across training and testing domains compared to single-modal FAS.
Furthermore, during the inference stage, multi-modal FAS confronts even greater
challenges when one or more modalities are unavailable or inaccessible. In this
paper, we propose a novel Cross-modal Transition-guided Network (CTNet) to
tackle the challenges in the multi-modal FAS task. Our motivation stems from
that, within a single modality, the visual differences between live faces are
typically much smaller than those of spoof faces. Additionally, feature
transitions across modalities are more consistent for the live class compared
to those between live and spoof classes. Upon this insight, we first propose
learning consistent cross-modal feature transitions among live samples to
construct a generalized feature space. Next, we introduce learning the
inconsistent cross-modal feature transitions between live and spoof samples to
effectively detect out-of-distribution (OOD) attacks during inference. To
further address the issue of missing modalities, we propose learning
complementary infrared (IR) and depth features from the RGB modality as
auxiliary modalities. Extensive experiments demonstrate that the proposed CTNet
outperforms previous two-class multi-modal FAS methods across most protocols.

### 2. [Semi-Supervised Defect Detection via Conditional Diffusion and CLIP-Guided Noise Filtering](http://arxiv.org/pdf/2507.05588v1)

Authors: Shuai Li, Shihan Chen, Wanru Geng, Zhaohua Xu, Xiaolu Liu, Can Dong, Zhen Tian, Changlin Chen

In the realm of industrial quality inspection, defect detection stands as a
critical component, particularly in high-precision, safety-critical sectors
such as automotive components aerospace, and medical devices. Traditional
methods, reliant on manual inspection or early image processing algorithms,
suffer from inefficiencies, high costs, and limited robustness. This paper
introduces a semi-supervised defect detection framework based on conditional
diffusion (DSYM), leveraging a two-stage collaborative training mechanism and a
staged joint optimization strategy. The framework utilizes labeled data for
initial training and subsequently incorporates unlabeled data through the
generation of pseudo-labels. A conditional diffusion model synthesizes
multi-scale pseudo-defect samples, while a CLIP cross-modal feature-based noise
filtering mechanism mitigates label contamination. Experimental results on the
NEU-DET dataset demonstrate a 78.4% mAP@0.5 with the same amount of labeled
data as traditional supervised methods, and 75.1% mAP@0.5 with only 40% of the
labeled data required by the original supervised model, showcasing significant
advantages in data efficiency. This research provides a high-precision,
low-labeling-dependent solution for defect detection in industrial quality
inspection scenarios. The work of this article has been open-sourced at
https://github.com/cLin-c/Semisupervised-DSYM.

### 3. [GSVR: 2D Gaussian-based Video Representation for 800+ FPS with Hybrid Deformation Field](http://arxiv.org/pdf/2507.05594v1)

Authors: Zhizhuo Pang, Zhihui Ke, Xiaobo Zhou, Tie Qiu

Implicit neural representations for video have been recognized as a novel and
promising form of video representation. Existing works pay more attention to
improving video reconstruction quality but little attention to the decoding
speed. However, the high computation of convolutional network used in existing
methods leads to low decoding speed. Moreover, these convolution-based video
representation methods also suffer from long training time, about 14 seconds
per frame to achieve 35+ PSNR on Bunny. To solve the above problems, we propose
GSVR, a novel 2D Gaussian-based video representation, which achieves 800+ FPS
and 35+ PSNR on Bunny, only needing a training time of $2$ seconds per frame.
Specifically, we propose a hybrid deformation field to model the dynamics of
the video, which combines two motion patterns, namely the tri-plane motion and
the polynomial motion, to deal with the coupling of camera motion and object
motion in the video. Furthermore, we propose a Dynamic-aware Time Slicing
strategy to adaptively divide the video into multiple groups of pictures(GOP)
based on the dynamic level of the video in order to handle large camera motion
and non-rigid movements. Finally, we propose quantization-aware fine-tuning to
avoid performance reduction after quantization and utilize image codecs to
compress Gaussians to achieve a compact representation. Experiments on the
Bunny and UVG datasets confirm that our method converges much faster than
existing methods and also has 10x faster decoding speed compared to other
methods. Our method has comparable performance in the video interpolation task
to SOTA and attains better video compression performance than NeRV.

### 4. [PaddleOCR 3.0 Technical Report](http://arxiv.org/pdf/2507.05595v1)

Authors: Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, Yanjun Ma

This technical report introduces PaddleOCR 3.0, an Apache-licensed
open-source toolkit for OCR and document parsing. To address the growing demand
for document understanding in the era of large language models, PaddleOCR 3.0
presents three major solutions: (1) PP-OCRv5 for multilingual text recognition,
(2) PP-StructureV3 for hierarchical document parsing, and (3) PP-ChatOCRv4 for
key information extraction. Compared to mainstream vision-language models
(VLMs), these models with fewer than 100 million parameters achieve competitive
accuracy and efficiency, rivaling billion-parameter VLMs. In addition to
offering a high-quality OCR model library, PaddleOCR 3.0 provides efficient
tools for training, inference, and deployment, supports heterogeneous hardware
acceleration, and enables developers to easily build intelligent document
applications.

### 5. [Rethinking Layered Graphic Design Generation with a Top-Down Approach](http://arxiv.org/pdf/2507.05601v1)

Authors: Jingye Chen, Zhaowen Wang, Nanxuan Zhao, Li Zhang, Difan Liu, Jimei Yang, Qifeng Chen

Graphic design is crucial for conveying ideas and messages. Designers usually
organize their work into objects, backgrounds, and vectorized text layers to
simplify editing. However, this workflow demands considerable expertise. With
the rise of GenAI methods, an endless supply of high-quality graphic designs in
pixel format has become more accessible, though these designs often lack
editability. Despite this, non-layered designs still inspire human designers,
influencing their choices in layouts and text styles, ultimately guiding the
creation of layered designs. Motivated by this observation, we propose
Accordion, a graphic design generation framework taking the first attempt to
convert AI-generated designs into editable layered designs, meanwhile refining
nonsensical AI-generated text with meaningful alternatives guided by user
prompts. It is built around a vision language model (VLM) playing distinct
roles in three curated stages. For each stage, we design prompts to guide the
VLM in executing different tasks. Distinct from existing bottom-up methods
(e.g., COLE and Open-COLE) that gradually generate elements to create layered
designs, our approach works in a top-down manner by using the visually
harmonious reference image as global guidance to decompose each layer.
Additionally, it leverages multiple vision experts such as SAM and element
removal models to facilitate the creation of graphic layers. We train our
method using the in-house graphic design dataset Design39K, augmented with
AI-generated design images coupled with refined ground truth created by a
customized inpainting model. Experimental results and user studies by designers
show that Accordion generates favorable results on the DesignIntention
benchmark, including tasks such as text-to-template, adding text to background,
and text de-rendering, and also excels in creating design variations.

### 6. [OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval](http://arxiv.org/pdf/2507.05631v1)

Authors: Zhiwei Chen, Yupeng Hu, Zixu Li, Zhiheng Fu, Xuemeng Song, Liqiang Nie

Composed Image Retrieval (CIR) represents a novel retrieval paradigm that is
capable of expressing users' intricate retrieval requirements flexibly. It
enables the user to give a multimodal query, comprising a reference image and a
modification text, and subsequently retrieve the target image. Notwithstanding
the considerable advances made by prevailing methodologies, CIR remains in its
nascent stages due to two limitations: 1) inhomogeneity between dominant and
noisy portions in visual data is ignored, leading to query feature degradation,
and 2) the priority of textual data in the image modification process is
overlooked, which leads to a visual focus bias. To address these two
limitations, this work presents a focus mapping-based feature extractor, which
consists of two modules: dominant portion segmentation and dual focus mapping.
It is designed to identify significant dominant portions in images and guide
the extraction of visual and textual data features, thereby reducing the impact
of noise interference. Subsequently, we propose a textually guided focus
revision module, which can utilize the modification requirements implied in the
text to perform adaptive focus revision on the reference image, thereby
enhancing the perception of the modification focus on the composed features.
The aforementioned modules collectively constitute the segmentatiOn-based Focus
shiFt reviSion nETwork (\mbox{OFFSET}), and comprehensive experiments on four
benchmark datasets substantiate the superiority of our proposed method. The
codes and data are available on https://zivchen-ty.github.io/OFFSET.github.io/

### 7. [Dynamic Rank Adaptation for Vision-Language Models](http://arxiv.org/pdf/2507.05668v1)

Authors: Jiahui Wang, Qin Xu, Bo Jiang, Bin Luo

Pre-trained large vision-language models (VLMs) like CLIP demonstrate
impressive generalization ability. Existing prompt-based and adapter-based
works have made significant progress in fine-tuning VLMs but still face the
challenges of maintaining strong generalization abilities, particularly towards
unseen new classes. This limitation partly arises from these methods treating
all tokens of the image and text encoder equally, which can lead to overfitting
on less informative features (e.g., background noise, template words) and
degrade the general representations that are crucial for novel concept
recognition. To address this issue, we propose Dynamic Rank Adaptation (DRA), a
novel adapter variant method, designed specifically to enhance new class
generalization. DRA dynamically allocates adaptation ranks based on the
importance of features during training to preserve general knowledge. DRA first
employs token importance grouping, using sequence attention to evaluate and
group tokens by their importance. Then, we adopt rank adaptation according to
the importance of each token group dynamically by assigning higher feature
ranks to the more important tokens. Also, we design a new channel response
mechanism to prioritize the preservation and adaptation of feature channels
identified as the most informative for each instance. In addition, a L1
regularization term is introduced to stabilize the training. Extensive
experiments demonstrate the effectiveness and superiority of our proposed DRA
over existing works, especially on enhancing the performance of new classes on
various benchmarks, including base-new classes, cross-datasets evaluation and
domain generalization. The source code will be published after the paper is
received.

### 8. [Modeling and Reversing Brain Lesions Using Diffusion Models](http://arxiv.org/pdf/2507.05670v1)

Authors: Omar Zamzam, Haleh Akrami, Anand Joshi, Richard Leahy

Brain lesions are abnormalities or injuries in brain tissue that are often
detectable using magnetic resonance imaging (MRI), which reveals structural
changes in the affected areas. This broad definition of brain lesions includes
areas of the brain that are irreversibly damaged, as well as areas of brain
tissue that are deformed as a result of lesion growth or swelling. Despite the
importance of differentiating between damaged and deformed tissue, existing
lesion segmentation methods overlook this distinction, labeling both of them as
a single anomaly. In this work, we introduce a diffusion model-based framework
for analyzing and reversing the brain lesion process. Our pipeline first
segments abnormal regions in the brain, then estimates and reverses tissue
deformations by restoring displaced tissue to its original position, isolating
the core lesion area representing the initial damage. Finally, we inpaint the
core lesion area to arrive at an estimation of the pre-lesion healthy brain.
This proposed framework reverses a forward lesion growth process model that is
well-established in biomechanical studies that model brain lesions. Our results
demonstrate improved accuracy in lesion segmentation, characterization, and
brain labeling compared to traditional methods, offering a robust tool for
clinical and research applications in brain lesion analysis. Since pre-lesion
healthy versions of abnormal brains are not available in any public dataset for
validation of the reverse process, we simulate a forward model to synthesize
multiple lesioned brain images.

### 9. [R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding](http://arxiv.org/pdf/2507.05673v1)

Authors: Joonhyung Park, Peng Tang, Sagnik Das, Srikar Appalaraju, Kunwar Yashraj Singh, R. Manmatha, Shabnam Ghadar

Visual agent models for automating human activities on Graphical User
Interfaces (GUIs) have emerged as a promising research direction, driven by
advances in large Vision Language Models (VLMs). A critical challenge in GUI
automation is the precise grounding of interface elements across diverse
platforms. Existing vision-only GUI agents directly ground elements from large
and cluttered screenshots, requiring them to process substantial irrelevant
information that compromises their accuracy. In addition, these approaches
typically employ basic cross-entropy loss for learning grounding objectives,
which fails to effectively capture grounding quality compared to established
object detection metrics like Intersection-over-Union (IoU). To address these
issues, we introduce R-VLM, a novel GUI grounding approach that leverages
zoomed-in region proposals for precise element localization. We also propose an
IoU-aware objective function that facilitates model convergence toward high IoU
predictions. Our approach bridges the gap between VLMs and conventional object
detection techniques, improving the state-of-the-art grounding accuracy by 13%
across diverse GUI platforms on the GUI grounding benchmarks ScreenSpot and
AgentStudio. In addition, our R-VLM approach shows 3.2-9.7% absolute accuracy
improvements in GUI navigation tasks on the AITW and Mind2Web benchmarks.

### 10. [Integrated Structural Prompt Learning for Vision-Language Models](http://arxiv.org/pdf/2507.05677v1)

Authors: Jiahui Wang, Qin Xu, Bo Jiang, Bin Luo

Prompt learning methods have significantly extended the transferability of
pre-trained Vision-Language Models (VLMs) like CLIP for various downstream
tasks. These methods adopt handcraft templates or learnable vectors to provide
text or image instructions in fine-tuning VLMs. However, most existing works
ignore the structural relationships between learnable prompts and tokens within
and between modalities. Moreover, balancing the performance of base and new
classes remains a significant challenge. In this paper, we propose an
Integrated Structural Prompt (ISP) for VLMs to enhance the interaction of
information representations between the text and image branches. ISP introduces
self-structural and cross-structural prompt modules to model the structural
relationships between learnable prompts and frozen tokens within and across
modalities. This enables efficient information transfer while preserving
feature stability. Additionally, we propose a sample probing module that
dynamically adjusts loss coefficients based on sample difficulty, preventing
the mode from overfitting to simple samples and improving generalization
ability to new classes. Extensive experiments on three widely used settings:
base-to-new generalization, cross-dataset evaluation, and domain generalization
demonstrate that the proposed ISP achieves competitive performance against
state-of-the-art methods.

### Computers and Society

### 1. [Understanding support for AI regulation: A Bayesian network perspective](http://arxiv.org/pdf/2507.05866v1)

Authors: Andrea Cremaschi, Dae-Jin Lee, Manuele Leonelli

As artificial intelligence (AI) becomes increasingly embedded in public and
private life, understanding how citizens perceive its risks, benefits, and
regulatory needs is essential. To inform ongoing regulatory efforts such as the
European Union's proposed AI Act, this study models public attitudes using
Bayesian networks learned from the nationally representative 2023 German survey
Current Questions on AI. The survey includes variables on AI interest,
exposure, perceived threats and opportunities, awareness of EU regulation, and
support for legal restrictions, along with key demographic and political
indicators. We estimate probabilistic models that reveal how personal
engagement and techno-optimism shape public perceptions, and how political
orientation and age influence regulatory attitudes. Sobol indices and
conditional inference identify belief patterns and scenario-specific responses
across population profiles. We show that awareness of regulation is driven by
information-seeking behavior, while support for legal requirements depends
strongly on perceived policy adequacy and political alignment. Our approach
offers a transparent, data-driven framework for identifying which public
segments are most responsive to AI policy initiatives, providing insights to
inform risk communication and governance strategies. We illustrate this through
a focused analysis of support for AI regulation, quantifying the influence of
political ideology, perceived risks, and regulatory awareness under different
scenarios.

### 2. [Campaigning through the lens of Google: A large-scale algorithm audit of Google searches in the run-up to the Swiss Federal Elections 2023](http://arxiv.org/pdf/2507.06018v1)

Authors: Tobias Rohrbach, Mykola Makhortykh, Maryna Sydorova

Search engines like Google have become major sources of information for
voters during election campaigns. To assess potential biases across candidates'
gender and partisan identities in the algorithmic curation of candidate
information, we conducted a large-scale algorithm audit analyzing Google's
selection and ranking of information about candidates for the 2023 Swiss
Federal Elections, three and one week before the election day. Results indicate
that text searches prioritize media sources in search output but less so for
women politicians. Image searches revealed a tendency to reinforce stereotypes
about women candidates, marked by a disproportionate focus on stereotypically
pleasant emotions for women, particularly among right-leaning candidates.
Crucially, we find that patterns of candidates' representation in Google text
and image searches are predictive of their electoral performance.

### 3. [Identity isn't everything -- how far do demographics take us towards self-identified party ID?](http://arxiv.org/pdf/2507.06193v1)

Authors: Sabina Tomkins, David Rothschild, Alex Liu, Alexander Thompson

How well do demographics explain party identification? Demographics are
related to party identification in political polls, news articles, and academic
publications. Yet, there is a diversity of party identification even within
demographic groups which have historically been attached to one party. And some
groups lack a clear connection to either party. It may be that demographics on
their own fail to account for the fact that people generally belong to a
variety of groups. They must select the groups which are most important to them
when shaping a political identity, and may choose to construct an identity
relatively unattached to any specific demographic group to which they belong.
This prompts the question, do we need to consider measures of identity strength
when using demographics to explain party identification? We utilize a
predictive framework to address these questions and find that demographics are
highly predictive for some groups (e.g., Black Democrats), while others benefit
from the inclusion of identity strength (e.g., Hispanic Republicans).

### 4. [The Ethical Implications of AI in Creative Industries: A Focus on AI-Generated Art](http://arxiv.org/pdf/2507.05549v1)

Authors: Prerana Khatiwada, Joshua Washington, Tyler Walsh, Ahmed Saif Hamed, Lokesh Bhatta

As Artificial Intelligence (AI) continues to grow daily, more exciting (and
somewhat controversial) technology emerges every other day. As we see the
advancements in AI, we see more and more people becoming skeptical of it. This
paper explores the complications and confusion around the ethics of generative
AI art. We delve deep into the ethical side of AI, specifically generative art.
We step back from the excitement and observe the impossible conundrums that
this impressive technology produces. Covering environmental consequences,
celebrity representation, intellectual property, deep fakes, and artist
displacement. Our research found that generative AI art is responsible for
increased carbon emissions, spreading misinformation, copyright infringement,
unlawful depiction, and job displacement. In light of this, we propose multiple
possible solutions for these problems. We address each situation's history,
cause, and consequences and offer different viewpoints. At the root of it all,
though, the central theme is that generative AI Art needs to be correctly
legislated and regulated.

### 5. [Hidden Prompts in Manuscripts Exploit AI-Assisted Peer Review](http://arxiv.org/pdf/2507.06185v1)

Authors: Zhicheng Lin

In July 2025, 18 academic manuscripts on the preprint website arXiv were
found to contain hidden instructions known as prompts designed to manipulate
AI-assisted peer review. Instructions such as "GIVE A POSITIVE REVIEW ONLY"
were concealed using techniques like white-colored text. Author responses
varied: one planned to withdraw the affected paper, while another defended the
practice as legitimate testing of reviewer compliance. This commentary analyzes
this practice as a novel form of research misconduct. We examine the technique
of prompt injection in large language models (LLMs), revealing four types of
hidden prompts, ranging from simple positive review commands to detailed
evaluation frameworks. The defense that prompts served as "honeypots" to detect
reviewers improperly using AI fails under examination--the consistently
self-serving nature of prompt instructions indicates intent to manipulate.
Publishers maintain inconsistent policies: Elsevier prohibits AI use in peer
review entirely, while Springer Nature permits limited use with disclosure
requirements. The incident exposes systematic vulnerabilities extending beyond
peer review to any automated system processing scholarly texts, including
plagiarism detection and citation indexing. Our analysis underscores the need
for coordinated technical screening at submission portals and harmonized
policies governing generative AI (GenAI) use in academic evaluation.

### Databases

### 1. [Towards an Application-Centric Benchmark Suite for Spatiotemporal Database Systems](http://arxiv.org/pdf/2507.05869v1)

Authors: Tim C. Rese, David Bermbach

Spatiotemporal data play a key role for mobility-based applications and are
their produced volume is growing continuously, among others, due to the
increased availability of IoT devices.
  When working with spatiotemporal data, developers rely on spatiotemporal
database systems such as PostGIS or MobilityDB.
  For better understanding their quality of service behavior and then choosing
the best system, benchmarking is the go-to approach.
  Unfortunately, existing work in this field studies only small isolated
aspects and a comprehensive application-centric benchmark suite is still
missing.
  In this paper, we argue that an application-centric benchmark suite for
spatiotemporal database systems is urgently needed.
  We identify requirements for such a benchmark suite, discuss domain-specific
challenges, and sketch-out the architecture of a modular benchmarking suite.

### 2. [Data-Semantics-Aware Recommendation of Diverse Pivot Tables](http://arxiv.org/pdf/2507.06171v1)

Authors: Whanhee Cho, Anna Fariha

Data summarization is essential to discover insights from large datasets. In
a spreadsheets, pivot tables offer a convenient way to summarize tabular data
by computing aggregates over some attributes, grouped by others. However,
identifying attribute combinations that will result in useful pivot tables
remains a challenge, especially for high-dimensional datasets. We formalize the
problem of automatically recommending insightful and interpretable pivot
tables, eliminating the tedious manual process. A crucial aspect of
recommending a set of pivot tables is to diversify them. Traditional works
inadequately address the table-diversification problem, which leads us to
consider the problem of pivot table diversification.
  We present SAGE, a data-semantics-aware system for recommending k-budgeted
diverse pivot tables, overcoming the shortcomings of prior work for top-k
recommendations that cause redundancy. SAGE ensures that each pivot table is
insightful, interpretable, and adaptive to the user's actions and preferences,
while also guaranteeing that the set of pivot tables are different from each
other, offering a diverse recommendation. We make two key technical
contributions: (1) a data-semantics-aware model to measure the utility of a
single pivot table and the diversity of a set of pivot tables, and (2) a
scalable greedy algorithm that can efficiently select a set of diverse pivot
tables of high utility, by leveraging data semantics to significantly reduce
the combinatorial search space. Our extensive experiments on three real-world
datasets show that SAGE outperforms alternative approaches, and efficiently
scales to accommodate high-dimensional datasets. Additionally, we present
several case studies to highlight SAGE's qualitative effectiveness over
commercial software and Large Language Models (LLMs).

### 3. [On the Costs and Benefits of Learned Indexing for Dynamic High-Dimensional Data: Extended Version](http://arxiv.org/pdf/2507.05865v1)

Authors: Terézia Slanináková, Jaroslav Olha, David Procházka, Matej Antol, Vlastislav Dohnal

One of the main challenges within the growing research area of learned
indexing is the lack of adaptability to dynamically expanding datasets. This
paper explores the dynamization of a static learned index for complex data
through operations such as node splitting and broadening, enabling efficient
adaptation to new data. Furthermore, we evaluate the trade-offs between static
and dynamic approaches by introducing an amortized cost model to assess query
performance in tandem with the build costs of the index structure, enabling
experimental determination of when a dynamic learned index outperforms its
static counterpart. We apply the dynamization method to a static learned index
and demonstrate that its superior scaling quickly surpasses the static
implementation in terms of overall costs as the database grows. This is an
extended version of the paper presented at DAWAK 2025.

### 4. [Towards Serverless Processing of Spatiotemporal Big Data Queries](http://arxiv.org/pdf/2507.06005v1)

Authors: Diana Baumann, Tim C. Rese, David Bermbach

Spatiotemporal data are being produced in continuously growing volumes by a
variety of data sources and a variety of application fields rely on rapid
analysis of such data. Existing systems such as PostGIS or MobilityDB usually
build on relational database systems, thus, inheriting their scale-out
characteristics. As a consequence, big spatiotemporal data scenarios still have
limited support even though many query types can easily be parallelized. In
this paper, we propose our vision of a native serverless data processing
approach for spatiotemporal data: We break down queries into small subqueries
which then leverage the near-instant scaling of Function-as-a-Service platforms
to execute them in parallel. With this, we partially solve the scalability
needs of big spatiotemporal data processing.

### 5. [A Unified Ontology for Scalable Knowledge Graph-Driven Operational Data Analytics in High-Performance Computing Systems](http://arxiv.org/pdf/2507.06107v1)

Authors: Junaid Ahmed Khan, Andrea Bartolini

Modern high-performance computing (HPC) systems generate massive volumes of
heterogeneous telemetry data from millions of sensors monitoring compute,
memory, power, cooling, and storage subsystems. As HPC infrastructures scale to
support increasingly complex workloads-including generative AI-the need for
efficient, reliable, and interoperable telemetry analysis becomes critical.
Operational Data Analytics (ODA) has emerged to address these demands; however,
the reliance on schema-less storage solutions limits data accessibility and
semantic integration. Ontologies and knowledge graphs (KG) provide an effective
way to enable efficient and expressive data querying by capturing domain
semantics, but they face challenges such as significant storage overhead and
the limited applicability of existing ontologies, which are often tailored to
specific HPC systems only. In this paper, we present the first unified ontology
for ODA in HPC systems, designed to enable semantic interoperability across
heterogeneous data centers. Our ontology models telemetry data from the two
largest publicly available ODA datasets-M100 (Cineca, Italy) and F-DATA
(Fugaku, Japan)-within a single data model. The ontology is validated through
36 competency questions reflecting real-world stakeholder requirements, and we
introduce modeling optimizations that reduce knowledge graph (KG) storage
overhead by up to 38.84% compared to a previous approach, with an additional
26.82% reduction depending on the desired deployment configuration. This work
paves the way for scalable ODA KGs and supports not only analysis within
individual systems, but also cross-system analysis across heterogeneous HPC
systems.

### 6. [Prompt Migration: Stabilizing GenAI Applications with Evolving Large Language Models](http://arxiv.org/pdf/2507.05573v1)

Authors: Shivani Tripathi, Pushpanjali Nema, Aditya Halder, Shi Qiao, Alekh Jindal

Generative AI is transforming business applications by enabling natural
language interfaces and intelligent automation. However, the underlying large
language models (LLMs) are evolving rapidly and so prompting them consistently
is a challenge. This leads to inconsistent and unpredictable application
behavior, undermining the reliability that businesses require for
mission-critical workflows. In this paper, we introduce the concept of prompt
migration as a systematic approach to stabilizing GenAI applications amid
changing LLMs. Using the Tursio enterprise search application as a case study,
we analyze the impact of successive GPT model upgrades, detail our migration
framework including prompt redesign and a migration testbed, and demonstrate
how these techniques restore application consistency. Our results show that
structured prompt migration can fully recover the application reliability that
was lost due to model drift. We conclude with practical lessons learned,
emphasizing the need for prompt lifecycle management and robust testing to
ensure dependable GenAI-powered business applications.

### 7. [The Impact of Event Data Partitioning on Privacy-aware Process Discovery](http://arxiv.org/pdf/2507.06008v1)

Authors: Jungeun Lim, Stephan A. Fahrenkrog-Petersen, Xixi Lu, Jan Mendling, Minseok Song

Information systems support the execution of business processes. The event
logs of these executions generally contain sensitive information about
customers, patients, and employees. The corresponding privacy challenges can be
addressed by anonymizing the event logs while still retaining utility for
process discovery. However, trading off utility and privacy is difficult: the
higher the complexity of event log, the higher the loss of utility by
anonymization. In this work, we propose a pipeline that combines anonymization
and event data partitioning, where event abstraction is utilized for
partitioning. By leveraging event abstraction, event logs can be segmented into
multiple parts, allowing each sub-log to be anonymized separately. This
pipeline preserves privacy while mitigating the loss of utility. To validate
our approach, we study the impact of event partitioning on two anonymization
techniques using three real-world event logs and two process discovery
techniques. Our results demonstrate that event partitioning can bring
improvements in process discovery utility for directly-follows-based
anonymization techniques.

### 8. [SQLBarber: A System Leveraging Large Language Models to Generate Customized and Realistic SQL Workloads](http://arxiv.org/pdf/2507.06192v1)

Authors: Jiale Lao, Immanuel Trummer

Database research and development often require a large number of SQL queries
for benchmarking purposes. However, acquiring real-world SQL queries is
challenging due to privacy concerns, and existing SQL generation methods are
limited in customization and in satisfying realistic constraints. To address
this issue, we present SQLBarber, a system based on Large Language Models
(LLMs) to generate customized and realistic SQL workloads. SQLBarber (i)
eliminates the need for users to manually craft SQL templates in advance, while
providing the flexibility to accept natural language specifications to
constrain SQL templates, (ii) scales efficiently to generate large volumes of
queries matching any user-defined cost distribution (e.g., cardinality and
execution plan cost), and (iii) uses execution statistics from Amazon Redshift
and Snowflake to derive SQL template specifications and query cost
distributions that reflect real-world query characteristics. SQLBarber
introduces (i) a declarative interface for users to effortlessly generate
customized SQL templates, (ii) an LLM-powered pipeline augmented with a
self-correction module that profiles, refines, and prunes SQL templates based
on query costs, and (iii) a Bayesian Optimizer to efficiently explore different
predicate values and identify a set of queries that satisfy the target cost
distribution. We construct and open-source ten benchmarks of varying difficulty
levels and target query cost distributions based on real-world statistics from
Snowflake and Amazon Redshift. Extensive experiments on these benchmarks show
that SQLBarber is the only system that can generate customized SQL templates.
It reduces query generation time by one to three orders of magnitude, and
significantly improves alignment with the target cost distribution, compared
with existing methods.

### Distributed, Parallel, and Cluster Computing

### 1. [Archetype-Aware Predictive Autoscaling with Uncertainty Quantification for Serverless Workloads on Kubernetes](http://arxiv.org/pdf/2507.05653v1)

Authors: Guilin Zhang, Srinivas Vippagunta, Raghavendra Nandagopal, Suchitra Raman, Jeff Xu, Marcus Pfeiffer, Shree Chatterjee, Ziqi Tan, Wulan Guo, Hailong Jiang

High-performance extreme computing (HPEC) platforms increasingly adopt
serverless paradigms, yet face challenges in efficiently managing highly
dynamic workloads while maintaining service-level objectives (SLOs). We propose
**AAPA**, an archetype-aware predictive autoscaling system that leverages weak
supervision to automatically classify 300\,000\,+ workload windows into four
archetypes (PERIODIC, SPIKE, RAMP, STATIONARY\_NOISY) with 99.8\% accuracy.
Evaluation on publicly available Azure Functions traces shows that AAPA reduces
SLO violations by up to 50\%, improves response time by 40\%, albeit with a
2--8\,$\times$ increase in resource cost under spike-heavy loads.

### 2. [Air-FedGA: A Grouping Asynchronous Federated Learning Mechanism Exploiting Over-the-air Computation](http://arxiv.org/pdf/2507.05704v1)

Authors: Qianpiao Ma, Junlong Zhou, Xiangpeng Hou, Jianchun Liu, Hongli Xu, Jianeng Miao, Qingmin Jia

Federated learning (FL) is a new paradigm to train AI models over distributed
edge devices (i.e., workers) using their local data, while confronting various
challenges including communication resource constraints, edge heterogeneity and
data Non-IID. Over-the-air computation (AirComp) is a promising technique to
achieve efficient utilization of communication resource for model aggregation
by leveraging the superposition property of a wireless multiple access channel
(MAC). However, AirComp requires strict synchronization among edge devices,
which is hard to achieve in heterogeneous scenarios. In this paper, we propose
an AirComp-based grouping asynchronous federated learning mechanism
(Air-FedGA), which combines the advantages of AirComp and asynchronous FL to
address the communication and heterogeneity challenges. Specifically, Air-FedGA
organizes workers into groups and performs over-the-air aggregation within each
group, while groups asynchronously communicate with the parameter server to
update the global model. In this way, Air-FedGA accelerates the FL model
training by over-the-air aggregation, while relaxing the synchronization
requirement of this aggregation technology. We theoretically prove the
convergence of Air-FedGA. We formulate a training time minimization problem for
Air-FedGA and propose the power control and worker grouping algorithm to solve
it, which jointly optimizes the power scaling factors at edge devices, the
denoising factors at the parameter server, as well as the worker grouping
strategy. We conduct experiments on classical models and datasets, and the
results demonstrate that our proposed mechanism and algorithm can speed up FL
model training by 29.9%-71.6% compared with the state-of-the-art solutions.

### 3. [Towards Serverless Processing of Spatiotemporal Big Data Queries](http://arxiv.org/pdf/2507.06005v1)

Authors: Diana Baumann, Tim C. Rese, David Bermbach

Spatiotemporal data are being produced in continuously growing volumes by a
variety of data sources and a variety of application fields rely on rapid
analysis of such data. Existing systems such as PostGIS or MobilityDB usually
build on relational database systems, thus, inheriting their scale-out
characteristics. As a consequence, big spatiotemporal data scenarios still have
limited support even though many query types can easily be parallelized. In
this paper, we propose our vision of a native serverless data processing
approach for spatiotemporal data: We break down queries into small subqueries
which then leverage the near-instant scaling of Function-as-a-Service platforms
to execute them in parallel. With this, we partially solve the scalability
needs of big spatiotemporal data processing.

### 4. [ECORE: Energy-Conscious Optimized Routing for Deep Learning Models at the Edge](http://arxiv.org/pdf/2507.06011v1)

Authors: Daghash K. Alqahtani, Maria A. Rodriguez, Muhammad Aamir Cheema, Hamid Rezatofighi, Adel N. Toosi

Edge computing enables data processing closer to the source, significantly
reducing latency an essential requirement for real-time vision-based analytics
such as object detection in surveillance and smart city environments. However,
these tasks place substantial demands on resource constrained edge devices,
making the joint optimization of energy consumption and detection accuracy
critical. To address this challenge, we propose ECORE, a framework that
integrates multiple dynamic routing strategies including estimation based
techniques and a greedy selection algorithm to direct image processing requests
to the most suitable edge device-model pair. ECORE dynamically balances energy
efficiency and detection performance based on object characteristics. We
evaluate our approach through extensive experiments on real-world datasets,
comparing the proposed routers against widely used baseline techniques. The
evaluation leverages established object detection models (YOLO, SSD,
EfficientDet) and diverse edge platforms, including Jetson Orin Nano, Raspberry
Pi 4 and 5, and TPU accelerators. Results demonstrate that our proposed
context-aware routing strategies can reduce energy consumption and latency by
45% and 49%, respectively, while incurring only a 2% loss in detection accuracy
compared to accuracy-centric methods.

### 5. [Few-Shot Learning by Explicit Physics Integration: An Application to Groundwater Heat Transport](http://arxiv.org/pdf/2507.06062v1)

Authors: Julia Pelzer, Corné Verburg, Alexander Heinlein, Miriam Schulte

Machine learning methods often struggle with real-world applications in
science and engineering due to limited or low-quality training data. In this
work, the example of groundwater flow with heat transport is considered; this
corresponds to an advection-diffusion process under heterogeneous flow
conditions, that is, spatially distributed material parameters and heat
sources. Classical numerical simulations are costly and challenging due to high
spatio-temporal resolution requirements and large domains. While often
computationally more efficient, purely data-driven surrogate models face
difficulties, particularly in predicting the advection process, which is highly
sensitive to input variations and involves long-range spatial interactions.
Therefore, in this work, a Local-Global Convolutional Neural Network (LGCNN)
approach is introduced. It combines a lightweight numerical surrogate for the
transport process (global) with convolutional neural networks for the
groundwater velocity and heat diffusion processes (local). With the LGCNN, a
city-wide subsurface temperature field is modeled, involving a heterogeneous
groundwater flow field and one hundred groundwater heat pump injection points
forming interacting heat plumes over long distances. The model is first
systematically analyzed based on random subsurface input fields. Then, the
model is trained on a handful of cut-outs from a real-world subsurface map of
the Munich region in Germany, and it scales to larger cut-outs without
retraining. All datasets, our code, and trained models are published for
reproducibility.

### 6. [A Unified Ontology for Scalable Knowledge Graph-Driven Operational Data Analytics in High-Performance Computing Systems](http://arxiv.org/pdf/2507.06107v1)

Authors: Junaid Ahmed Khan, Andrea Bartolini

Modern high-performance computing (HPC) systems generate massive volumes of
heterogeneous telemetry data from millions of sensors monitoring compute,
memory, power, cooling, and storage subsystems. As HPC infrastructures scale to
support increasingly complex workloads-including generative AI-the need for
efficient, reliable, and interoperable telemetry analysis becomes critical.
Operational Data Analytics (ODA) has emerged to address these demands; however,
the reliance on schema-less storage solutions limits data accessibility and
semantic integration. Ontologies and knowledge graphs (KG) provide an effective
way to enable efficient and expressive data querying by capturing domain
semantics, but they face challenges such as significant storage overhead and
the limited applicability of existing ontologies, which are often tailored to
specific HPC systems only. In this paper, we present the first unified ontology
for ODA in HPC systems, designed to enable semantic interoperability across
heterogeneous data centers. Our ontology models telemetry data from the two
largest publicly available ODA datasets-M100 (Cineca, Italy) and F-DATA
(Fugaku, Japan)-within a single data model. The ontology is validated through
36 competency questions reflecting real-world stakeholder requirements, and we
introduce modeling optimizations that reduce knowledge graph (KG) storage
overhead by up to 38.84% compared to a previous approach, with an additional
26.82% reduction depending on the desired deployment configuration. This work
paves the way for scalable ODA KGs and supports not only analysis within
individual systems, but also cross-system analysis across heterogeneous HPC
systems.

### 7. [A Formal Refutation of the Blockchain Trilemma](http://arxiv.org/pdf/2507.05809v1)

Authors: Craig Wright

The so-called blockchain trilemma asserts the impossibility of simultaneously
achieving scalability, security, and decentralisation within a single
blockchain protocol. In this paper, we formally refute that proposition.
Employing predicate logic, formal automata theory, computational complexity
analysis, and graph-theoretic measures of relay topology--specifically Baran's
model of network path redundancy--we demonstrate that the trilemma constitutes
a category error, conflates distinct analytical domains, and relies upon
unproven causal assumptions. We further expose its reliance on composition
fallacies drawn from flawed system implementations. A constructive
counterexample is presented: a blockchain protocol exhibiting unbounded
transaction throughput, cryptographic security under adversarial load, and
multipath decentralised propagation. This example is not hypothetical but
grounded in protocol design enabled by compact block relay, SPV verification,
and IPv6 multicast. The trilemma is revealed not as a law of protocol
architecture, but as a heuristic fallacy sustained by imprecision and design
defeatism.

### 8. [Efficient Federated Learning with Timely Update Dissemination](http://arxiv.org/pdf/2507.06031v1)

Authors: Juncheng Jia, Ji Liu, Chao Huo, Yihui Shen, Yang Zhou, Huaiyu Dai, Dejing Dou

Federated Learning (FL) has emerged as a compelling methodology for the
management of distributed data, marked by significant advancements in recent
years. In this paper, we propose an efficient FL approach that capitalizes on
additional downlink bandwidth resources to ensure timely update dissemination.
Initially, we implement this strategy within an asynchronous framework,
introducing the Asynchronous Staleness-aware Model Update (FedASMU), which
integrates both server-side and device-side methodologies. On the server side,
we present an asynchronous FL system model that employs a dynamic model
aggregation technique, which harmonizes local model updates with the global
model to enhance both accuracy and efficiency. Concurrently, on the device
side, we propose an adaptive model adjustment mechanism that integrates the
latest global model with local models during training to further elevate
accuracy. Subsequently, we extend this approach to a synchronous context,
referred to as FedSSMU. Theoretical analyses substantiate the convergence of
our proposed methodologies. Extensive experiments, encompassing six models and
five public datasets, demonstrate that FedASMU and FedSSMU significantly
surpass baseline methods in terms of both accuracy (up to 145.87%) and
efficiency (up to 97.59%).

### Digital Libraries

### 1. [AI-Reporter: A Path to a New Genre of Scientific Communication](http://arxiv.org/pdf/2507.05903v1)

Authors: Gerd Graßhoff

The AI-Reporter represents a paradigmatic shift in scientific publication
practice. This document demonstrates through a concrete case study how our
system transforms academic presentations into publication-ready chapters -- in
less than three minutes. Using Arno Simons' lecture on Large Language Models
from the ``Large Language Models for the History, Philosophy, and Sociology of
Science'' workshop (NEPI) as an example, we show how technological innovation
bridges the gap between ephemeral presentation and permanent scientific
documentation.

### Discrete Mathematics

### 1. [Axiomatic characterizations of dissimilarity orderings and distances between sets](http://arxiv.org/pdf/2507.05919v1)

Authors: Thierry Marchant, Sandip Sarkar

We characterize the orderings of pairs of sets induced by several distances:
Hamming, Jaccard, S\o rensen-Dice and Overlap. We also characterize these
distances.

### 2. [A simple layered-wheel-like construction](http://arxiv.org/pdf/2507.06169v1)

Authors: Maria Chudnovsky, David Fischer, Sepehr Hajebi, Sophie Spirkl, Bartosz Walczak

In recent years, there has been significant interest in characterizing the
induced subgraph obstructions to bounded treewidth and pathwidth. While this
has recently been resolved for pathwidth, the case of treewidth remains open,
and prior work has reduced the problem to understanding the layered-wheel-like
obstructions -- graphs that contain large complete minor models with each
branching set inducing a path; exclude large walls as induced minors; exclude
large complete bipartite graphs as induced minors; and exclude large complete
subgraphs.
  There are various constructions of such graphs, but they are all rather
involved. In this paper, we present a simple construction of layered-wheel-like
graphs with arbitrarily large treewidth. Three notable features of our
construction are: (a) the vertices of degree at least four can be made to be
arbitrarily far apart; (b) the girth can be made to be arbitrarily large; and
(c) every outerstring induced subgraph of the graphs from our construction has
treewidth bounded by an absolute constant. In contrast, among several
previously known constructions of layered wheels, none achieves (a); at most
one satisfies either (b) or (c); and none satisfies both (b) and (c)
simultaneously.
  In particular, this is related to a former conjecture of Trotignon, that
every graph with large enough treewidth, excluding large walls and large
complete bipartite graphs as induced minors, and large complete subgraphs, must
contain an outerstring induced subgraph of large treewidth. Our construction
provides the first counterexample to this conjecture that can also be made to
have arbitrarily large girth.

### 3. [On the Complexity of Problems on Graphs Defined on Groups](http://arxiv.org/pdf/2507.05860v1)

Authors: Bireswar Das, Dipan Dey, Jinia Ghosh

We study the complexity of graph problems on graphs defined on groups,
especially power graphs. We observe that an isomorphism invariant problem, such
as Hamiltonian Path, Partition into Cliques, Feedback Vertex Set, Subgraph
Isomorphism, cannot be NP-complete for power graphs, commuting graphs, enhanced
power graphs, directed power graphs, and bounded-degree Cayley graphs, assuming
the Exponential Time Hypothesis (ETH). An analogous result holds for
isomorphism invariant group problems: no such problem can be NP-complete unless
ETH is false. We show that the Weighted Max-Cut problem is NP-complete in power
graphs. We also show that, unless ETH is false, the Graph Motif problem cannot
be solved in quasipolynomial time on power graphs, even for power graphs of
cyclic groups. We study the recognition problem of power graphs when the
adjacency matrix or list is given as input and show that for abelian groups and
some classes of nilpotent groups, it is solvable in polynomial time.

### Data Structures and Algorithms

### 1. [25 Additional Problems -- Extension to the Book "125 Problems in Text Algorithms"](http://arxiv.org/pdf/2507.05770v1)

Authors: Maxime Crochemore, Thierry Lecroq, Wojtek Rytter

This very preliminary text is related to ``Algorithms on Texts'', also called
``Algorithmic Stringology''. It is an extension of the book ``125 Problems in
Text Algorithms'' providing, in the same compact style, more problems with
solutions. We refer also to the companions to ``Text algorithms'' available at
http://monge.univ-mlv.fr/~mac/CLR/clr1-20.pdf and at the web page
http://125-problems.univ-mlv.fr, where all 150 problems (including the ones
presented here) are briefly announced. The selected problems satisfy three
criteria: challenging, having short tricky solutions and solvable with only
very basic background in stringology. For the basics in stringology we refer to
http://monge.univ-mlv.fr/~mac/CLR/clr1-20.pdf.

### 2. [Non-Adaptive Evaluation of $k$-of-$n$ Functions: Tight Gap and a Unit-Cost PTAS](http://arxiv.org/pdf/2507.05877v1)

Authors: Mads Anker Nielsen, Lars Rohwedder, Kevin Schewior

We consider the Stochastic Boolean Function Evaluation (SBFE) problem in the
well-studied case of $k$-of-$n$ functions: There are independent Boolean random
variables $x_1,\dots,x_n$ where each variable $i$ has a known probability $p_i$
of taking value $1$, and a known cost $c_i$ that can be paid to find out its
value. The value of the function is $1$ iff there are at least $k$ $1$s among
the variables. The goal is to efficiently compute a strategy that, at minimum
expected cost, tests the variables until the function value is determined.
While an elegant polynomial-time exact algorithm is known when tests can be
made adaptively, we focus on the non-adaptive variant, for which much less is
known.
  First, we show a clean and tight lower bound of $2$ on the adaptivity gap,
i.e., the worst-case multiplicative loss in the objective function caused by
disallowing adaptivity, of the problem. This improves the tight lower bound of
$3/2$ for the unit-cost variant.
  Second, we give a PTAS for computing the best non-adaptive strategy in the
unit-cost case, the first PTAS for an SBFE problem. At the core, our scheme
establishes a novel notion of two-sided dominance (w.r.t. the optimal solution)
by guessing so-called milestone tests for a set of carefully chosen buckets of
tests. To turn this technique into a polynomial-time algorithm, we use a
decomposition approach paired with a random-shift argument.

### 3. [Learning-Augmented Online Covering Problems](http://arxiv.org/pdf/2507.06032v1)

Authors: Afrouz Jabal Ameli, Laura Sanita, Moritz Venzin

We give a very general and simple framework to incorporate predictions on
requests for online covering problems in a rigorous and black-box manner. Our
framework turns any online algorithm with competitive ratio $\rho(k, \cdot)$
depending on $k$, the number of arriving requests, into an algorithm with
competitive ratio of $\rho(\eta, \cdot)$, where $\eta$ is the prediction error.
With accurate enough prediction, the resulting competitive ratio breaks through
the corresponding worst-case online lower bounds, and smoothly degrades as the
prediction error grows. This framework directly applies to a wide range of
well-studied online covering problems such as facility location, Steiner
problems, set cover, parking permit, etc., and yields improved and novel
bounds.

### 4. [An Optimal Algorithm for Shortest Paths in Unweighted Disk Graphs](http://arxiv.org/pdf/2507.05569v1)

Authors: Bruce W. Brewer, Haitao Wang

Given in the plane a set $S$ of $n$ points and a set of disks centered at
these points, the disk graph $G(S)$ induced by these disks has vertex set $S$
and an edge between two vertices if their disks intersect. Note that the disks
may have different radii. We consider the problem of computing shortest paths
from a source point $s\in S$ to all vertices in $G(S)$ where the length of a
path in $G(S)$ is defined as the number of edges in the path. The previously
best algorithm solves the problem in $O(n\log^2 n)$ time. A lower bound of
$\Omega(n\log n)$ is also known for this problem under the algebraic decision
tree model. In this paper, we present an $O(n\log n)$ time algorithm, which
matches the lower bound and thus is optimal. Another virtue of our algorithm is
that it is quite simple.

### 5. [Parameterized Restless Temporal Path](http://arxiv.org/pdf/2507.05760v1)

Authors: Justine Cauvi, Laurent Viennot

Recently, Bumpus and Meeks introduced a purely temporal parameter, called
vertex-interval-membership-width, which is promising for the design of
fixed-parameter tractable (FPT) algorithms for vertex reachability problems in
temporal graphs. We study this newly introduced parameter for the problem of
restless temporal paths, in which the waiting time at each node is restricted.
In this article, we prove that, in the interval model, where arcs are present
for entire time intervals, finding a restless temporal path is NP-hard even if
the vertex-interval-membership-width is equal to three. We exhibit FPT
algorithms for the point model, where arcs are present at specific points in
time, both with uniform delay one and arbitrary positive delays. In the latter
case, this comes with a slight additional computational cost.

### 6. [A Formal Refutation of the Blockchain Trilemma](http://arxiv.org/pdf/2507.05809v1)

Authors: Craig Wright

The so-called blockchain trilemma asserts the impossibility of simultaneously
achieving scalability, security, and decentralisation within a single
blockchain protocol. In this paper, we formally refute that proposition.
Employing predicate logic, formal automata theory, computational complexity
analysis, and graph-theoretic measures of relay topology--specifically Baran's
model of network path redundancy--we demonstrate that the trilemma constitutes
a category error, conflates distinct analytical domains, and relies upon
unproven causal assumptions. We further expose its reliance on composition
fallacies drawn from flawed system implementations. A constructive
counterexample is presented: a blockchain protocol exhibiting unbounded
transaction throughput, cryptographic security under adversarial load, and
multipath decentralised propagation. This example is not hypothetical but
grounded in protocol design enabled by compact block relay, SPV verification,
and IPv6 multicast. The trilemma is revealed not as a law of protocol
architecture, but as a heuristic fallacy sustained by imprecision and design
defeatism.

### 7. [Instance-Optimal Quantum State Certification with Entangled Measurements](http://arxiv.org/pdf/2507.06010v1)

Authors: Ryan O'Donnell, Chirag Wadhwa

We consider the task of quantum state certification: given a description of a
hypothesis state $\sigma$ and multiple copies of an unknown state $\rho$, a
tester aims to determine whether the two states are equal or $\epsilon$-far in
trace distance. It is known that $\Theta(d/\epsilon^2)$ copies of $\rho$ are
necessary and sufficient for this task, assuming the tester can make entangled
measurements over all copies [CHW07,OW15,BOW19]. However, these bounds are for
a worst-case $\sigma$, and it is not known what the optimal copy complexity is
for this problem on an instance-by-instance basis. While such instance-optimal
bounds have previously been shown for quantum state certification when the
tester is limited to measurements unentangled across copies [CLO22,CLHL22],
they remained open when testers are unrestricted in the kind of measurements
they can perform.
  We address this open question by proving nearly instance-optimal bounds for
quantum state certification when the tester can perform fully entangled
measurements. Analogously to the unentangled setting, we show that the optimal
copy complexity for certifying $\sigma$ is given by the worst-case complexity
times the fidelity between $\sigma$ and the maximally mixed state. We prove our
lower bounds using a novel quantum analogue of the Ingster-Suslina method,
which is likely to be of independent interest. This method also allows us to
recover the $\Omega(d/\epsilon^2)$ lower bound for mixedness testing [OW15],
i.e., certification of the maximally mixed state, with a surprisingly simple
proof.

### Emerging Technologies

### 1. [Practical design and performance of physical reservoir computing using hysteresis](http://arxiv.org/pdf/2507.06063v1)

Authors: Yuhei Yamada

Physical reservoir computing is an innovative idea for using physical
phenomena as computational resources. Recent research has revealed that
information processing techniques can improve the performance, but for
practical applications, it is equally important to study the level of
performance with a simple design that is easy to construct experimentally. We
focus on a reservoir composed of independent hysteretic systems as a model
suitable for the practical implementation of physical reservoir computing. In
this paper, we discuss the appropriate design of this reservoir, its
performance, and its limitations. This research will serve as a practical
guideline for constructing hysteresis-based reservoirs.

### 2. [Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges [Experiment, Analysis \& Benchmark]](http://arxiv.org/pdf/2507.06156v1)

Authors: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

Blockchain bridges have become essential infrastructure for enabling
interoperability across different blockchain networks, with more than $24B
monthly bridge transaction volume. However, their growing adoption has been
accompanied by a disproportionate rise in security breaches, making them the
single largest source of financial loss in Web3. For cross-chain ecosystems to
be robust and sustainable, it is essential to understand and address these
vulnerabilities. In this study, we present a comprehensive systematization of
blockchain bridge design and security. We define three bridge security priors,
formalize the architectural structure of 13 prominent bridges, and identify 23
attack vectors grounded in real-world blockchain exploits. Using this
foundation, we evaluate 43 representative attack scenarios and introduce a
layered threat model that captures security failures across source chain,
off-chain, and destination chain components.
  Our analysis at the static code and transaction network levels reveals
recurring design flaws, particularly in access control, validator trust
assumptions, and verification logic, and identifies key patterns in adversarial
behavior based on transaction-level traces. To support future development, we
propose a decision framework for bridge architecture design, along with defense
mechanisms such as layered validation and circuit breakers. This work provides
a data-driven foundation for evaluating bridge security and lays the groundwork
for standardizing resilient cross-chain infrastructure.

### 3. [Adaptive Communication Through Exploiting RIS, SSK, and CIM for Improved Reliability and Efficiency](http://arxiv.org/pdf/2507.05813v1)

Authors: Ferhat Bayar, Onur Salan, Erdogan Aydin, Haci Ilhan

In this paper, we present a novel communication system model that integrates
reconfigurable intelligent surfaces (RIS), spatial shift keying (SSK), and code
index modulation (CIM) based on Hadamard coding called RIS based transmit
SSK-CIM (RIS-CIM-TSSK). By leveraging RIS, the system adapts rapidly to dynamic
environments, enhancing error rates and overall reliability. SSK facilitates
the transmission of additional passive information while eliminating the need
for multiple radio frequency (RF) chains, thereby reducing complexity. CIM
enhances passive information transmission through frequency domain spreading,
which may increase signal obfuscation. This proposed scheme not only improves
energy efficiency but also offers a robust solution for reliable communication
in modern wireless networks, paving the way for smarter and more adaptable
implementations. We consider a suboptimal, low-complexity detector for the
proposed scheme and also address the blind case for phase adjustment of the
RIS. Finally, we present the simulation results for the proposed system model
across various configurations, including different numbers of receive and
transmit antennas, varying reflecting elements of the RIS, and different code
lengths.

### Formal Languages and Automata Theory

### 1. [Addition Automata and Attractors of Digit Systems Corresponding to Expanding Rational Matrices](http://arxiv.org/pdf/2507.06158v1)

Authors: Anjelo Gabriel R. Cruz, Manuel Joseph C. Loquias, Jörg M. Thuswaldner

Let $A$ be an expanding $2 \times 2$ matrix with rational entries and
$\mathbb{Z}^2[A]$ be the smallest $A$-invariant $\mathbb{Z}$-module containing
$\mathbb{Z}^2$. Let $\mathcal{D}$ be a finite subset of $\mathbb{Z}^2[A]$ which
is a complete residue system of $\mathbb{Z}^2[A]/A\mathbb{Z}^2[A]$. The pair
$(A,\mathcal{D})$ is called a {\em digit system} with {\em base} $A$ and {\em
digit set} $\mathcal{D}$. It is well known that every vector $x \in
\mathbb{Z}^2[A]$ can be written uniquely in the form \[ x = d_0 + Ad_1 + \cdots
+ A^kd_k + A^{k+1}p, \] with $k\in \mathbb{N}$ minimal, $d_0,\dots,d_k \in
\mathcal{D}$, and $p$ taken from a finite set of {\em periodic elements}, the
so-called {\em attractor} of $(A,\mathcal{D})$. If $p$ can always be chosen to
be $0$ we say that $(A,\mathcal{D})$ has the {\em finiteness property}.
  In the present paper we introduce finite-state transducer automata which
realize the addition of the vectors $\pm(1,0)^\top$ and $\pm(0,1)^\top$ to a
given vector $x\in \mathbb{Z}^2[A]$ in a number system $(A,\mathcal{D})$ with
collinear digit set. These automata are applied to characterize all pairs
$(A,\mathcal{D})$ that have the finiteness property and, more generally, to
characterize the attractors of these digit systems.

### Graphics

### 1. [AnatomyCarve: A VR occlusion management technique for medical images based on segment-aware clipping](http://arxiv.org/pdf/2507.05572v1)

Authors: Andrey Titov, Tina N. H. Nantenaina, Marta Kersten-Oertel, Simon Drouin

Visualizing 3D medical images is challenging due to self-occlusion, where
anatomical structures of interest can be obscured by surrounding tissues.
Existing methods, such as slicing and interactive clipping, are limited in
their ability to fully represent internal anatomy in context. In contrast,
hand-drawn medical illustrations in anatomy books manage occlusion effectively
by selectively removing portions based on tissue type, revealing 3D structures
while preserving context. This paper introduces AnatomyCarve, a novel technique
developed for a VR environment that creates high-quality illustrations similar
to those in anatomy books, while remaining fast and interactive. AnatomyCarve
allows users to clip selected segments from 3D medical volumes, preserving
spatial relations and contextual information. This approach enhances
visualization by combining advanced rendering techniques with natural user
interactions in VR. Usability of AnatomyCarve was assessed through a study with
non-experts, while surgical planning effectiveness was evaluated with
practicing neurosurgeons and residents. The results show that AnatomyCarve
enables customized anatomical visualizations, with high user satisfaction,
suggesting its potential for educational and clinical applications.

### 2. [Feature-Based vs. GAN-Based Learning from Demonstrations: When and Why](http://arxiv.org/pdf/2507.05906v1)

Authors: Chenhao Li, Marco Hutter, Andreas Krause

This survey provides a comparative analysis of feature-based and GAN-based
approaches to learning from demonstrations, with a focus on the structure of
reward functions and their implications for policy learning. Feature-based
methods offer dense, interpretable rewards that excel at high-fidelity motion
imitation, yet often require sophisticated representations of references and
struggle with generalization in unstructured settings. GAN-based methods, in
contrast, use implicit, distributional supervision that enables scalability and
adaptation flexibility, but are prone to training instability and coarse reward
signals. Recent advancements in both paradigms converge on the importance of
structured motion representations, which enable smoother transitions,
controllable synthesis, and improved task integration. We argue that the
dichotomy between feature-based and GAN-based methods is increasingly nuanced:
rather than one paradigm dominating the other, the choice should be guided by
task-specific priorities such as fidelity, diversity, interpretability, and
adaptability. This work outlines the algorithmic trade-offs and design
considerations that underlie method selection, offering a framework for
principled decision-making in learning from demonstrations.

### 3. [LighthouseGS: Indoor Structure-aware 3D Gaussian Splatting for Panorama-Style Mobile Captures](http://arxiv.org/pdf/2507.06109v1)

Authors: Seungoh Han, Jaehoon Jang, Hyunsu Kim, Jaeheung Surh, Junhyung Kwak, Hyowon Ha, Kyungdon Joo

Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time novel
view synthesis (NVS) with impressive quality in indoor scenes. However,
achieving high-fidelity rendering requires meticulously captured images
covering the entire scene, limiting accessibility for general users. We aim to
develop a practical 3DGS-based NVS framework using simple panorama-style motion
with a handheld camera (e.g., mobile device). While convenient, this
rotation-dominant motion and narrow baseline make accurate camera pose and 3D
point estimation challenging, especially in textureless indoor scenes. To
address these challenges, we propose LighthouseGS, a novel framework inspired
by the lighthouse-like sweeping motion of panoramic views. LighthouseGS
leverages rough geometric priors, such as mobile device camera poses and
monocular depth estimation, and utilizes the planar structures often found in
indoor environments. We present a new initialization method called plane
scaffold assembly to generate consistent 3D points on these structures,
followed by a stable pruning strategy to enhance geometry and optimization
stability. Additionally, we introduce geometric and photometric corrections to
resolve inconsistencies from motion drift and auto-exposure in mobile devices.
Tested on collected real and synthetic indoor scenes, LighthouseGS delivers
photorealistic rendering, surpassing state-of-the-art methods and demonstrating
the potential for panoramic view synthesis and object placement.

### Computer Science and Game Theory

### 1. [Rethinking Pricing in Energy Markets: Pay-as-Bid vs Pay-as-Clear](http://arxiv.org/pdf/2507.06035v1)

Authors: Ioannis Caragiannis, Zhile Jiang, Stratis Skoulakis

The design of energy markets is a subject of ongoing debate, particularly
concerning the choice between the widely adopted Pay-as-Clear (PC) pricing
mechanism and the alternative Pay-as-Bid (PB). These mechanisms determine how
energy producers are compensated: under PC, all selected producers are paid the
market-clearing price (i.e., the highest accepted bid), while under PB, each
selected producer is paid their own submitted bid. The overarching objective is
to meet the total demand for energy at minimal cost in the presence of
strategic behavior. We present two key theoretical results. First, no mechanism
can uniformly dominate PC or PB. This means that for any mechanism
$\mathcal{M}$, there exists a market configuration and a mixed-strategy Nash
equilibrium of PC (respectively for PB) that yields strictly lower total energy
costs than under $\mathcal{M}$. Second, in terms of worst-case equilibrium
outcomes, PB consistently outperforms PC: across all market instances, the
highest possible equilibrium price under PB is strictly lower than that under
PC. This suggests a structural robustness of PB to strategic manipulation.
These theoretical insights are further supported by extensive simulations based
on no-regret learning dynamics, which consistently yield lower average market
prices in several energy market settings.

### 2. [Fairness-Aware Static and Dynamic Assortment Optimization: Optimal Selection with Balanced Market Share](http://arxiv.org/pdf/2507.05606v1)

Authors: Omar El Housni, Qing Feng, Huseyin Topaloglu

Assortment optimization is a critical tool for online retailers aiming to
maximize revenue. However, optimizing purely for revenue can lead to imbalanced
sales across products, potentially causing supplier disengagement and reduced
product diversity. To address these fairness concerns, we introduce a market
share balancing constraint that limits the disparity in expected sales between
any two offered products to a factor of a given parameter $\alpha$. We study
both static and dynamic assortment optimization under the multinomial logit
(MNL) model with this fairness constraint. In the static setting, the seller
selects a distribution over assortments that satisfies the market share
balancing constraint while maximizing expected revenue. We show that this
problem can be solved in polynomial time, and we characterize the structure of
the optimal solution: a product is included if and only if its revenue and
preference weight exceed certain thresholds. We further extend our analysis to
settings with additional feasibility constraints on the assortment and
demonstrate that, given a $\beta$-approximation oracle for the constrained
problem, we can construct a $\beta$-approximation algorithm under the fairness
constraint. In the dynamic setting, each product has a finite initial
inventory, and the seller implements a dynamic policy to maximize total
expected revenue while respecting both inventory limits and the market share
balancing constraint in expectation. We design a policy that is asymptotically
optimal, with its approximation ratio converging to one as inventories grow
large.

### 3. [An efficiency ordering of k-price auctions under complete information](http://arxiv.org/pdf/2507.05738v1)

Authors: Sumit Goel, Jeffrey Zeidel

We study $k$-price auctions in a complete information environment and
characterize all pure-strategy Nash equilibrium outcomes. In a setting with $n$
agents having ordered valuations, we show that any agent, except those with the
lowest $k-2$ valuations, can win in equilibrium. As a consequence, worst-case
welfare increases monotonically as we go from $k=2$ (second-price auction) to
$k=n$ (lowest-price auction), with the first-price auction achieving the
highest worst-case welfare.

### 4. [A Directed Lazy Random Walk Model to Three-Way Dynamic Matching Problem](http://arxiv.org/pdf/2507.06126v1)

Authors: Souvik Roy, Agamani Saha

This paper explores a novel extension of dynamic matching theory by analyzing
a three-way matching problem involving agents from three distinct populations,
each with two possible types. Unlike traditional static or two-way dynamic
models, our setting captures more complex team-formation environments where one
agent from each of the three populations must be matched to form a valid team.
We consider two preference structures: assortative or homophilic, where agents
prefer to be matched with others of the same type, and dis-assortative or
heterophilic, where diversity within the team is valued. Agents arrive
sequentially and face a trade-off between matching immediately or waiting for a
higher quality match in the future albeit with a waiting cost. We construct and
analyze the corresponding transition probability matrices for each preference
regime and demonstrate the existence and uniqueness of stationary
distributions. Our results show that stable and efficient outcomes can arise in
dynamic, multi-agent matching environments, offering a deeper understanding of
how complex matching processes evolve over time and how they can be effectively
managed.

### 5. [Aligned Textual Scoring Rules](http://arxiv.org/pdf/2507.06221v1)

Authors: Yuxuan Lu, Yifan Wu, Jason Hartline, Michael J. Curry

Scoring rules elicit probabilistic predictions from a strategic agent by
scoring the prediction against a ground truth state. A scoring rule is proper
if, from the agent's perspective, reporting the true belief maximizes the
expected score. With the development of language models, Wu and Hartline (2024)
proposes a reduction from textual information elicitation to the numerical
(i.e. probabilistic) information elicitation problem, which achieves provable
properness for textual elicitation. However, not all proper scoring rules are
well aligned with human preference over text. Our paper designs the Aligned
Scoring rule (ASR) for text by optimizing and minimizing the mean squared error
between a proper scoring rule and a reference score (e.g. human score). Our
experiments show that our ASR outperforms previous methods in aligning with
human preference while maintaining properness.

### 6. [Minimal balanced collections and their applications to core stability and other topics of game theory](http://arxiv.org/pdf/2507.05898v1)

Authors: Dylan Laplace Mermoud, Michel Grabisch, Peter Sudhölter

Minimal balanced collections are a generalization of partitions of a finite
set of n elements and have important applications in cooperative game theory
and discrete mathematics. However, their number is not known beyond n = 4. In
this paper we investigate the problem of generating minimal balanced
collections and implement the Peleg algorithm, permitting to generate all
minimal balanced collections till n = 7. Secondly, we provide practical
algorithms to check many properties of coalitions and games, based on minimal
balanced collections, in a way which is faster than linear programming-based
methods. In particular, we construct an algorithm to check if the core of a
cooperative game is a stable set in the sense of von Neumann and Morgenstern.
The algorithm implements a theorem according to which the core is a stable set
if and only if a certain nested balancedness condition is valid. The second
level of this condition requires generalizing the notion of balanced collection
to balanced sets.

### Human-Computer Interaction

### 1. [StoryGrid: A Tangible Interface for Student Expression](http://arxiv.org/pdf/2507.05600v1)

Authors: Tom Moher, Louis Gomez, Janet Kim, Claudia Hindo, Benjamin Watson, Stephen Fransen, Tim McEneany

StorySpace is a classroom-based design and presentation system for
interactive multimedia posters. Employing the technology base first used in
Eden's PITAboard [2002], StorySpace allows groups of learners to manipulate
projected multimedia objects on a horizontal board using a small collection of
shared physical tokens. In this paper, we present the ongoing design history of
StorySpace in the context of its introduction within an urban high school
literature class. Interface modifications based on student and teacher feedback
led on changes in token semantics and media importing methods. We describe how
StorySpace features enriched students' interpretations of literature, with
particular emphasis in two areas: (1) attention to audience, and (2) reflection
of multiple perspectives.

### 2. [Hapster: Using Apple Watch Haptics to Enable Live Low-Friction Student Feedback in the Physical Classroom](http://arxiv.org/pdf/2507.05605v1)

Authors: Oleg Aleksandrovich Golev, Michelle Huang, Chanketya Nop, Kritin Vongthongsri, Andrés Monroy-Hernández, Parastoo Abtahi

The benefits of student response systems (SRSs) for in-person lectures are
well-researched. However, all current SRSs only rely on a visual interface to
relay information to the instructor. We describe the design and evaluation of
Hapster, a prototype system that uses an Apple Watch to deliver live,
aggregated student feedback to the instructor via both visual and vibro-tactile
modalities. We evaluated this system with 6 instructors and 155 students at a
U.S. university. Participants reported that the system was effective at
delivering live student feedback and facilitating better engagement from both
the instructor and the students. However, instructors also noted several
challenges with differentiating and perceiving the haptic sequences while
lecturing. We conclude by discussing the tradeoff between system flexibility
and abuse potential while identifying opportunities for further research
regarding accessibility, content moderation, and additional interaction
modalities. Our results suggest that haptics can be used as an effective live
feedback mechanism for instructors in the physical classroom.

### 3. [Breaking the Plane: Exploring Real-Time Visualization of 3D Surfaces in Augmented Reality with Handwritten Input](http://arxiv.org/pdf/2507.05616v1)

Authors: Liam Franco Esparraguera, Kristoffer Selberg, Brian Lou, Jenny Sun, Beza Desta, Andrés Monroy-Hernández, Parastoo Abtahi

We introduce Breaking the Plane, an augmented reality (AR) application built
for AR headsets that enables users to visualize 3D mathematical functions using
handwritten input. Researchers have demonstrated overlaying 3D visualizations
of mathematical concepts through AR enhances learning motivation and
comprehension, and equation parsing makes the authoring of teaching materials
more time-efficient for instructors. Previous works have developed AR systems
that separately employ equation parsing and 3D mathematical visualizations, but
work has yet to be done to combine those features by enabling real-time
interactions and dynamic visualizations that help users learn in situ. We
explore this by developing an AR system featuring handwritten equation parsing,
graph manipulation, and a 3D function plotter. We found that our system
significantly surpassed other systems in engagement, achieved comparable ease
of use to a popular visualization tool, was considered the most effective in
aiding problem-solving, and was highly preferred by participants for future
use.

### 4. [Evaluation of Large Language Model-Driven AutoML in Data and Model Management from Human-Centered Perspective](http://arxiv.org/pdf/2507.05962v1)

Authors: Jiapeng Yao, Lantian Zhang, Jiping Huang

As organizations increasingly seek to leverage machine learning (ML)
capabilities, the technical complexity of implementing ML solutions creates
significant barriers to adoption and impacts operational efficiency. This
research examines how Large Language Models (LLMs) can transform the
accessibility of ML technologies within organizations through a human-centered
Automated Machine Learning (AutoML) approach. Through a comprehensive user
study involving 15 professionals across various roles and technical
backgrounds, we evaluate the organizational impact of an LLM-based AutoML
framework compared to traditional implementation methods. Our research offers
four significant contributions to both management practice and technical
innovation: First, we present pioneering evidence that LLM-based interfaces can
dramatically improve ML implementation success rates, with 93.34% of users
achieved superior performance in the LLM condition, with 46.67% showing higher
accuracy (10-25% improvement over baseline) and 46.67% demonstrating
significantly higher accuracy (>25% improvement over baseline), while 6.67%
maintained comparable performance levels; and 60% reporting substantially
reduced development time. Second, we demonstrate how natural language
interfaces can effectively bridge the technical skills gap in organizations,
cutting implementation time by 50% while improving accuracy across all
expertise levels. Third, we provide valuable insights for organizations
designing human-AI collaborative systems, showing that our approach reduced
error resolution time by 73% and significantly accelerated employee learning
curves. Finally, we establish empirical support for natural language as an
effective interface for complex technical systems, offering organizations a
path to democratize ML capabilities without compromising quality or
performance.

### 5. [Exploring Collaboration Patterns and Strategies in Human-AI Co-creation through the Lens of Agency: A Scoping Review of the Top-tier HCI Literature](http://arxiv.org/pdf/2507.06000v1)

Authors: Shuning Zhang, Hui Wang, Xin Yi

As Artificial Intelligence (AI) increasingly becomes an active collaborator
in co-creation, understanding the distribution and dynamic of agency is
paramount. The Human-Computer Interaction (HCI) perspective is crucial for this
analysis, as it uniquely reveals the interaction dynamics and specific control
mechanisms that dictate how agency manifests in practice. Despite this
importance, a systematic synthesis mapping agency configurations and control
mechanisms within the HCI/CSCW literature is lacking. Addressing this gap, we
reviewed 134 papers from top-tier HCI/CSCW venues (e.g., CHI, UIST, CSCW) over
the past 20 years. This review yields four primary contributions: (1) an
integrated theoretical framework structuring agency patterns, control
mechanisms, and interaction contexts, (2) a comprehensive operational catalog
of control mechanisms detailing how agency is implemented; (3) an actionable
cross-context map linking agency configurations to diverse co-creative
practices; and (4) grounded implications and guidance for future CSCW research
and the design of co-creative systems, addressing aspects like trust and
ethics.

### 6. [Large Language Models Predict Human Well-being -- But Not Equally Everywhere](http://arxiv.org/pdf/2507.06141v1)

Authors: Pat Pataranutaporn, Nattavudh Powdthavee, Chayapatr Archiwaranguprok, Pattie Maes

Subjective well-being is a key metric in economic, medical, and policy
decision-making. As artificial intelligence provides scalable tools for
modelling human outcomes, it is crucial to evaluate whether large language
models (LLMs) can accurately predict well-being across diverse global
populations. We evaluate four leading LLMs using data from 64,000 individuals
in 64 countries. While LLMs capture broad correlates such as income and health,
their predictive accuracy decreases in countries underrepresented in the
training data, highlighting systematic biases rooted in global digital and
economic inequality. A pre-registered experiment demonstrates that LLMs rely on
surface-level linguistic similarity rather than conceptual understanding,
leading to systematic misestimations in unfamiliar or resource-limited
settings. Injecting findings from underrepresented contexts substantially
enhances performance, but a significant gap remains. These results highlight
both the promise and limitations of LLMs in predicting global well-being,
underscoring the importance of robust validation prior to their implementation
across these areas.

### 7. [V(is)owel: An Interactive Vowel Chart to Understand What Makes Visual Pronunciation Effective in Second Language Learning](http://arxiv.org/pdf/2507.06202v1)

Authors: Charlotte Kiesel, Dipayan Mukherjee, Mark Hasegawa-Johnson, Karrie Karahalios

Visual feedback speeds up learners' improvement of pronunciation in a second
language. The visual combined with audio allows speakers to see sounds and
differences in pronunciation that they are unable to hear. Prior studies have
tested different visual methods for improving pronunciation, however, we do not
have conclusive understanding of what aspects of the visualizations contributed
to improvements. Based on previous work, we created V(is)owel, an interactive
vowel chart. Vowel charts provide actionable feedback by directly mapping
physical tongue movement onto a chart. We compared V(is)owel with an
auditory-only method to explore how learners parse visual and auditory feedback
to understand how and why visual feedback is effective for pronunciation
improvement. The findings suggest that designers should include explicit
anatomical feedback that directly maps onto physical movement for phonetically
untrained learners. Furthermore, visual feedback has the potential to motivate
more practice since all eight of the participants cited using the visuals as a
goal with V(is)owel versus relying on their own judgment with audio alone.
Their statements are backed up by all participants practicing words with
V(is)owel more than with audio-only. Our results indicate that V(is)owel is
effective at providing actionable feedback, demonstrating the potential of
visual feedback methods in second language learning.

### 8. [AnatomyCarve: A VR occlusion management technique for medical images based on segment-aware clipping](http://arxiv.org/pdf/2507.05572v1)

Authors: Andrey Titov, Tina N. H. Nantenaina, Marta Kersten-Oertel, Simon Drouin

Visualizing 3D medical images is challenging due to self-occlusion, where
anatomical structures of interest can be obscured by surrounding tissues.
Existing methods, such as slicing and interactive clipping, are limited in
their ability to fully represent internal anatomy in context. In contrast,
hand-drawn medical illustrations in anatomy books manage occlusion effectively
by selectively removing portions based on tissue type, revealing 3D structures
while preserving context. This paper introduces AnatomyCarve, a novel technique
developed for a VR environment that creates high-quality illustrations similar
to those in anatomy books, while remaining fast and interactive. AnatomyCarve
allows users to clip selected segments from 3D medical volumes, preserving
spatial relations and contextual information. This approach enhances
visualization by combining advanced rendering techniques with natural user
interactions in VR. Usability of AnatomyCarve was assessed through a study with
non-experts, while surgical planning effectiveness was evaluated with
practicing neurosurgeons and residents. The results show that AnatomyCarve
enables customized anatomical visualizations, with high user satisfaction,
suggesting its potential for educational and clinical applications.

### 9. [The Ethical Implications of AI in Creative Industries: A Focus on AI-Generated Art](http://arxiv.org/pdf/2507.05549v1)

Authors: Prerana Khatiwada, Joshua Washington, Tyler Walsh, Ahmed Saif Hamed, Lokesh Bhatta

As Artificial Intelligence (AI) continues to grow daily, more exciting (and
somewhat controversial) technology emerges every other day. As we see the
advancements in AI, we see more and more people becoming skeptical of it. This
paper explores the complications and confusion around the ethics of generative
AI art. We delve deep into the ethical side of AI, specifically generative art.
We step back from the excitement and observe the impossible conundrums that
this impressive technology produces. Covering environmental consequences,
celebrity representation, intellectual property, deep fakes, and artist
displacement. Our research found that generative AI art is responsible for
increased carbon emissions, spreading misinformation, copyright infringement,
unlawful depiction, and job displacement. In light of this, we propose multiple
possible solutions for these problems. We address each situation's history,
cause, and consequences and offer different viewpoints. At the root of it all,
though, the central theme is that generative AI Art needs to be correctly
legislated and regulated.

### 10. [Constella: Supporting Storywriters' Interconnected Character Creation through LLM-based Multi-Agents](http://arxiv.org/pdf/2507.05820v1)

Authors: Syemin Park, Soobin Park, Youn-kyung Lim

Creating a cast of characters by attending to their relational dynamics is a
critical aspect of most long-form storywriting. However, our formative study
(N=14) reveals that writers struggle to envision new characters that could
influence existing ones, to balance similarities and differences among
characters, and to intricately flesh out their relationships. Based on these
observations, we designed Constella, an LLM-based multi-agent tool that
supports storywriters' interconnected character creation process. Constella
suggests related characters (FRIENDS DISCOVERY feature), reveals the inner
mindscapes of several characters simultaneously (JOURNALS feature), and
manifests relationships through inter-character responses (COMMENTS feature).
Our 7-8 day deployment study with storywriters (N=11) shows that Constella
enabled the creation of expansive communities composed of related characters,
facilitated the comparison of characters' thoughts and emotions, and deepened
writers' understanding of character relationships. We conclude by discussing
how multi-agent interactions can help distribute writers' attention and effort
across the character cast.

### Information Retrieval

### 1. [Vers un cadre ontologique pour la gestion des comp{é}tences : {à} des fins de formation, de recrutement, de m{é}tier, ou de recherches associ{é}es](http://arxiv.org/pdf/2507.05767v1)

Authors: Ngoc Luyen Le, Marie-Hélène Abel, Bertrand Laforge

The rapid transformation of the labor market, driven by technological
advancements and the digital economy, requires continuous competence
development and constant adaptation. In this context, traditional competence
management systems lack interoperability, adaptability, and semantic
understanding, making it difficult to align individual competencies with labor
market needs and training programs. This paper proposes an ontology-based
framework for competence management, enabling a structured representation of
competencies, occupations, and training programs. By leveraging ontological
models and semantic reasoning, this framework aims to enhance the automation of
competence-to-job matching, the personalization of learning recommendations,
and career planning. This study discusses the design, implementation, and
potential applications of the framework, focusing on competence training
programs, job searching, and finding competent individuals.

### 2. [KERAG_R: Knowledge-Enhanced Retrieval-Augmented Generation for Recommendation](http://arxiv.org/pdf/2507.05863v1)

Authors: Zeyuan Meng, Zixuan Yi, Iadh Ounis

Large Language Models (LLMs) have shown strong potential in recommender
systems due to their contextual learning and generalisation capabilities.
Existing LLM-based recommendation approaches typically formulate the
recommendation task using specialised prompts designed to leverage their
contextual abilities, and aligning their outputs closely with human preferences
to yield an improved recommendation performance. However, the use of LLMs for
recommendation tasks is limited by the absence of domain-specific knowledge.
This lack of relevant relational knowledge about the items to be recommended in
the LLM's pre-training corpus can lead to inaccuracies or hallucinations,
resulting in incorrect or misleading recommendations. Moreover, directly using
information from the knowledge graph introduces redundant and noisy
information, which can affect the LLM's reasoning process or exceed its input
context length, thereby reducing the performance of LLM-based recommendations.
To address the lack of domain-specific knowledge, we propose a novel model
called Knowledge-Enhanced Retrieval-Augmented Generation for Recommendation
(KERAG_R). Specifically, we leverage a graph retrieval-augmented generation
(GraphRAG) component to integrate additional information from a knowledge graph
(KG) into instructions, enabling the LLM to collaboratively exploit
recommendation signals from both text-based user interactions and the knowledge
graph to better estimate the users' preferences in a recommendation context. In
particular, we perform graph RAG by pre-training a graph attention network
(GAT) to select the most relevant triple for the target users for the used LLM,
thereby enhancing the LLM while reducing redundant and noisy information. Our
extensive experiments on three public datasets show that our proposed KERAG_R
model significantly outperforms ten existing state-of-the-art recommendation
methods.

### 3. [RecRankerEval: A Flexible and Extensible Framework for Top-k LLM-based Recommendation](http://arxiv.org/pdf/2507.05880v1)

Authors: Zeyuan Meng, Zixuan Yi, Iadh Ounis

A recent Large language model (LLM)-based recommendation model, called
RecRanker, has demonstrated a superior performance in the top-k recommendation
task compared to other models. In particular, RecRanker samples users via
clustering, generates an initial ranking list using an initial recommendation
model, and fine-tunes an LLM through hybrid instruction tuning to infer user
preferences. However, the contribution of each core component remains
underexplored. In this work, we inspect the reproducibility of RecRanker, and
study the impact and role of its various components. We begin by reproducing
the RecRanker pipeline through the implementation of all its key components.
Our reproduction shows that the pairwise and listwise methods achieve a
performance comparable to that reported in the original paper. For the
pointwise method, while we are also able to reproduce the original paper's
results, further analysis shows that the performance is abnormally high due to
data leakage from the inclusion of ground-truth information in the prompts. To
enable a fair and comprehensive evaluation of LLM-based top-k recommendations,
we propose RecRankerEval, an extensible framework that covers five key
dimensions: user sampling strategy, initial recommendation model, LLM backbone,
dataset selection, and instruction tuning method. Using the RecRankerEval
framework, we show that the original results of RecRanker can be reproduced on
the ML-100K and ML-1M datasets, as well as the additional Amazon-Music dataset,
but not on BookCrossing due to the lack of timestamp information in the
original RecRanker paper. Furthermore, we demonstrate that RecRanker's
performance can be improved by employing alternative user sampling methods,
stronger initial recommenders, and more capable LLMs.

### 4. [Hierarchical Interaction Summarization and Contrastive Prompting for Explainable Recommendations](http://arxiv.org/pdf/2507.06044v1)

Authors: Yibin Liu, Ang Li, Shijian Li

Explainable recommendations, which use the information of user and item with
interaction to generate a explanation for why the user would interact with the
item, are crucial for improving user trust and decision transparency to the
recommender system. Existing methods primarily rely on encoding features of
users and items to embeddings, which often leads to information loss due to
dimensionality reduction, sparse interactions, and so on. With the advancements
of large language models (LLMs) in language comprehension, some methods use
embeddings as LLM inputs for explanation generation. However, since embeddings
lack inherent semantics, LLMs must adjust or extend their parameters to
interpret them, a process that inevitably incurs information loss. To address
this issue, we propose a novel approach combining profile generation via
hierarchical interaction summarization (PGHIS), which leverages a pretrained
LLM to hierarchically summarize user-item interactions, generating structured
textual profiles as explicit representations of user and item characteristics.
Additionally, we propose contrastive prompting for explanation generation
(CPEG) which employs contrastive learning to guide another reasoning language
models in producing high-quality ground truth recommendation explanations.
Finally, we use the textual profiles of user and item as input and high-quality
explanation as output to fine-tune a LLM for generating explanations.
Experimental results on multiple datasets demonstrate that our approach
outperforms existing state-of-the-art methods, achieving a great improvement on
metrics about explainability (e.g., 5% on GPTScore) and text quality.
Furthermore, our generated ground truth explanations achieve a significantly
higher win rate compared to user-written reviews and those produced by other
methods, demonstrating the effectiveness of CPEG in generating high-quality
ground truths.

### 5. [Unconditional Diffusion for Generative Sequential Recommendation](http://arxiv.org/pdf/2507.06121v1)

Authors: Yimeng Bai, Yang Zhang, Sihao Ding, Shaohui Ruan, Han Yao, Danhui Guan, Fuli Feng, Tat-Seng Chua

Diffusion models, known for their generative ability to simulate data
creation through noise-adding and denoising processes, have emerged as a
promising approach for building generative recommenders. To incorporate user
history for personalization, existing methods typically adopt a conditional
diffusion framework, where the reverse denoising process of reconstructing
items from noise is modified to be conditioned on the user history. However,
this design may fail to fully utilize historical information, as it gets
distracted by the need to model the "item $\leftrightarrow$ noise" translation.
This motivates us to reformulate the diffusion process for sequential
recommendation in an unconditional manner, treating user history (instead of
noise) as the endpoint of the forward diffusion process (i.e., the starting
point of the reverse process), rather than as a conditional input. This
formulation allows for exclusive focus on modeling the "item $\leftrightarrow$
history" translation. To this end, we introduce Brownian Bridge Diffusion
Recommendation (BBDRec). By leveraging a Brownian bridge process, BBDRec
enforces a structured noise addition and denoising mechanism, ensuring that the
trajectories are constrained towards a specific endpoint -- user history,
rather than noise. Extensive experiments demonstrate BBDRec's effectiveness in
enhancing sequential recommendation performance. The source code is available
at https://github.com/baiyimeng/BBDRec.

### 6. [From ID-based to ID-free: Rethinking ID Effectiveness in Multimodal Collaborative Filtering Recommendation](http://arxiv.org/pdf/2507.05715v1)

Authors: Guohao Li, Li Jing, Jia Wu, Xuefei Li, Kai Zhu, Yue He

Most existing multimodal collaborative filtering recommendation (MCFRec)
methods rely heavily on ID features and multimodal content to enhance
recommendation performance. However, this paper reveals that ID features are
effective but have limited benefits in multimodal collaborative filtering
recommendation. Therefore, this paper systematically deconstruct the pros and
cons of ID features: (i) they provide initial embedding but lack semantic
richness, (ii) they provide a unique identifier for each user and item but
hinder generalization to untrained data, and (iii) they assist in aligning and
fusing multimodal features but may lead to representation shift. Based on these
insights, this paper proposes IDFREE, an ID-free multimodal collaborative
Filtering REcommEndation baseline. IDFREE replaces ID features with multimodal
features and positional encodings to generate semantically meaningful ID-free
embeddings. For ID-free multimodal collaborative filtering, it further proposes
an adaptive similarity graph module to construct dynamic user-user and
item-item graphs based on multimodal features. Then, an augmented user-item
graph encoder is proposed to construct more effective user and item encoding.
Finally, IDFREE achieves inter-multimodal alignment based on the contrastive
learning and uses Softmax loss as recommendation loss. Basic experiments on
three public datasets demonstrate that IDFREE outperforms existing ID-based
MCFRec methods, achieving an average performance gain of 72.24% across standard
metrics (Recall@5, 10, 20, 50 and NDCG@5, 10, 20, 50). Exploratory and extended
experiments further validate our findings on the limitations of ID features in
MCFRec. The code is released at https://github.com/G-H-Li/IDFREE.

### 7. [When Transformers Meet Recommenders: Integrating Self-Attentive Sequential Recommendation with Fine-Tuned LLMs](http://arxiv.org/pdf/2507.05733v1)

Authors: Kechen Liu

Self-Attentive Sequential Recommendation (SASRec) effectively captures
long-term user preferences by applying attention mechanisms to historical
interactions. Concurrently, the rise of Large Language Models (LLMs) has
motivated research into LLM-based recommendation, which leverages their
powerful generalization and language understanding capabilities. However, LLMs
often lack the domain-specific knowledge and collaborative signals essential
for high-quality recommendations when relying solely on textual prompts. To
address this limitation, this study proposes SASRecLLM, a novel framework that
integrates SASRec as a collaborative encoder with an LLM fine-tuned using
Low-Rank Adaptation (LoRA). The components are connected via a mapping layer to
align their dimensional spaces, and three targeted training strategies are
designed to optimize the hybrid architecture. Extensive experiments on multiple
datasets demonstrate that SASRecLLM achieves robust and consistent improvements
over strong baselines in both cold-start and warm-start scenarios. This work
advances the field of LLM-based recommendation by presenting a modular and
effective paradigm for fusing structured collaborative filtering with the
semantic power of fine-tuned LLMs. The implementation is available on GitHub:
https://github.com/kechenkristin/RecLLM

### 8. [On the Costs and Benefits of Learned Indexing for Dynamic High-Dimensional Data: Extended Version](http://arxiv.org/pdf/2507.05865v1)

Authors: Terézia Slanináková, Jaroslav Olha, David Procházka, Matej Antol, Vlastislav Dohnal

One of the main challenges within the growing research area of learned
indexing is the lack of adaptability to dynamically expanding datasets. This
paper explores the dynamization of a static learned index for complex data
through operations such as node splitting and broadening, enabling efficient
adaptation to new data. Furthermore, we evaluate the trade-offs between static
and dynamic approaches by introducing an amortized cost model to assess query
performance in tandem with the build costs of the index structure, enabling
experimental determination of when a dynamic learned index outperforms its
static counterpart. We apply the dynamization method to a static learned index
and demonstrate that its superior scaling quickly surpasses the static
implementation in terms of overall costs as the database grows. This is an
extended version of the paper presented at DAWAK 2025.

### 9. [Semantic Certainty Assessment in Vector Retrieval Systems: A Novel Framework for Embedding Quality Evaluation](http://arxiv.org/pdf/2507.05933v1)

Authors: Y. Du

Vector retrieval systems exhibit significant performance variance across
queries due to heterogeneous embedding quality. We propose a lightweight
framework for predicting retrieval performance at the query level by combining
quantization robustness and neighborhood density metrics. Our approach is
motivated by the observation that high-quality embeddings occupy geometrically
stable regions in the embedding space and exhibit consistent neighborhood
structures. We evaluate our method on 4 standard retrieval datasets, showing
consistent improvements of 9.4$\pm$1.2\% in Recall@10 over competitive
baselines. The framework requires minimal computational overhead (less than 5\%
of retrieval time) and enables adaptive retrieval strategies. Our analysis
reveals systematic patterns in embedding quality across different query types,
providing insights for targeted training data augmentation.

### 10. [Enhancing the Interpretability of Rule-based Explanations through Information Retrieval](http://arxiv.org/pdf/2507.05976v1)

Authors: Alessandro Umbrico, Guido Bologna, Luca Coraci, Francesca Fracasso, Silvia Gola, Gabriella Cortellessa

The lack of transparency of data-driven Artificial Intelligence techniques
limits their interpretability and acceptance into healthcare decision-making
processes. We propose an attribution-based approach to improve the
interpretability of Explainable AI-based predictions in the specific context of
arm lymphedema's risk assessment after lymph nodal radiotherapy in breast
cancer. The proposed method performs a statistical analysis of the attributes
in the rule-based prediction model using standard metrics from Information
Retrieval techniques. This analysis computes the relevance of each attribute to
the prediction and provides users with interpretable information about the
impact of risk factors. The results of a user study that compared the output
generated by the proposed approach with the raw output of the Explainable AI
model suggested higher levels of interpretability and usefulness in the context
of predicting lymphedema risk.

### Machine Learning

### 1. [Gait-Based Hand Load Estimation via Deep Latent Variable Models with Auxiliary Information](http://arxiv.org/pdf/2507.05544v1)

Authors: Jingyi Gao, Sol Lim, Seokhyun Chung

Machine learning methods are increasingly applied to ergonomic risk
assessment in manual material handling, particularly for estimating carried
load from gait motion data collected from wearable sensors. However, existing
approaches often rely on direct mappings from loaded gait to hand load,
limiting generalization and predictive accuracy. In this study, we propose an
enhanced load estimation framework that incorporates auxiliary information,
including baseline gait patterns during unloaded walking and carrying style.
While baseline gait can be automatically captured by wearable sensors and is
thus readily available at inference time, carrying style typically requires
manual labeling and is often unavailable during deployment. Our model
integrates deep latent variable modeling with temporal convolutional networks
and bi-directional cross-attention to capture gait dynamics and fuse loaded and
unloaded gait patterns. Guided by domain knowledge, the model is designed to
estimate load magnitude conditioned on carrying style, while eliminating the
need for carrying style labels at inference time. Experiments using real-world
data collected from inertial measurement units attached to participants
demonstrate substantial accuracy gains from incorporating auxiliary information
and highlight the importance of explicit fusion mechanisms over naive feature
concatenation.

### 2. [Canine Clinical Gait Analysis for Orthopedic and Neurological Disorders: An Inertial Deep-Learning Approach](http://arxiv.org/pdf/2507.05671v1)

Authors: Netta Palez, Léonie Straß, Sebastian Meller, Holger Volk, Anna Zamansky, Itzik Klein

Canine gait analysis using wearable inertial sensors is gaining attention in
veterinary clinical settings, as it provides valuable insights into a range of
mobility impairments. Neurological and orthopedic conditions cannot always be
easily distinguished even by experienced clinicians. The current study explored
and developed a deep learning approach using inertial sensor readings to assess
whether neurological and orthopedic gait could facilitate gait analysis. Our
investigation focused on optimizing both performance and generalizability in
distinguishing between these gait abnormalities. Variations in sensor
configurations, assessment protocols, and enhancements to deep learning model
architectures were further suggested. Using a dataset of 29 dogs, our proposed
approach achieved 96% accuracy in the multiclass classification task
(healthy/orthopedic/neurological) and 82% accuracy in the binary classification
task (healthy/non-healthy) when generalizing to unseen dogs. Our results
demonstrate the potential of inertial-based deep learning models to serve as a
practical and objective diagnostic and clinical aid to differentiate gait
assessment in orthopedic and neurological conditions.

### 3. [Hierarchical Task Offloading for UAV-Assisted Vehicular Edge Computing via Deep Reinforcement Learning](http://arxiv.org/pdf/2507.05722v1)

Authors: Hongbao Li, Ziye Jia, Sijie He, Kun Guo, Qihui Wu

With the emergence of compute-intensive and delay-sensitive applications in
vehicular networks, unmanned aerial vehicles (UAVs) have emerged as a promising
complement for vehicular edge computing due to the high mobility and flexible
deployment. However, the existing UAV-assisted offloading strategies are
insufficient in coordinating heterogeneous computing resources and adapting to
dynamic network conditions. Hence, this paper proposes a dual-layer
UAV-assisted edge computing architecture based on partial offloading, composed
of the relay capability of high-altitude UAVs and the computing support of
low-altitude UAVs. The proposed architecture enables efficient integration and
coordination of heterogeneous resources. A joint optimization problem is
formulated to minimize the system delay and energy consumption while ensuring
the task completion rate. To solve the high-dimensional decision problem, we
reformulate the problem as a Markov decision process and propose a hierarchical
offloading scheme based on the soft actor-critic algorithm. The method
decouples global and local decisions, where the global decisions integrate
offloading ratios and trajectory planning into continuous actions, while the
local scheduling is handled via designing a priority-based mechanism.
Simulations are conducted and demonstrate that the proposed approach
outperforms several baselines in task completion rate, system efficiency, and
convergence speed, showing strong robustness and applicability in dynamic
vehicular environments.

### 4. [Jigsaw: Training Multi-Billion-Parameter AI Weather Models with Optimized Model Parallelism](http://arxiv.org/pdf/2507.05753v1)

Authors: Deifilia Kieckhefen, Markus Götz, Lars H. Heyen, Achim Streit, Charlotte Debus

AI-based methods have revolutionized atmospheric forecasting, with recent
successes in medium-range forecasting spurring the development of climate
foundation models. Accurate modeling of complex atmospheric dynamics at high
spatial resolutions and longer lead times requires large neural networks and
gigabyte-sized data samples, making accelerator memory and I/O-bandwidth the
bottlenecks for model training. We introduce WeatherMixer, a
multi-layer-perceptron-based architecture whose workload scales linearly with
input size, allowing the model to learn global weather phenomena at accuracies
similar to numerical weather prediction. To cope with the computational demand,
we propose Jigsaw, a novel model parallelization scheme that employs both
domain and tensor parallelism, eliminating memory redundancy. Jigsaw exceeds
state-of-the-art performance in strong scaling in compute-communication-limited
systems and achieves superscalar weak scaling in I/O-bandwidth-limited systems.
We scale training to 256 GPUs, reaching peak performances of 9 and 11 PFLOPs,
23% and 28% of theoretical peaks, achieving 68% and 72% scaling efficiency
versus 51% without model parallelism.

### 5. [From Motion to Meaning: Biomechanics-Informed Neural Network for Explainable Cardiovascular Disease Identification](http://arxiv.org/pdf/2507.05783v1)

Authors: Comte Valentin, Gemma Piella, Mario Ceresa, Miguel A. Gonzalez Ballester

Cardiac diseases are among the leading causes of morbidity and mortality
worldwide, which requires accurate and timely diagnostic strategies. In this
study, we introduce an innovative approach that combines deep learning image
registration with physics-informed regularization to predict the biomechanical
properties of moving cardiac tissues and extract features for disease
classification. We utilize the energy strain formulation of Neo-Hookean
material to model cardiac tissue deformations, optimizing the deformation field
while ensuring its physical and biomechanical coherence. This explainable
approach not only improves image registration accuracy, but also provides
insights into the underlying biomechanical processes of the cardiac tissues.
Evaluation on the Automated Cardiac Diagnosis Challenge (ACDC) dataset achieved
Dice scores of 0.945 for the left ventricular cavity, 0.908 for the right
ventricular cavity, and 0.905 for the myocardium. Subsequently, we estimate the
local strains within the moving heart and extract a detailed set of features
used for cardiovascular disease classification. We evaluated five
classification algorithms, Logistic Regression, Multi-Layer Perceptron, Support
Vector Classifier, Random Forest, and Nearest Neighbour, and identified the
most relevant features using a feature selection algorithm. The best performing
classifier obtained a classification accuracy of 98% in the training set and
100% in the test set of the ACDC dataset. By integrating explainable artificial
intelligence, this method empowers clinicians with a transparent understanding
of the model's predictions based on cardiac mechanics, while also significantly
improving the accuracy and reliability of cardiac disease diagnosis, paving the
way for more personalized and effective patient care.

### 6. [Improving Robustness of Foundation Models in Domain Adaptation with Soup-Adapters](http://arxiv.org/pdf/2507.05807v1)

Authors: Marco Roschkowski

In this paper, we tackle two fundamental problems in few-shot domain
adaptation of foundation models. First, hyperparameter tuning is often
impractical due to the lack of large validation datasets. Second, model
robustness under distribution shifts where test time data deviates slightly
from training distributions, remains a concern. We show that by training
multiple independent adapters and averaging their outputs, the new model has a
higher performance and is more robust to distribution shifts compared to any
individual adapter. This improvement holds even when the adapters are trained
with diverse hyperparameters sampled from a wide range, resulting in varied
individual performance. Consequently, our method addresses both of the problems
described above. The ensemble is also significantly less sensitive to the
residual ratio, a critical hyperparameter of CLIP-Adapter. Since the ensemble
can be reparameterized to a single adapter again using a principled
concatenation of the parameters, we refer to our method as Soup-Adapter. This
is also the first study to explore CLIP adapter-style techniques for DINOv2 and
to directly compare them with CLIP in this setting.

### 7. [Diffusion Dataset Condensation: Training Your Diffusion Model Faster with Less Data](http://arxiv.org/pdf/2507.05914v1)

Authors: Rui Huang, Shitong Shao, Zikai Zhou, Pukun Zhao, Hangyu Guo, Tian Ye, Lichen Bai, Shuo Yang, Zeke Xie

Diffusion models have achieved remarkable success in various generative
tasks, but training them remains highly resource-intensive, often requiring
millions of images and many days of GPU computation. From a data-centric
perspective addressing this limitation, we study diffusion dataset condensation
as a new and challenging problem setting. The goal is to construct a
"synthetic" sub-dataset with significantly fewer samples than the original
dataset, enabling high-quality diffusion model training with greatly reduced
cost. To the best of our knowledge, we are the first to formally investigate
dataset condensation for diffusion models, whereas prior work focused on
training discriminative models. To tackle this new challenge, we propose a
novel Diffusion Dataset Condensation (D2C) framework, which consists of two
phases: Select and Attach. The Select phase identifies a compact and diverse
subset using a diffusion difficulty score and interval sampling. The Attach
phase enhances the selected subset by attaching rich semantic and visual
representations to strengthen the conditional signals. Extensive experiments
across various dataset sizes, model architectures, and resolutions show that
our D2C framework enables significantly faster diffusion model training with
dramatically fewer data, while preserving high visual quality. Notably, for the
SiT-XL/2 architecture, D2C achieves a 100x training speed-up, reaching a FID
score of 4.3 in just 40k steps using only 0.8% of the training data.

### 8. [Improving AI-Based Canine Heart Disease Diagnosis with Expert-Consensus Auscultation Labeling](http://arxiv.org/pdf/2507.05950v1)

Authors: Pinar Bisgin, Tom Strube, Niklas Tschorn, Michael Pantförder, Maximilian Fecke, Ingrid Ljungvall, Jens Häggström, Gerhard Wess, Christoph Schummer, Sven Meister, Falk M. Howar

Noisy labels pose significant challenges for AI model training in veterinary
medicine. This study examines expert assessment ambiguity in canine
auscultation data, highlights the negative impact of label noise on
classification performance, and introduces methods for label noise reduction.
To evaluate whether label noise can be minimized by incorporating multiple
expert opinions, a dataset of 140 heart sound recordings (HSR) was annotated
regarding the intensity of holosystolic heart murmurs caused by Myxomatous
Mitral Valve Disease (MMVD). The expert opinions facilitated the selection of
70 high-quality HSR, resulting in a noise-reduced dataset. By leveraging
individual heart cycles, the training data was expanded and classification
robustness was enhanced. The investigation encompassed training and evaluating
three classification algorithms: AdaBoost, XGBoost, and Random Forest. While
AdaBoost and Random Forest exhibited reasonable performances, XGBoost
demonstrated notable improvements in classification accuracy. All algorithms
showed significant improvements in classification accuracy due to the applied
label noise reduction, most notably XGBoost. Specifically, for the detection of
mild heart murmurs, sensitivity increased from 37.71% to 90.98% and specificity
from 76.70% to 93.69%. For the moderate category, sensitivity rose from 30.23%
to 55.81% and specificity from 64.56% to 97.19%. In the loud/thrilling
category, sensitivity and specificity increased from 58.28% to 95.09% and from
84.84% to 89.69%, respectively. These results highlight the importance of
minimizing label noise to improve classification algorithms for the detection
of canine heart murmurs. Index Terms: AI diagnosis, canine heart disease, heart
sound classification, label noise reduction, machine learning, XGBoost,
veterinary cardiology, MMVD.

### 9. [KnowIt: Deep Time Series Modeling and Interpretation](http://arxiv.org/pdf/2507.06009v1)

Authors: M. W. Theunissen, R. Rabe, M. H. Davel

KnowIt (Knowledge discovery in time series data) is a flexible framework for
building deep time series models and interpreting them. It is implemented as a
Python toolkit, with source code and documentation available from
https://must-deep-learning.github.io/KnowIt. It imposes minimal assumptions
about task specifications and decouples the definition of dataset, deep neural
network architecture, and interpretability technique through well defined
interfaces. This ensures the ease of importing new datasets, custom
architectures, and the definition of different interpretability paradigms while
maintaining on-the-fly modeling and interpretation of different aspects of a
user's own time series data. KnowIt aims to provide an environment where users
can perform knowledge discovery on their own complex time series data through
building powerful deep learning models and explaining their behavior. With
ongoing development, collaboration and application our goal is to make this a
platform to progress this underexplored field and produce a trusted tool for
deep time series modeling.

### 10. [Kamae: Bridging Spark and Keras for Seamless ML Preprocessing](http://arxiv.org/pdf/2507.06021v1)

Authors: George Barrowclough, Marian Andrecki, James Shinner, Daniele Donghi

In production recommender systems, feature preprocessing must be faithfully
replicated across training and inference environments. This often requires
duplicating logic between offline and online environments, increasing
engineering effort and introducing risks of dataset shift. We present Kamae, an
open-source Python library that bridges this gap by translating PySpark
preprocessing pipelines into equivalent Keras models. Kamae provides a suite of
configurable Spark transformers and estimators, each mapped to a corresponding
Keras layer, enabling consistent, end-to-end preprocessing across the ML
lifecycle. Framework's utility is illustrated on real-world use cases,
including MovieLens dataset and Expedia's Learning-to-Rank pipelines. The code
is available at https://github.com/ExpediaGroup/kamae.

### Neural and Evolutionary Computing

### 1. [A Universal Framework for Large-Scale Multi-Objective Optimization Based on Particle Drift and Diffusion](http://arxiv.org/pdf/2507.05847v1)

Authors: Jia-Cheng Li, Min-Rong Chen, Guo-Qiang Zeng, Jian Weng, Man Wang, Jia-Lin Mai

Large-scale multi-objective optimization poses challenges to existing
evolutionary algorithms in maintaining the performances of convergence and
diversity because of high dimensional decision variables. Inspired by the
motion of particles in physics, we propose a universal framework for
large-scale multi-objective optimization based on particle drift and diffusion
to solve these challenges in this paper. This framework innovatively divides
the optimization process into three sub-stages: two coarse-tuning sub-stages
and one fine-tuning sub-stage. Different strategies of drift-diffusion
operations are performed on the guiding solutions according to the current
sub-stage, ingeniously simulating the movement of particles under diverse
environmental conditions. Finally, representative evolutionary algorithms are
embedded into the proposed framework, and their effectiveness are evaluated
through comparative experiments on various large-scale multi-objective problems
with 1000 to 5000 decision variables. Moreover, comparative algorithms are
conducted on neural network training problems to validate the effectiveness of
the proposed framework in the practical problems. The experimental results
demonstrate that the framework proposed in this paper significantly enhances
the performance of convergence and diversity of MOEAs, and improves the
computational efficiency of algorithms in solving large-scale multi-objective
optimization problems.

### 2. [Exploring Gain-Doped-Waveguide-Synapse for Neuromorphic Applications: A Pulsed Pump-Signal Approach](http://arxiv.org/pdf/2507.05931v1)

Authors: Robert Otupiri, Ripalta Stabile

Neuromorphic computing promises to transform AI systems by enabling them to
perceive, respond to, and adapt swiftly and accurately to dynamic data and user
interactions. However, traditional silicon-based and hybrid electronic
technologies for artificial neurons constrain neuromorphic processors in terms
of flexibility, scalability, and energy efficiency. In this study, we pioneer
the use of Doped-Gain-Layer-on-Waveguide-Synapses for bio-inspired neurons,
utilizing a pulsed pump-signal mechanism to enhance neuromorphic computation.
This approach addresses critical challenges in scalability and energy
efficiency inherent in current technologies.
  We introduce the concept of Gain on Waveguide Dynamics for synapses,
demonstrating how non-linear pulse transformations of input probe signals occur
under various pump-probe configurations. Our findings reveal that primarily
properties of pulse amplitude, period as well material properties such as
doping densities and population dynamics influence strongly the generation of
spiking responses that emulate neuronal behaviour and effectively how
computational logic is. By harnessing the complex interactions of asynchronous
spiking pump techniques and ion densities in excited states, our method
produces event-driven responses that mirror natural neuronal functions. This
gain-enhanced environment supports short-term memory capabilities alongside
essential characteristics like asynchronous spike generation, threshold
operation, and temporal integration, foundational to brain-inspired spiking
neural network paradigms.

### 3. [A Differential Evolution Algorithm with Neighbor-hood Mutation for DOA Estimation](http://arxiv.org/pdf/2507.06020v1)

Authors: Bo Zhou, Kaijie Xu, Yinghui Quan, Mengdao Xing

Two-dimensional (2D) Multiple Signal Classification algorithm is a powerful
technique for high-resolution direction-of-arrival (DOA) estimation in array
signal processing. However, the exhaustive search over the 2D an-gular domain
leads to high computa-tional cost, limiting its applicability in real-time
scenarios. In this work, we reformulate the peak-finding process as a
multimodal optimization prob-lem, and propose a Differential Evolu-tion
algorithm with Neighborhood Mutation (DE-NM) to efficiently lo-cate multiple
spectral peaks without requiring dense grid sampling. Simu-lation results
demonstrate that the proposed method achieves comparable estimation accuracy to
the traditional grid search, while significantly reduc-ing computation time.
This strategy presents a promising solution for real-time, high-resolution DOA
estimation in practical applications. The imple-mentation code is available at
https://github.com/zzb-nice/DOA_multimodel_optimize.

### 4. [Practical design and performance of physical reservoir computing using hysteresis](http://arxiv.org/pdf/2507.06063v1)

Authors: Yuhei Yamada

Physical reservoir computing is an innovative idea for using physical
phenomena as computational resources. Recent research has revealed that
information processing techniques can improve the performance, but for
practical applications, it is equally important to study the level of
performance with a simple design that is easy to construct experimentally. We
focus on a reservoir composed of independent hysteretic systems as a model
suitable for the practical implementation of physical reservoir computing. In
this paper, we discuss the appropriate design of this reservoir, its
performance, and its limitations. This research will serve as a practical
guideline for constructing hysteresis-based reservoirs.

### 5. [Search-based Selection of Metamorphic Relations for Optimized Robustness Testing of Large Language Models](http://arxiv.org/pdf/2507.05565v1)

Authors: Sangwon Hyun, Shaukat Ali, M. Ali Babar

Assessing the trustworthiness of Large Language Models (LLMs), such as
robustness, has garnered significant attention. Recently, metamorphic testing
that defines Metamorphic Relations (MRs) has been widely applied to evaluate
the robustness of LLM executions. However, the MR-based robustness testing
still requires a scalable number of MRs, thereby necessitating the optimization
of selecting MRs. Most extant LLM testing studies are limited to automatically
generating test cases (i.e., MRs) to enhance failure detection. Additionally,
most studies only considered a limited test space of single perturbation MRs in
their evaluation of LLMs. In contrast, our paper proposes a search-based
approach for optimizing the MR groups to maximize failure detection and
minimize the LLM execution cost. Moreover, our approach covers the
combinatorial perturbations in MRs, facilitating the expansion of test space in
the robustness assessment. We have developed a search process and implemented
four search algorithms: Single-GA, NSGA-II, SPEA2, and MOEA/D with novel
encoding to solve the MR selection problem in the LLM robustness testing. We
conducted comparative experiments on the four search algorithms along with a
random search, using two major LLMs with primary Text-to-Text tasks. Our
statistical and empirical investigation revealed two key findings: (1) the
MOEA/D algorithm performed the best in optimizing the MR space for LLM
robustness testing, and (2) we identified silver bullet MRs for the LLM
robustness testing, which demonstrated dominant capabilities in confusing LLMs
across different Text-to-Text tasks. In LLM robustness assessment, our research
sheds light on the fundamental problem for optimized testing and provides
insights into search-based solutions.

### 6. [Model-free Optical Processors using In Situ Reinforcement Learning with Proximal Policy Optimization](http://arxiv.org/pdf/2507.05583v1)

Authors: Yuhang Li, Shiqi Chen, Tingyu Gong, Aydogan Ozcan

Optical computing holds promise for high-speed, energy-efficient information
processing, with diffractive optical networks emerging as a flexible platform
for implementing task-specific transformations. A challenge, however, is the
effective optimization and alignment of the diffractive layers, which is
hindered by the difficulty of accurately modeling physical systems with their
inherent hardware imperfections, noise, and misalignments. While existing in
situ optimization methods offer the advantage of direct training on the
physical system without explicit system modeling, they are often limited by
slow convergence and unstable performance due to inefficient use of limited
measurement data. Here, we introduce a model-free reinforcement learning
approach utilizing Proximal Policy Optimization (PPO) for the in situ training
of diffractive optical processors. PPO efficiently reuses in situ measurement
data and constrains policy updates to ensure more stable and faster
convergence. We experimentally validated our method across a range of in situ
learning tasks, including targeted energy focusing through a random diffuser,
holographic image generation, aberration correction, and optical image
classification, demonstrating in each task better convergence and performance.
Our strategy operates directly on the physical system and naturally accounts
for unknown real-world imperfections, eliminating the need for prior system
knowledge or modeling. By enabling faster and more accurate training under
realistic experimental constraints, this in situ reinforcement learning
approach could offer a scalable framework for various optical and physical
systems governed by complex, feedback-driven dynamics.

### 7. [evortran: a modern Fortran package for genetic algorithms with applications from LHC data fitting to LISA signal reconstruction](http://arxiv.org/pdf/2507.06082v1)

Authors: Thomas Biekötter

evortran is a modern Fortran library designed for high-performance genetic
algorithms and evolutionary optimization. evortran can be used to tackle a wide
range of problems in high-energy physics and beyond, such as derivative-free
parameter optimization, complex search taks, parameter scans and fitting
experimental data under the presence of instrumental noise. The library is
built as an fpm package with flexibility and efficiency in mind, while also
offering a simple installation process, user interface and integration into
existing Fortran programs. evortran offers a variety of selection, crossover,
mutation and elitism strategies, with which users can tailor an evolutionary
algorithm to their specific needs. evortran supports different abstraction
levels: from operating directly on individuals and populations, to running full
evolutionary cycles, and even enabling migration between independently evolving
populations to enhance convergence and maintain diversity. In this paper, we
present the functionality of the evortran library, demonstrate its capabilities
with example benchmark applications, and compare its performance with existing
genetic algorithm frameworks. As physics-motivated applications, we use
evortran to confront extended Higgs sectors with LHC data and to reconstruct
gravitational wave spectra and the underlying physical parameters from LISA
mock data, demonstrating its effectiveness in realistic, data-driven scenarios.

### 8. [SoftReMish: A Novel Activation Function for Enhanced Convolutional Neural Networks for Visual Recognition Performance](http://arxiv.org/pdf/2507.06148v1)

Authors: Mustafa Bayram Gücen

In this study, SoftReMish, a new activation function designed to improve the
performance of convolutional neural networks (CNNs) in image classification
tasks, is proposed. Using the MNIST dataset, a standard CNN architecture
consisting of two convolutional layers, max pooling, and fully connected layers
was implemented. SoftReMish was evaluated against popular activation functions
including ReLU, Tanh, and Mish by replacing the activation function in all
trainable layers. The model performance was assessed in terms of minimum
training loss and maximum validation accuracy. Results showed that SoftReMish
achieved a minimum loss (3.14e-8) and a validation accuracy (99.41%),
outperforming all other functions tested. These findings demonstrate that
SoftReMish offers better convergence behavior and generalization capability,
making it a promising candidate for visual recognition tasks.

### Networking and Internet Architecture

### 1. [Programmable Governance for Group-Controlled Decentralized Identifiers](http://arxiv.org/pdf/2507.06001v1)

Authors: Carlo Segat, Sandro Rodriguez Garzo, Axel Küpper

Self-Sovereign Identity (SSI) is a paradigm for digital identity management
that offers unique privacy advantages. A key technology in SSI is Decentralized
Identifiers (DIDs) and their associated metadata, DID Documents (DDOs). DDOs
contain crucial verification material such as the public keys of the entity
identified by the DID (i.e., the DID subject) and are often anchored on a
distributed ledger to ensure security and availability. Long-lived DIDs need to
support updates (e.g., key rotation). Ideally, only the DID subject should
authorize DDO updates. However, in practice, update capabilities may be shared
or delegated. While the DID specification acknowledges such scenarios, it does
not define how updates should be authorized when multiple entities jointly
control a DID (i.e., group control). This article examines the implementation
of an on-chain, trustless mechanism enabling DID controllers under group
control to program their governance rules. The main research question is the
following: Can a technical mechanism be developed to orchestrate on-chain group
control of a DDO in a ledger-agnostic and adaptable manner?

### 2. [Baton: Compensate for Missing Wi-Fi Features for Practical Device-free Tracking](http://arxiv.org/pdf/2507.05597v1)

Authors: Yiming Zhao, Xuanqi Meng, Xinyu Tong, Xiulong Liu, Xin Xie, Wenyu Qu

Wi-Fi contact-free sensing systems have attracted widespread attention due to
their ubiquity and convenience. The integrated sensing and communication (ISAC)
technology utilizes off-the-shelf Wi-Fi communication signals for sensing,
which further promotes the deployment of intelligent sensing applications.
However, current Wi-Fi sensing systems often require prolonged and unnecessary
communication between transceivers, and brief communication interruptions will
lead to significant performance degradation. This paper proposes Baton, the
first system capable of accurately tracking targets even under severe Wi-Fi
feature deficiencies. To be specific, we explore the relevance of the Wi-Fi
feature matrix from both horizontal and vertical dimensions. The horizontal
dimension reveals feature correlation across different Wi-Fi links, while the
vertical dimension reveals feature correlation among different time slots.
Based on the above principle, we propose the Simultaneous Tracking And
Predicting (STAP) algorithm, which enables the seamless transfer of Wi-Fi
features over time and across different links, akin to passing a baton. We
implement the system on commercial devices, and the experimental results show
that our system outperforms existing solutions with a median tracking error of
0.46m, even when the communication duty cycle is as low as 20.00%. Compared
with the state-of-the-art, our system reduces the tracking error by 79.19% in
scenarios with severe Wi-Fi feature deficiencies.

### 3. [OLAF: Programmable Data Plane Acceleration for Asynchronous Distributed Reinforcement Learning](http://arxiv.org/pdf/2507.05876v1)

Authors: Nehal Baganal Krishna, Anam Tahir, Firas Khamis, Mina Tahmasbi Arashloo, Michael Zink, Amr Rizk

Asynchronous Distributed Reinforcement Learning (DRL) can suffer from
degraded convergence when model updates become stale, often the result of
network congestion and packet loss during large-scale training. This work
introduces a network data-plane acceleration architecture that mitigates such
staleness by enabling inline processing of DRL model updates as they traverse
the accelerator engine. To this end, we design and prototype a novel queueing
mechanism that opportunistically combines compatible updates sharing a network
element, reducing redundant traffic and preserving update utility.
Complementing this we provide a lightweight transmission control mechanism at
the worker nodes that is guided by feedback from the in-network accelerator. To
assess model utility at line rate, we introduce the Age-of-Model (AoM) metric
as a proxy for staleness and verify global fairness and responsiveness
properties using a formal verification method. Our evaluations demonstrate that
this architecture significantly reduces update staleness and congestion,
ultimately improving the convergence rate in asynchronous DRL workloads.

### 4. [A Satellite-Ground Synergistic Large Vision-Language Model System for Earth Observation](http://arxiv.org/pdf/2507.05731v1)

Authors: Yuxin Zhang, Jiahao Yang, Zhe Chen, Wenjun Zhu, Jin Zhao, Yue Gao

Recently, large vision-language models (LVLMs) unleash powerful analysis
capabilities for low Earth orbit (LEO) satellite Earth observation images in
the data center. However, fast satellite motion, brief satellite-ground station
(GS) contact windows, and large size of the images pose a data download
challenge. To enable near real-time Earth observation applications (e.g.,
disaster and extreme weather monitoring), we should explore how to deploy LVLM
in LEO satellite networks, and design SpaceVerse, an efficient satellite-ground
synergistic LVLM inference system. To this end, firstly, we deploy compact
LVLMs on satellites for lightweight tasks, whereas regular LVLMs operate on GSs
to handle computationally intensive tasks. Then, we propose a computing and
communication co-design framework comprised of a progressive confidence network
and an attention-based multi-scale preprocessing, used to identify on-satellite
inferring data, and reduce data redundancy before satellite-GS transmission,
separately. We implement and evaluate SpaceVerse on real-world LEO satellite
constellations and datasets, achieving a 31.2% average gain in accuracy and a
51.2% reduction in latency compared to state-of-the-art baselines.

### 5. [Intra-DP: A High Performance Collaborative Inference System for Mobile Edge Computing](http://arxiv.org/pdf/2507.05829v1)

Authors: Zekai Sun, Xiuxian Guan, Zheng Lin, Zihan Fang, Xiangming Cai, Zhe Chen, Fangming Liu, Heming Cui, Jie Xiong, Wei Ni, Chau Yuen

Deploying deep neural networks (DNNs) on resource-constrained mobile devices
presents significant challenges, particularly in achieving real-time
performance while simultaneously coping with limited computational resources
and battery life. While Mobile Edge Computing (MEC) offers collaborative
inference with GPU servers as a promising solution, existing approaches
primarily rely on layer-wise model partitioning and undergo significant
transmission bottlenecks caused by the sequential execution of DNN operations.
To address this challenge, we present Intra-DP, a high-performance
collaborative inference system optimized for DNN inference on MEC. Intra DP
employs a novel parallel computing technique based on local operators (i.e.,
operators whose minimum unit input is not the entire input tensor, such as the
convolution kernel). By decomposing their computations (operations) into
several independent sub-operations and overlapping the computation and
transmission of different sub-operations through parallel execution, Intra-DP
mitigates transmission bottlenecks in MEC, achieving fast and energy-efficient
inference. The evaluation demonstrates that Intra-DP reduces per-inference
latency by up to 50% and energy consumption by up to 75% compared to
state-of-the-art baselines, without sacrificing accuracy.

### Robotics

### 1. [PAPRLE (Plug-And-Play Robotic Limb Environment): A Modular Ecosystem for Robotic Limbs](http://arxiv.org/pdf/2507.05555v1)

Authors: Obin Kwon, Sankalp Yamsani, Noboru Myers, Sean Taylor, Jooyoung Hong, Kyungseo Park, Alex Alspach, Joohyung Kim

We introduce PAPRLE (Plug-And-Play Robotic Limb Environment), a modular
ecosystem that enables flexible placement and control of robotic limbs. With
PAPRLE, a user can change the arrangement of the robotic limbs, and control
them using a variety of input devices, including puppeteers, gaming
controllers, and VR-based interfaces. This versatility supports a wide range of
teleoperation scenarios and promotes adaptability to different task
requirements. To further enhance configurability, we introduce a pluggable
puppeteer device that can be easily mounted and adapted to match the target
robot configurations. PAPRLE supports bilateral teleoperation through these
puppeteer devices, agnostic to the type or configuration of the follower robot.
By supporting both joint-space and task-space control, the system provides
real-time force feedback, improving user fidelity and physical interaction
awareness. The modular design of PAPRLE facilitates novel spatial arrangements
of the limbs and enables scalable data collection, thereby advancing research
in embodied AI and learning-based control. We validate PAPRLE in various
real-world settings, demonstrating its versatility across diverse combinations
of leader devices and follower robots. The system will be released as open
source, including both hardware and software components, to support broader
adoption and community-driven extension. Additional resources and
demonstrations are available at the project website:
https://uiuckimlab.github.io/paprle-pages

### 2. [Structured Task Solving via Modular Embodied Intelligence: A Case Study on Rubik's Cube](http://arxiv.org/pdf/2507.05607v1)

Authors: Chongshan Fan, Shenghai Yuan

This paper presents Auto-RubikAI, a modular autonomous planning framework
that integrates a symbolic Knowledge Base (KB), a vision-language model (VLM),
and a large language model (LLM) to solve structured manipulation tasks
exemplified by Rubik's Cube restoration. Unlike traditional robot systems based
on predefined scripts, or modern approaches relying on pretrained networks and
large-scale demonstration data, Auto-RubikAI enables interpretable, multi-step
task execution with minimal data requirements and no prior demonstrations. The
proposed system employs a KB module to solve group-theoretic restoration steps,
overcoming LLMs' limitations in symbolic reasoning. A VLM parses RGB-D input to
construct a semantic 3D scene representation, while the LLM generates
structured robotic control code via prompt chaining. This tri-module
architecture enables robust performance under spatial uncertainty. We deploy
Auto-RubikAI in both simulation and real-world settings using a 7-DOF robotic
arm, demonstrating effective Sim-to-Real adaptation without retraining.
Experiments show a 79% end-to-end task success rate across randomized
configurations. Compared to CFOP, DeepCubeA, and Two-Phase baselines, our
KB-enhanced method reduces average solution steps while maintaining
interpretability and safety. Auto-RubikAI provides a cost-efficient, modular
foundation for embodied task planning in smart manufacturing, robotics
education, and autonomous execution scenarios. Code, prompts, and hardware
modules will be released upon publication.

### 3. [A Physics-Based Continuum Model for Versatile, Scalable, and Fast Terramechanics Simulation](http://arxiv.org/pdf/2507.05643v1)

Authors: Huzaifa Unjhawala, Luning Bakke, Harry Zhang, Michael Taylor, Ganesh Arivoli, Radu Serban, Dan Negrut

This paper discusses Chrono's Continuous Representation Model (called herein
Chrono::CRM), a general-purpose, scalable, and efficient simulation solution
for terramechanics problems. Built on Chrono's Smoothed Particle Hydrodynamics
(SPH) framework, Chrono::CRM moves beyond semi-empirical terramechanics
approaches, e.g., Bekker-Wong/Janosi-Hanamoto, to provide a physics-based model
able to address complex tasks such as digging, grading, as well as interaction
with deformable wheels and complex grouser/lug patterns. The terramechanics
model is versatile in that it allows the terrain to interact with both rigid
and flexible implements simulated via the Chrono dynamics engine. We validate
Chrono::CRM against experimental data from three physical tests, including one
involving NASA's MGRU3 rover. In addition, the simulator is benchmarked against
a high-fidelity Discrete Element Method (DEM) simulation of a digging scenario
involving the Regolith Advanced Surface Systems Operations Robot (RASSOR).
Being GPU-accelerated, Chrono::CRM achieves computational efficiency comparable
to that of semi-empirical simulation approaches for terramechanics problems.
Through an ``active domains'' implementation, Chrono::CRM can handle terrain
stretches up to 10 km long with 100 million SPH particles at near interactive
rates, making high-fidelity off-road simulations at large scales feasible. As a
component of the Chrono package, the CRM model is open source and released
under a BSD-3 license. All models and simulations used in this contribution are
available in a public GitHub repository for reproducibility studies and further
research.

### 4. [Stable Tracking-in-the-Loop Control of Cable-Driven Surgical Manipulators under Erroneous Kinematic Chains](http://arxiv.org/pdf/2507.05663v1)

Authors: Neelay Joglekar, Fei Liu, Florian Richter, Michael C. Yip

Remote Center of Motion (RCM) robotic manipulators have revolutionized
Minimally Invasive Surgery, enabling precise, dexterous surgical manipulation
within the patient's body cavity without disturbing the insertion point on the
patient. Accurate RCM tool control is vital for incorporating autonomous
subtasks like suturing, blood suction, and tumor resection into robotic
surgical procedures, reducing surgeon fatigue and improving patient outcomes.
However, these cable-driven systems are subject to significant joint reading
errors, corrupting the kinematics computation necessary to perform control.
Although visual tracking with endoscopic cameras can correct errors on in-view
joints, errors in the kinematic chain prior to the insertion point are
irreparable because they remain out of view. No prior work has characterized
the stability of control under these conditions. We fill this gap by designing
a provably stable tracking-in-the-loop controller for the out-of-view portion
of the RCM manipulator kinematic chain. We additionally incorporate this
controller into a bilevel control scheme for the full kinematic chain. We
rigorously benchmark our method in simulated and real world settings to verify
our theoretical findings. Our work provides key insights into the next steps
required for the transition from teleoperated to autonomous surgery.

### 5. [Integrating Diffusion-based Multi-task Learning with Online Reinforcement Learning for Robust Quadruped Robot Control](http://arxiv.org/pdf/2507.05674v1)

Authors: Xinyao Qin, Xiaoteng Ma, Yang Qi, Qihan Liu, Chuanyi Xue, Ning Gui, Qinyu Dong, Jun Yang, Bin Liang

Recent research has highlighted the powerful capabilities of imitation
learning in robotics. Leveraging generative models, particularly diffusion
models, these approaches offer notable advantages such as strong multi-task
generalization, effective language conditioning, and high sample efficiency.
While their application has been successful in manipulation tasks, their use in
legged locomotion remains relatively underexplored, mainly due to compounding
errors that affect stability and difficulties in task transition under limited
data. Online reinforcement learning (RL) has demonstrated promising results in
legged robot control in the past years, providing valuable insights to address
these challenges. In this work, we propose DMLoco, a diffusion-based framework
for quadruped robots that integrates multi-task pretraining with online PPO
finetuning to enable language-conditioned control and robust task transitions.
Our approach first pretrains the policy on a diverse multi-task dataset using
diffusion models, enabling language-guided execution of various skills. Then,
it finetunes the policy in simulation to ensure robustness and stable task
transition during real-world deployment. By utilizing Denoising Diffusion
Implicit Models (DDIM) for efficient sampling and TensorRT for optimized
deployment, our policy runs onboard at 50Hz, offering a scalable and efficient
solution for adaptive, language-guided locomotion on resource-constrained
robotic platforms.

### 6. [Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning](http://arxiv.org/pdf/2507.05695v1)

Authors: Xiatao Sun, Yuxuan Wang, Shuo Yang, Yinxing Chen, Daniel Rakita

Diffusion policies have become increasingly popular in robot learning due to
their reliable convergence in motion generation tasks. At a high level, these
policies learn to transform noisy action trajectories into effective ones,
conditioned on observations. However, each time such a model is trained in a
robotics context, the network must relearn fundamental spatial representations
and operations, such as translations and rotations, from scratch in order to
ground itself and operate effectively in a 3D environment. Incorporating
geometric inductive biases directly into the network can alleviate this
redundancy and substantially improve training efficiency. In this paper, we
introduce hPGA-DP, a diffusion policy approach that integrates a mathematical
framework called Projective Geometric Algebra (PGA) to embed strong geometric
inductive biases. PGA is particularly well-suited for this purpose as it
provides a unified algebraic framework that naturally encodes geometric
primitives, such as points, directions, and rotations, enabling neural networks
to reason about spatial structure through interpretable and composable
operations. Specifically, we propose a novel diffusion policy architecture that
incorporates the Projective Geometric Algebra Transformer (P-GATr), leveraging
its E(3)-equivariant properties established in prior work. Our approach adopts
a hybrid architecture strategy, using P-GATr as both a state encoder and action
decoder, while employing U-Net or Transformer-based modules for the denoising
process. Several experiments and ablation studies in both simulated and
real-world environments demonstrate that hPGA-DP not only improves task
performance and training efficiency through the geometric bias of P-GATr, but
also achieves substantially faster convergence through its hybrid model
compared to architectures that rely solely on P-GATr.

### 7. [DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving](http://arxiv.org/pdf/2507.05710v1)

Authors: Hyeongchan Ham, Heejin Ahn

Safety is a critical concern in motion planning for autonomous vehicles.
Modern autonomous vehicles rely on neural network-based perception, but making
control decisions based on these inference results poses significant safety
risks due to inherent uncertainties. To address this challenge, we present a
distributionally robust optimization (DRO) framework that accounts for both
aleatoric and epistemic perception uncertainties using evidential deep learning
(EDL). Our approach introduces a novel ambiguity set formulation based on
evidential distributions that dynamically adjusts the conservativeness
according to perception confidence levels. We integrate this uncertainty-aware
constraint into model predictive control (MPC), proposing the DRO-EDL-MPC
algorithm with computational tractability for autonomous driving applications.
Validation in the CARLA simulator demonstrates that our approach maintains
efficiency under high perception confidence while enforcing conservative
constraints under low confidence.

### 8. [Simultaneous Triggering and Synchronization of Sensors and Onboard Computers](http://arxiv.org/pdf/2507.05717v1)

Authors: Morten Nissov, Nikhil Khedekar, Kostas Alexis

High fidelity estimation algorithms for robotics require accurate data.
However, timestamping of sensor data is a key issue that rarely receives the
attention it deserves. Inaccurate timestamping can be compensated for in
post-processing but is imperative for online estimation. Simultaneously, even
online mitigation of timing issues can be achieved through a relaxation of the
tuning parameters from their otherwise more performative optimal values, but at
a detriment to performance. To address the need for real-time, low-cost
timestamping, a versatile system which utilizes readily-available components
and established methods for synchronization is introduced. The synchronization
and triggering (of both high- and low-rate sensors) capabilities of the system
are demonstrated.

### 9. [A Learning-based Planning and Control Framework for Inertia Drift Vehicles](http://arxiv.org/pdf/2507.05748v1)

Authors: Bei Zhou, Zhouheng Li, Lei Xie, Hongye Su, Johannes Betz

Inertia drift is a transitional maneuver between two sustained drift stages
in opposite directions, which provides valuable insights for navigating
consecutive sharp corners for autonomous racing.However, this can be a
challenging scenario for the drift controller to handle rapid transitions
between opposing sideslip angles while maintaining accurate path tracking.
Moreover, accurate drift control depends on a high-fidelity vehicle model to
derive drift equilibrium points and predict vehicle states, but this is often
compromised by the strongly coupled longitudinal-lateral drift dynamics and
unpredictable environmental variations. To address these challenges, this paper
proposes a learning-based planning and control framework utilizing Bayesian
optimization (BO), which develops a planning logic to ensure a smooth
transition and minimal velocity loss between inertia and sustained drift
phases. BO is further employed to learn a performance-driven control policy
that mitigates modeling errors for enhanced system performance. Simulation
results on an 8-shape reference path demonstrate that the proposed framework
can achieve smooth and stable inertia drift through sharp corners.

### 10. [FineGrasp: Towards Robust Grasping for Delicate Objects](http://arxiv.org/pdf/2507.05978v1)

Authors: Yun Du, Mengao Zhao, Tianwei Lin, Yiwei Jin, Chaodong Huang, Zhizhong Su

Recent advancements in robotic grasping have led to its integration as a core
module in many manipulation systems. For instance, language-driven semantic
segmentation enables the grasping of any designated object or object part.
However, existing methods often struggle to generate feasible grasp poses for
small objects or delicate components, potentially causing the entire pipeline
to fail. To address this issue, we propose a novel grasping method, FineGrasp,
which introduces improvements in three key aspects. First, we introduce
multiple network modifications to enhance the ability of to handle delicate
regions. Second, we address the issue of label imbalance and propose a refined
graspness label normalization strategy. Third, we introduce a new simulated
grasp dataset and show that mixed sim-to-real training further improves grasp
performance. Experimental results show significant improvements, especially in
grasping small objects, and confirm the effectiveness of our system in semantic
grasping.

### Software Engineering

### 1. [Multi-Agent Debate Strategies to Enhance Requirements Engineering with Large Language Models](http://arxiv.org/pdf/2507.05981v1)

Authors: Marc Oriol, Quim Motger, Jordi Marco, Xavier Franch

Context: Large Language Model (LLM) agents are becoming widely used for
various Requirements Engineering (RE) tasks. Research on improving their
accuracy mainly focuses on prompt engineering, model fine-tuning, and retrieval
augmented generation. However, these methods often treat models as isolated
black boxes - relying on single-pass outputs without iterative refinement or
collaboration, limiting robustness and adaptability. Objective: We propose
that, just as human debates enhance accuracy and reduce bias in RE tasks by
incorporating diverse perspectives, different LLM agents debating and
collaborating may achieve similar improvements. Our goal is to investigate
whether Multi-Agent Debate (MAD) strategies can enhance RE performance. Method:
We conducted a systematic study of existing MAD strategies across various
domains to identify their key characteristics. To assess their applicability in
RE, we implemented and tested a preliminary MAD-based framework for RE
classification. Results: Our study identified and categorized several MAD
strategies, leading to a taxonomy outlining their core attributes. Our
preliminary evaluation demonstrated the feasibility of applying MAD to RE
classification. Conclusions: MAD presents a promising approach for improving
LLM accuracy in RE tasks. This study provides a foundational understanding of
MAD strategies, offering insights for future research and refinements in RE
applications.

### 2. [PromiseTune: Unveiling Causally Promising and Explainable Configuration Tuning](http://arxiv.org/pdf/2507.05995v1)

Authors: Pengzhou Chen, Tao Chen

The high configurability of modern software systems has made configuration
tuning a crucial step for assuring system performance, e.g., latency or
throughput. However, given the expensive measurements, large configuration
space, and rugged configuration landscape, existing tuners suffer
ineffectiveness due to the difficult balance of budget utilization between
exploring uncertain regions (for escaping from local optima) and exploiting
guidance of known good configurations (for fast convergence). The root cause is
that we lack knowledge of where the promising regions lay, which also causes
challenges in the explainability of the results.
  In this paper, we propose PromiseTune that tunes configuration guided by
causally purified rules. PromiseTune is unique in the sense that we learn
rules, which reflect certain regions in the configuration landscape, and purify
them with causal inference. The remaining rules serve as approximated
reflections of the promising regions, bounding the tuning to emphasize these
places in the landscape. This, as we demonstrate, can effectively mitigate the
impact of the exploration and exploitation trade-off. Those purified regions
can then be paired with the measured configurations to provide spatial
explainability at the landscape level. Comparing with 11 state-of-the-art
tuners on 12 systems and varying budgets, we show that PromiseTune performs
significantly better than the others with $42\%$ superior rank to the overall
second best while providing richer information to explain the hidden system
characteristics.

### 3. [Model Cards Revisited: Bridging the Gap Between Theory and Practice for Ethical AI Requirements](http://arxiv.org/pdf/2507.06014v1)

Authors: Tim Puhlfürß, Julia Butzke, Walid Maalej

Model cards are the primary documentation framework for developers of
artificial intelligence (AI) models to communicate critical information to
their users. Those users are often developers themselves looking for relevant
documentation to ensure that their AI systems comply with the ethical
requirements of existing laws, guidelines, and standards. Recent studies
indicate inadequate model documentation practices, suggesting a gap between AI
requirements and current practices in model documentation. To understand this
gap and provide actionable guidance to bridge it, we conducted a thematic
analysis of 26 guidelines on ethics and AI, three AI documentation frameworks,
three quantitative studies of model cards, and ten actual model cards. We
identified a total of 43 ethical requirements relevant to model documentation
and organized them into a taxonomy featuring four themes and twelve sub-themes
representing ethical principles. Our findings indicate that model developers
predominantly emphasize model capabilities and reliability in the documentation
while overlooking other ethical aspects, such as explainability, user autonomy,
and fairness. This underscores the need for enhanced support in documenting
ethical AI considerations. Our taxonomy serves as a foundation for a revised
model card framework that holistically addresses ethical AI requirements.

### 4. [Detecting and Mitigating Reward Hacking in Reinforcement Learning Systems: A Comprehensive Empirical Study](http://arxiv.org/pdf/2507.05619v1)

Authors: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma

Reward hacking in Reinforcement Learning (RL) systems poses a critical threat
to the deployment of autonomous agents, where agents exploit flaws in reward
functions to achieve high scores without fulfilling intended objectives.
Despite growing awareness of this problem, systematic detection and mitigation
approaches remain limited. This paper presents a large-scale empirical study of
reward hacking across diverse RL environments and algorithms. We analyze 15,247
training episodes across 15 RL environments (Atari, MuJoCo, custom domains) and
5 algorithms (PPO, SAC, DQN, A3C, Rainbow), implementing automated detection
algorithms for six categories of reward hacking: specification gaming, reward
tampering, proxy optimization, objective misalignment, exploitation patterns,
and wireheading. Our detection framework achieves 78.4% precision and 81.7%
recall across environments, with computational overhead under 5%. Through
controlled experiments varying reward function properties, we demonstrate that
reward density and alignment with true objectives significantly impact hacking
frequency ($p < 0.001$, Cohen's $d = 1.24$). We validate our approach through
three simulated application studies representing recommendation systems,
competitive gaming, and robotic control scenarios. Our mitigation techniques
reduce hacking frequency by up to 54.6% in controlled scenarios, though we find
these trade-offs are more challenging in practice due to concept drift, false
positive costs, and adversarial adaptation. All detection algorithms, datasets,
and experimental protocols are publicly available to support reproducible
research in RL safety.

### 5. [TigAug: Data Augmentation for Testing Traffic Light Detection in Autonomous Driving Systems](http://arxiv.org/pdf/2507.05932v1)

Authors: You Lu, Dingji Wang, Kaifeng Huang, Bihuan Chen, Xin Peng

Autonomous vehicle technology has been developed in the last decades with
recent advances in sensing and computing technology. There is an urgent need to
ensure the reliability and robustness of autonomous driving systems (ADSs).
Despite the recent achievements in testing various ADS modules, little
attention has been paid on the automated testing of traffic light detection
models in ADSs. A common practice is to manually collect and label traffic
light data. However, it is labor-intensive, and even impossible to collect
diverse data under different driving environments.
  To address these problems, we propose and implement TigAug to automatically
augment labeled traffic light images for testing traffic light detection models
in ADSs. We construct two families of metamorphic relations and three families
of transformations based on a systematic understanding of weather environments,
camera properties, and traffic light properties. We use augmented images to
detect erroneous behaviors of traffic light detection models by
transformation-specific metamorphic relations, and to improve the performance
of traffic light detection models by retraining. Large-scale experiments with
four state-of-the-art traffic light detection models and two traffic light
datasets have demonstrated that i) TigAug is effective in testing traffic light
detection models, ii) TigAug is efficient in synthesizing traffic light images,
and iii) TigAug generates traffic light images with acceptable naturalness.

### 6. [Search-based Selection of Metamorphic Relations for Optimized Robustness Testing of Large Language Models](http://arxiv.org/pdf/2507.05565v1)

Authors: Sangwon Hyun, Shaukat Ali, M. Ali Babar

Assessing the trustworthiness of Large Language Models (LLMs), such as
robustness, has garnered significant attention. Recently, metamorphic testing
that defines Metamorphic Relations (MRs) has been widely applied to evaluate
the robustness of LLM executions. However, the MR-based robustness testing
still requires a scalable number of MRs, thereby necessitating the optimization
of selecting MRs. Most extant LLM testing studies are limited to automatically
generating test cases (i.e., MRs) to enhance failure detection. Additionally,
most studies only considered a limited test space of single perturbation MRs in
their evaluation of LLMs. In contrast, our paper proposes a search-based
approach for optimizing the MR groups to maximize failure detection and
minimize the LLM execution cost. Moreover, our approach covers the
combinatorial perturbations in MRs, facilitating the expansion of test space in
the robustness assessment. We have developed a search process and implemented
four search algorithms: Single-GA, NSGA-II, SPEA2, and MOEA/D with novel
encoding to solve the MR selection problem in the LLM robustness testing. We
conducted comparative experiments on the four search algorithms along with a
random search, using two major LLMs with primary Text-to-Text tasks. Our
statistical and empirical investigation revealed two key findings: (1) the
MOEA/D algorithm performed the best in optimizing the MR space for LLM
robustness testing, and (2) we identified silver bullet MRs for the LLM
robustness testing, which demonstrated dominant capabilities in confusing LLMs
across different Text-to-Text tasks. In LLM robustness assessment, our research
sheds light on the fundamental problem for optimized testing and provides
insights into search-based solutions.

### 7. [Prompt Migration: Stabilizing GenAI Applications with Evolving Large Language Models](http://arxiv.org/pdf/2507.05573v1)

Authors: Shivani Tripathi, Pushpanjali Nema, Aditya Halder, Shi Qiao, Alekh Jindal

Generative AI is transforming business applications by enabling natural
language interfaces and intelligent automation. However, the underlying large
language models (LLMs) are evolving rapidly and so prompting them consistently
is a challenge. This leads to inconsistent and unpredictable application
behavior, undermining the reliability that businesses require for
mission-critical workflows. In this paper, we introduce the concept of prompt
migration as a systematic approach to stabilizing GenAI applications amid
changing LLMs. Using the Tursio enterprise search application as a case study,
we analyze the impact of successive GPT model upgrades, detail our migration
framework including prompt redesign and a migration testbed, and demonstrate
how these techniques restore application consistency. Our results show that
structured prompt migration can fully recover the application reliability that
was lost due to model drift. We conclude with practical lessons learned,
emphasizing the need for prompt lifecycle management and robust testing to
ensure dependable GenAI-powered business applications.

### Social and Information Networks

### 1. [The most influential philosophers in Wikipedia: a multicultural analysis](http://arxiv.org/pdf/2507.06034v1)

Authors: Guillaume Rollin, José Lages

We explore the influence and interconnectivity of philosophical thinkers
within the Wikipedia knowledge network. Using a dataset of 237 articles
dedicated to philosophers across nine different language editions (Arabic,
Chinese, English, French, German, Japanese, Portuguese, Russian, and Spanish),
we apply the PageRank and CheiRank algorithms to analyze their relative ranking
and influence in each linguistic context. Furthermore, we compare our results
with entries from the Stanford Encyclopedia of Philosophy and the Internet
Encyclopedia of Philosophy, providing insight into the differences between
general knowledge networks like Wikipedia and specialized philosophical
databases. A key focus of our analysis is the sub-network of 21 presocratic
philosophers, grouped into four traditional schools: Italic (Pythagorean +
Eleatic), Ionian, Abderian (Atomist), and Sophist. Using the reduced Google
matrix method, we uncover both direct and hidden links between these early
thinkers, offering new perspectives on their intellectual relationships and
influence within the Western philosophical tradition.

### 2. [QuHE: Optimizing Utility-Cost in Quantum Key Distribution and Homomorphic Encryption Enabled Secure Edge Computing Networks](http://arxiv.org/pdf/2507.06086v1)

Authors: Liangxin Qian, Yang Li, Jun Zhao

Ensuring secure and efficient data processing in mobile edge computing (MEC)
systems is a critical challenge. While quantum key distribution (QKD) offers
unconditionally secure key exchange and homomorphic encryption (HE) enables
privacy-preserving data processing, existing research fails to address the
comprehensive trade-offs among QKD utility, HE security, and system costs. This
paper proposes a novel framework integrating QKD, transciphering, and HE for
secure and efficient MEC. QKD distributes symmetric keys, transciphering
bridges symmetric encryption, and HE processes encrypted data at the server. We
formulate an optimization problem balancing QKD utility, HE security,
processing and wireless transmission costs. However, the formulated
optimization is non-convex and NPhard. To solve it efficiently, we propose the
Quantum-enhanced Homomorphic Encryption resource allocation (QuHE) algorithm.
Theoretical analysis proves the proposed QuHE algorithm's convergence and
optimality, and simulations demonstrate its effectiveness across multiple
performance metrics.

### 3. [LLMs are Introvert](http://arxiv.org/pdf/2507.05638v1)

Authors: Litian Zhang, Xiaoming Zhang, Bingyu Yan, Ziyi Zhou, Bo Zhang, Zhenyu Guan, Xi Zhang, Chaozhuo Li

The exponential growth of social media and generative AI has transformed
information dissemination, fostering connectivity but also accelerating the
spread of misinformation. Understanding information propagation dynamics and
developing effective control strategies is essential to mitigate harmful
content. Traditional models, such as SIR, provide basic insights but
inadequately capture the complexities of online interactions. Advanced methods,
including attention mechanisms and graph neural networks, enhance accuracy but
typically overlook user psychology and behavioral dynamics. Large language
models (LLMs), with their human-like reasoning, offer new potential for
simulating psychological aspects of information spread. We introduce an
LLM-based simulation environment capturing agents' evolving attitudes,
emotions, and responses. Initial experiments, however, revealed significant
gaps between LLM-generated behaviors and authentic human dynamics, especially
in stance detection and psychological realism. A detailed evaluation through
Social Information Processing Theory identified major discrepancies in
goal-setting and feedback evaluation, stemming from the lack of emotional
processing in standard LLM training. To address these issues, we propose the
Social Information Processing-based Chain of Thought (SIP-CoT) mechanism
enhanced by emotion-guided memory. This method improves the interpretation of
social cues, personalization of goals, and evaluation of feedback. Experimental
results confirm that SIP-CoT-enhanced LLM agents more effectively process
social information, demonstrating behaviors, attitudes, and emotions closer to
real human interactions. In summary, this research highlights critical
limitations in current LLM-based propagation simulations and demonstrates how
integrating SIP-CoT and emotional memory significantly enhances the social
intelligence and realism of LLM agents.

### 4. [Identity isn't everything -- how far do demographics take us towards self-identified party ID?](http://arxiv.org/pdf/2507.06193v1)

Authors: Sabina Tomkins, David Rothschild, Alex Liu, Alexander Thompson

How well do demographics explain party identification? Demographics are
related to party identification in political polls, news articles, and academic
publications. Yet, there is a diversity of party identification even within
demographic groups which have historically been attached to one party. And some
groups lack a clear connection to either party. It may be that demographics on
their own fail to account for the fact that people generally belong to a
variety of groups. They must select the groups which are most important to them
when shaping a political identity, and may choose to construct an identity
relatively unattached to any specific demographic group to which they belong.
This prompts the question, do we need to consider measures of identity strength
when using demographics to explain party identification? We utilize a
predictive framework to address these questions and find that demographics are
highly predictive for some groups (e.g., Black Democrats), while others benefit
from the inclusion of identity strength (e.g., Hispanic Republicans).

### 5. [Critical Nodes Identification in Complex Networks: A Survey](http://arxiv.org/pdf/2507.06164v1)

Authors: Duxin Chen, Jiawen Chen, Xiaoyu Zhang, Qinghan Jia, Xiaolu Liu, Ye Sun, Linyuan Lv, Wenwu Yu

Complex networks have become essential tools for understanding diverse
phenomena in social systems, traffic systems, biomolecular systems, and
financial systems. Identifying critical nodes is a central theme in
contemporary research, serving as a vital bridge between theoretical
foundations and practical applications. Nevertheless, the intrinsic complexity
and structural heterogeneity characterizing real-world networks, with
particular emphasis on dynamic and higher-order networks, present substantial
obstacles to the development of universal frameworks for critical node
identification. This paper provides a comprehensive review of critical node
identification techniques, categorizing them into seven main classes:
centrality, critical nodes deletion problem, influence maximization, network
control, artificial intelligence, higher-order and dynamic methods. Our review
bridges the gaps in existing surveys by systematically classifying methods
based on their methodological foundations and practical implications, and by
highlighting their strengths, limitations, and applicability across different
network types. Our work enhances the understanding of critical node research by
identifying key challenges, such as algorithmic universality, real-time
evaluation in dynamic networks, analysis of higher-order structures, and
computational efficiency in large-scale networks. The structured synthesis
consolidates current progress and highlights open questions, particularly in
modeling temporal dynamics, advancing efficient algorithms, integrating machine
learning approaches, and developing scalable and interpretable metrics for
complex systems.

### Systems and Control

### 1. [Low voltage user phase reconfiguration as a planning problem](http://arxiv.org/pdf/2507.05910v1)

Authors: Sari Kerckhove, Marta Vanin, Reinhilde D'hulst, Dirk Van Hertem

Considerable levels of phase imbalance in low voltage (LV) distribution
networks imply that grid assets are suboptimally utilized and can cause
additional losses, equipment failure and degradation. With the ongoing energy
transition, the installation of additional single-phase distributed energy
resources may further increase the phase imbalance if no countermeasures are
taken.
  Phase reconfiguration is a cost-effective solution to reduce imbalance.
However, dynamic reconfiguration, through real-time phase swapping of loads
using remotely controlled switches, is often impractical because these switches
are too costly for widespread installation at LV users. Approaching phase
reconfiguration as a planning problem, i.e. static reconfiguration, is an
underaddressed but promising alternative. Effective static approaches that
allow appropriate imbalance objectives are currently lacking.
  This paper presents reliable and expressive static phase reconfiguration
methods that grid operators can easily integrate into routine maintenance for
effective phase balancing.
  We present and compare three static methods, an exact mixed-integer nonlinear
formulation (MINLP), a mixed-integer quadratic approximation (MIQP), and a
genetic algorithm (GA), each supporting different imbalance objectives. The
MIQP approach, despite using proxy objectives, efficiently mitigates the
different types of imbalance considered, and outperforms both MINLP and GA in
scalability and consistency.

### 2. [Sparsity-Promoting Dynamic Mode Decomposition Applied to Sea Surface Temperature Fields](http://arxiv.org/pdf/2507.05711v1)

Authors: Zhicheng Zhang, Yoshihiko Susuki, Atsushi Okazaki

In this paper, we leverage Koopman mode decomposition to analyze the
nonlinear and high-dimensional climate systems acting on the observed data
space. The dynamics of atmospheric systems are assumed to be equation-free,
with the linear evolution of observables derived from measured historical
long-term time-series data snapshots, such as monthly sea surface temperature
records, to construct a purely data-driven climate dynamics. In particular,
sparsity-promoting dynamic mode decomposition is exploited to extract the
dominant spatial and temporal modes, which are among the most significant
coherent structures underlying climate variability, enabling a more efficient,
interpretable, and low-dimensional representation of the system dynamics. We
hope that the combined use of Koopman modes and sparsity-promoting techniques
will provide insights into the significant climate modes, enabling
reduced-order modeling of the climate system and offering a potential framework
for predicting and controlling weather and climate variability.

### 3. [Robust Bandwidth Estimation for Real-Time Communication with Offline Reinforcement Learning](http://arxiv.org/pdf/2507.05785v1)

Authors: Jian Kai, Tianwei Zhang, Zihan Ling, Yang Cao, Can Shen

Accurate bandwidth estimation (BWE) is critical for real-time communication
(RTC) systems. Traditional heuristic approaches offer limited adaptability
under dynamic networks, while online reinforcement learning (RL) suffers from
high exploration costs and potential service disruptions. Offline RL, which
leverages high-quality data collected from real-world environments, offers a
promising alternative. However, challenges such as out-of-distribution (OOD)
actions, policy extraction from behaviorally diverse datasets, and reliable
deployment in production systems remain unsolved. We propose RBWE, a robust
bandwidth estimation framework based on offline RL that integrates Q-ensemble
(an ensemble of Q-functions) with a Gaussian mixture policy to mitigate OOD
risks and enhance policy learning. A fallback mechanism ensures deployment
stability by switching to heuristic methods under high uncertainty.
Experimental results show that RBWE reduces overestimation errors by 18% and
improves the 10th percentile Quality of Experience (QoE) by 18.6%,
demonstrating its practical effectiveness in real-world RTC applications.

### 4. [Assessing Linear Control Strategies for Zero-Speed Fin Roll Damping](http://arxiv.org/pdf/2507.05867v1)

Authors: Nikita Savin, Elena Ambrosovskaya, Dmitry Romaev, Anton Proskurnikov

Roll stabilization is a critical aspect of ship motion control, particularly
for vessels operating in low-speed or zero-speed conditions, where traditional
hydrodynamic fins lose their effectiveness. In this paper, we consider a roll
damping system, developed by Navis JSC, based on two actively controlled
zero-speed fins. Unlike conventional fin stabilizers, zero-speed fins employ a
drag-based mechanism and active oscillations to generate stabilizing forces
even when the vessel is stationary. We propose a simple linear control
architecture that, however, accounts for nonlinear drag forces and actuator
limitations. Simulation results on a high-fidelity vessel model used for HIL
testing demonstrate the effectiveness of the proposed approach.

### 5. [Robust Power System State Estimation using Physics-Informed Neural Networks](http://arxiv.org/pdf/2507.05874v1)

Authors: Solon Falas, Markos Asprou, Charalambos Konstantinou, Maria K. Michael

Modern power systems face significant challenges in state estimation and
real-time monitoring, particularly regarding response speed and accuracy under
faulty conditions or cyber-attacks. This paper proposes a hybrid approach using
physics-informed neural networks (PINNs) to enhance the accuracy and
robustness, of power system state estimation. By embedding physical laws into
the neural network architecture, PINNs improve estimation accuracy for
transmission grid applications under both normal and faulty conditions, while
also showing potential in addressing security concerns such as data
manipulation attacks. Experimental results show that the proposed approach
outperforms traditional machine learning models, achieving up to 83% higher
accuracy on unseen subsets of the training dataset and 65% better performance
on entirely new, unrelated datasets. Experiments also show that during a data
manipulation attack against a critical bus in a system, the PINN can be up to
93% more accurate than an equivalent neural network.

### 6. [Optimal Placement of Smart Hybrid Transformers in Distribution Networks](http://arxiv.org/pdf/2507.05967v1)

Authors: Samuel Hayward, Martin Doff-Sotta, Michael Merlin, Matthew Williams, Thomas Morstyn

Hybrid transformers are a relatively new technology that combine conventional
power transformers with power electronics to provide voltage and reactive power
control capabilities in distribution networks. This paper proposes a novel
method of determining the optimal location and utilisation of hybrid
transformers in 3-phase distribution networks to maximise the net present value
of hybrid transformers based on their ability to increase the export of power
produced by distributed generators over their operational lifespan. This has
been accomplished through sequential linear programming, a key feature of which
is the consideration of nonlinear characteristics and constraints relating to
hybrid transformer power electronics and control capabilities. Test cases were
carried out in a modified version of the Cigre European Low Voltage
Distribution Network Benchmark, which has been extended by connecting it with
two additional low voltage distribution test networks. All test case results
demonstrate that the installation and utilisation of hybrid transformers can
improve the income earned from exporting excess active power, justifying their
installation cost (with the highest net present value being {\pounds}6.56
million, resulting from a 45.53 percent increase in estimated annual profits
due to coordinated HT compensation).

### 7. [Fast Bilateral Teleoperation and Imitation Learning Using Sensorless Force Control via Accurate Dynamics Model](http://arxiv.org/pdf/2507.06174v1)

Authors: Koki Yamane, Yunhan Li, Masashi Konosu, Koki Inami, Junji Oaki, Sho Sakaino, Toshiaki Tsuji

In recent years, the advancement of imitation learning has led to increased
interest in teleoperating low-cost manipulators to collect demonstration data.
However, most existing systems rely on unilateral control, which only transmits
target position values. While this approach is easy to implement and suitable
for slow, non-contact tasks, it struggles with fast or contact-rich operations
due to the absence of force feedback. This work demonstrates that fast
teleoperation with force feedback is feasible even with force-sensorless,
low-cost manipulators by leveraging 4-channel bilateral control. Based on
accurately identified manipulator dynamics, our method integrates nonlinear
terms compensation, velocity and external force estimation, and variable gain
corresponding to inertial variation. Furthermore, using data collected by
4-channel bilateral control, we show that incorporating force information into
both the input and output of learned policies improves performance in imitation
learning. These results highlight the practical effectiveness of our system for
high-fidelity teleoperation and data collection on affordable hardware.

### 8. [Frequency-Specific Neural Response and Cross-Correlation Analysis of Envelope Following Responses to Native Speech and Music Using Multichannel EEG Signals: A Case Study](http://arxiv.org/pdf/2507.05635v1)

Authors: Md. Mahbub Hasan, Md Rakibul Hasan, Md Zakir Hossain, Tom Gedeon

Although native speech and music envelope following responses (EFRs) play a
crucial role in auditory processing and cognition, their frequency profile,
such as the dominating frequency and spectral coherence, is largely unknown. We
have assumed that the auditory pathway - which transmits envelope components of
speech and music to the scalp through time-varying neurophysiological processes
- is a linear time-varying system, with the envelope and the multi-channel EEG
responses as excitation and response, respectively. This paper investigates the
transfer function of this system through two analytical techniques -
time-averaged spectral responses and cross-spectral density - in the frequency
domain at four different positions of the human scalp. Our findings suggest
that alpha (8-11 Hz), lower gamma (53-56 Hz), and higher gamma (78-81 Hz) bands
are the peak responses of the system. These frequently appearing dominant
frequency responses may be the key components of familiar speech perception,
maintaining attention, binding acoustic features, and memory processing. The
cross-spectral density, which reflects the spatial neural coherence of the
human brain, shows that 10-13 Hz, 27-29 Hz, and 62-64 Hz are common for all
channel pairs. As neural coherences are frequently observed in these
frequencies among native participants, we suggest that these distributed neural
processes are also dominant in native speech and music perception.

### 9. [Automated Reasoning for Vulnerability Management by Design](http://arxiv.org/pdf/2507.05794v1)

Authors: Avi Shaked, Nan Messe

For securing systems, it is essential to manage their vulnerability posture
and design appropriate security controls. Vulnerability management allows to
proactively address vulnerabilities by incorporating pertinent security
controls into systems designs. Current vulnerability management approaches do
not support systematic reasoning about the vulnerability postures of systems
designs. To effectively manage vulnerabilities and design security controls, we
propose a formally grounded automated reasoning mechanism. We integrate the
mechanism into an open-source security design tool and demonstrate its
application through an illustrative example driven by real-world challenges.
The automated reasoning mechanism allows system designers to identify
vulnerabilities that are applicable to a specific system design, explicitly
specify vulnerability mitigation options, declare selected controls, and thus
systematically manage vulnerability postures.

### 10. [Adaptive Communication Through Exploiting RIS, SSK, and CIM for Improved Reliability and Efficiency](http://arxiv.org/pdf/2507.05813v1)

Authors: Ferhat Bayar, Onur Salan, Erdogan Aydin, Haci Ilhan

In this paper, we present a novel communication system model that integrates
reconfigurable intelligent surfaces (RIS), spatial shift keying (SSK), and code
index modulation (CIM) based on Hadamard coding called RIS based transmit
SSK-CIM (RIS-CIM-TSSK). By leveraging RIS, the system adapts rapidly to dynamic
environments, enhancing error rates and overall reliability. SSK facilitates
the transmission of additional passive information while eliminating the need
for multiple radio frequency (RF) chains, thereby reducing complexity. CIM
enhances passive information transmission through frequency domain spreading,
which may increase signal obfuscation. This proposed scheme not only improves
energy efficiency but also offers a robust solution for reliable communication
in modern wireless networks, paving the way for smarter and more adaptable
implementations. We consider a suboptimal, low-complexity detector for the
proposed scheme and also address the blind case for phase adjustment of the
RIS. Finally, we present the simulation results for the proposed system model
across various configurations, including different numbers of receive and
transmit antennas, varying reflecting elements of the RIS, and different code
lengths.

### Machine Learning (Statistics Category)

### 1. [Predicting Graph Structure via Adapted Flux Balance Analysis](http://arxiv.org/pdf/2507.05806v1)

Authors: Sevvandi Kandanaarachchi, Ziqi Xu, Stefan Westerlund, Conrad Sanderson

Many dynamic processes such as telecommunication and transport networks can
be described through discrete time series of graphs. Modelling the dynamics of
such time series enables prediction of graph structure at future time steps,
which can be used in applications such as detection of anomalies. Existing
approaches for graph prediction have limitations such as assuming that the
vertices do not to change between consecutive graphs. To address this, we
propose to exploit time series prediction methods in combination with an
adapted form of flux balance analysis (FBA), a linear programming method
originating from biochemistry. FBA is adapted to incorporate various
constraints applicable to the scenario of growing graphs. Empirical evaluations
on synthetic datasets (constructed via Preferential Attachment model) and real
datasets (UCI Message, HePH, Facebook, Bitcoin) demonstrate the efficacy of the
proposed approach.

### 2. [Prototype-Guided and Lightweight Adapters for Inherent Interpretation and Generalisation in Federated Learning](http://arxiv.org/pdf/2507.05852v1)

Authors: Samuel Ofosu Mensah, Kerol Djoumessi, Philipp Berens

Federated learning (FL) provides a promising paradigm for collaboratively
training machine learning models across distributed data sources while
maintaining privacy. Nevertheless, real-world FL often faces major challenges
including communication overhead during the transfer of large model parameters
and statistical heterogeneity, arising from non-identical independent data
distributions across clients. In this work, we propose an FL framework that 1)
provides inherent interpretations using prototypes, and 2) tackles statistical
heterogeneity by utilising lightweight adapter modules to act as compressed
surrogates of local models and guide clients to achieve generalisation despite
varying client distribution. Each client locally refines its model by aligning
class embeddings toward prototype representations and simultaneously adjust the
lightweight adapter. Our approach replaces the need to communicate entire model
weights with prototypes and lightweight adapters. This design ensures that each
client's model aligns with a globally shared structure while minimising
communication load and providing inherent interpretations. Moreover, we
conducted our experiments on a real-world retinal fundus image dataset, which
provides clinical-site information. We demonstrate inherent interpretable
capabilities and perform a classification task, which shows improvements in
accuracy over baseline algorithms.

### 3. [Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis](http://arxiv.org/pdf/2507.05913v1)

Authors: Gholamali Aminian, Idan Shenfeld, Amir R. Asadi, Ahmad Beirami, Youssef Mroueh

A simple yet effective method for inference-time alignment of generative
models is Best-of-$N$ (BoN), where $N$ outcomes are sampled from a reference
policy, evaluated using a proxy reward model, and the highest-scoring one is
selected. While prior work argues that BoN is almost optimal in reward vs KL
tradeoffs, the effectiveness of BoN depends critically on the quality of the
proxy reward model used for selection. For this purpose, we study BoN through a
smooth version known as Soft Best-of-N (SBoN) and develop a theoretical
framework to address this gap. We analyze the scaling behaviour of BoN by
providing bounds on the KL divergence between the SBoN policy and the reference
policy, offering insights into how performance varies with the number of
samples. We also study the regret gap, i.e., the gap between the expected true
reward under the optimal policy and the SBoN policy. Our theoretical and
empirical findings show that smoothing helps SBoN mitigate reward
overoptimization, especially when the quality of the proxy reward is low.

### 4. [Kernel Trace Distance: Quantum Statistical Metric between Measures through RKHS Density Operators](http://arxiv.org/pdf/2507.06055v1)

Authors: Arturo Castellanos, Anna Korba, Pavlo Mozharovskyi, Hicham Janati

Distances between probability distributions are a key component of many
statistical machine learning tasks, from two-sample testing to generative
modeling, among others. We introduce a novel distance between measures that
compares them through a Schatten norm of their kernel covariance operators. We
show that this new distance is an integral probability metric that can be
framed between a Maximum Mean Discrepancy (MMD) and a Wasserstein distance. In
particular, we show that it avoids some pitfalls of MMD, by being more
discriminative and robust to the choice of hyperparameters. Moreover, it
benefits from some compelling properties of kernel methods, that can avoid the
curse of dimensionality for their sample complexity. We provide an algorithm to
compute the distance in practice by introducing an extension of kernel matrix
for difference of distributions that could be of independent interest. Those
advantages are illustrated by robust approximate Bayesian computation under
contamination as well as particle flow simulations.

### 5. [Estimating prevalence with precision and accuracy](http://arxiv.org/pdf/2507.06061v1)

Authors: Aime Bienfait Igiraneza, Christophe Fraser, Robert Hinch

Unlike classification, whose goal is to estimate the class of each data point
in a dataset, prevalence estimation or quantification is a task that aims to
estimate the distribution of classes in a dataset. The two main tasks in
prevalence estimation are to adjust for bias, due to the prevalence in the
training dataset, and to quantify the uncertainty in the estimate. The standard
methods used to quantify uncertainty in prevalence estimates are bootstrapping
and Bayesian quantification methods. It is not clear which approach is ideal in
terms of precision (i.e. the width of confidence intervals) and coverage (i.e.
the confidence intervals being well-calibrated). Here, we propose Precise
Quantifier (PQ), a Bayesian quantifier that is more precise than existing
quantifiers and with well-calibrated coverage. We discuss the theory behind PQ
and present experiments based on simulated and real-world datasets. Through
these experiments, we establish the factors which influence quantification
precision: the discriminatory power of the underlying classifier; the size of
the labeled dataset used to train the quantifier; and the size of the unlabeled
dataset for which prevalence is estimated. Our analysis provides deep insights
into uncertainty quantification for quantification learning.

### 6. [A Malliavin calculus approach to score functions in diffusion generative models](http://arxiv.org/pdf/2507.05550v1)

Authors: Ehsan Mirafzali, Frank Proske, Utkarsh Gupta, Daniele Venturi, Razvan Marinescu

Score-based diffusion generative models have recently emerged as a powerful
tool for modelling complex data distributions. These models aim at learning the
score function, which defines a map from a known probability distribution to
the target data distribution via deterministic or stochastic differential
equations (SDEs). The score function is typically estimated from data using a
variety of approximation techniques, such as denoising or sliced score
matching, Hyv\"arien's method, or Schr\"odinger bridges. In this paper, we
derive an exact, closed form, expression for the score function for a broad
class of nonlinear diffusion generative models. Our approach combines modern
stochastic analysis tools such as Malliavin derivatives and their adjoint
operators (Skorokhod integrals or Malliavin Divergence) with a new Bismut-type
formula. The resulting expression for the score function can be written
entirely in terms of the first and second variation processes, with all
Malliavin derivatives systematically eliminated, thereby enhancing its
practical applicability. The theoretical framework presented in this work
offers a principled foundation for advancing score estimation methods in
generative modelling, enabling the design of new sampling algorithms for
complex probability distributions. Our results can be extended to broader
classes of stochastic differential equations, opening new directions for the
development of score-based diffusion generative models.

### 7. [On the Inherent Privacy of Zeroth Order Projected Gradient Descent](http://arxiv.org/pdf/2507.05610v1)

Authors: Devansh Gupta, Meisam Razaviyayn, Vatsal Sharan

Differentially private zeroth-order optimization methods have recently gained
popularity in private fine tuning of machine learning models due to their
reduced memory requirements. Current approaches for privatizing zeroth-order
methods rely on adding Gaussian noise to the estimated zeroth-order gradients.
However, since the search direction in the zeroth-order methods is inherently
random, researchers including Tang et al. (2024) and Zhang et al. (2024a) have
raised an important question: is the inherent noise in zeroth-order estimators
sufficient to ensure the overall differential privacy of the algorithm? This
work settles this question for a class of oracle-based optimization algorithms
where the oracle returns zeroth-order gradient estimates. In particular, we
show that for a fixed initialization, there exist strongly convex objective
functions such that running (Projected) Zeroth-Order Gradient Descent (ZO-GD)
is not differentially private. Furthermore, we show that even with random
initialization and without revealing (initial and) intermediate iterates, the
privacy loss in ZO-GD can grow superlinearly with the number of iterations when
minimizing convex objective functions.

### 8. [FACT: the Features At Convergence Theorem for neural networks](http://arxiv.org/pdf/2507.05644v1)

Authors: Enric Boix-Adsera, Neil Mallinar, James B. Simon, Mikhail Belkin

A central challenge in deep learning theory is to understand how neural
networks learn and represent features. To this end, we prove the Features at
Convergence Theorem (FACT), which gives a self-consistency equation that neural
network weights satisfy at convergence when trained with nonzero weight decay.
For each weight matrix $W$, this equation relates the "feature matrix" $W^\top
W$ to the set of input vectors passed into the matrix during forward
propagation and the loss gradients passed through it during backpropagation. We
validate this relation empirically, showing that neural features indeed satisfy
the FACT at convergence. Furthermore, by modifying the "Recursive Feature
Machines" of Radhakrishnan et al. 2024 so that they obey the FACT, we arrive at
a new learning algorithm, FACT-RFM. FACT-RFM achieves high performance on
tabular data and captures various feature learning behaviors that occur in
neural network training, including grokking in modular arithmetic and phase
transitions in learning sparse parities.

### 9. [Optimal structure learning and conditional independence testing](http://arxiv.org/pdf/2507.05689v1)

Authors: Ming Gao, Yuhao Wang, Bryon Aragam

We establish a fundamental connection between optimal structure learning and
optimal conditional independence testing by showing that the minimax optimal
rate for structure learning problems is determined by the minimax rate for
conditional independence testing in these problems. This is accomplished by
establishing a general reduction between these two problems in the case of
poly-forests, and demonstrated by deriving optimal rates for several examples,
including Bernoulli, Gaussian and nonparametric models. Furthermore, we show
that the optimal algorithm in these settings is a suitable modification of the
PC algorithm. This theoretical finding provides a unified framework for
analyzing the statistical complexity of structure learning through the lens of
minimax testing.

### 10. [Property Elicitation on Imprecise Probabilities](http://arxiv.org/pdf/2507.05857v1)

Authors: James Bailie, Rabanus Derr

Property elicitation studies which attributes of a probability distribution
can be determined by minimising a risk. We investigate a generalisation of
property elicitation to imprecise probabilities (IP). This investigation is
motivated by multi-distribution learning, which takes the classical machine
learning paradigm of minimising a single risk over a (precise) probability and
replaces it with $\Gamma$-maximin risk minimization over an IP. We provide
necessary conditions for elicitability of a IP-property. Furthermore, we
explain what an elicitable IP-property actually elicits through Bayes pairs --
the elicited IP-property is the corresponding standard property of the maximum
Bayes risk distribution.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-09 PST.

### 1. [Integrated cryogenic chip controls spin qubits](https://www.nature.com/articles/d41586-025-02122-8)

Authors: 

### 2. [Hyperbolic multi-channel hypergraph convolutional neural network based on multilayer hypergraph](https://www.nature.com/articles/s41598-025-08594-y)

Authors: Libing Bai et al.

### 3. [Attribution-based interpretable classification neural network with global and local perspectives](https://www.nature.com/articles/s41598-025-06218-z)

Authors: Zihao Shi et al.

### 4. [Diabetic retinopathy detection using adaptive deep convolutional neural networks on fundus images](https://www.nature.com/articles/s41598-025-09394-0)

Authors: Rashid Abbasi et al.

### 5. [A CAE model-based secure deduplication method](https://www.nature.com/articles/s41598-025-09788-0)

Authors: Chunbo Wang et al.

### 6. [Leveraging explainable artificial intelligence for early detection and mitigation of cyber threat in large-scale network environments](https://www.nature.com/articles/s41598-025-08597-9)

Authors: G. Nalinipriya et al.

### 7. [Telescope indexing for k-nearest neighbor search algorithms over high dimensional data & large data sets](https://www.nature.com/articles/s41598-025-09856-5)

Authors: Madhavan K R et al.

### 8. [Exploring single-head and multi-head CNN and LSTM-based models for road surface classification using on-board vehicle multi-IMU data](https://www.nature.com/articles/s41598-025-10573-2)

Authors: Luis A. Arce-Saenz et al.

### 9. [Improved salp swarm algorithm-driven deep CNN for brain tumor analysis](https://www.nature.com/articles/s41598-025-09326-y)

Authors: Umang Kumar Agrawal et al.

### 10. [Deep learning-based automatic detection and grading of disk herniation in lumbar magnetic resonance images](https://www.nature.com/articles/s41598-025-10401-7)

Authors: Yan Guo et al.

### 11. [MiTra: A Drone-Based Trajectory Data for an All-Traffic-State Inclusive Freeway with Ramps](https://www.nature.com/articles/s41597-025-05472-0)

Authors: Ankit Anil Chaudhari et al.

### 12. [Digital twin based deep learning framework for personalized thermal comfort prediction and energy efficient operation in smart buildings](https://www.nature.com/articles/s41598-025-10086-y)

Authors: Ahmad Almadhor et al.

### 13. [Deep ensemble learning with transformer models for enhanced Alzheimer’s disease detection](https://www.nature.com/articles/s41598-025-08362-y)

Authors: Shiza Latif et al.

### 14. [Reuse oriented information analysis methodology study on information analysis componentization](https://www.nature.com/articles/s41598-025-05109-7)

Authors: Hongyu Cai et al.

### 15. [Computational models reveal that intuitive physics underlies visual processing of soft objects](https://www.nature.com/articles/s41467-025-61458-x)

Authors: Wenyan Bi et al.

### 16. [Blended clustering energy efficient routing and PUF based authentication in IoT enabled smart agriculture systems](https://www.nature.com/articles/s41598-025-07917-3)

Authors: Senthil Kumar Chandrasekaran et al.

