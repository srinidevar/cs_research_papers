# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-01 17:06:03.557207 PST.

### Artificial Intelligence

### 1. [The Confidence Paradox: Can LLM Know When It's Wrong](http://arxiv.org/pdf/2506.23464v1)

Authors: Sahil Tripathi, Md Tabrez Nafis, Imran Hussain, Jiechao Gao

Document Visual Question Answering (DocVQA) systems are increasingly deployed
in real world applications, yet they remain ethically opaque-often producing
overconfident answers to ambiguous questions or failing to communicate
uncertainty in a trustworthy manner. This misalignment between model confidence
and actual knowledge poses significant risks, particularly in domains requiring
ethical accountability. Existing approaches such as LayoutLMv3, UDOP, and DONUT
have advanced SOTA performance by focusing on architectural sophistication and
accuracy; however, they fall short in ethical responsiveness.
  To address these limitations, we introduce HonestVQA, a self-supervised
honesty calibration framework for ethically aligned DocVQA. Our model-agnostic
method quantifies uncertainty to identify knowledge gaps, aligns model
confidence with actual correctness using weighted loss functions, and enforces
ethical response behavior via contrastive learning. We further introduce two
principled evaluation metrics--Honesty Score (H-Score) and Ethical Confidence
Index (ECI)--to benchmark alignment between confidence, accuracy, and ethical
communication. Empirically, HonestVQA improves DocVQA accuracy by up to 4.3%
and F1 by 4.3% across SpDocVQA, InfographicsVQA, and SROIE datasets. It reduces
overconfidence, lowering H-Score and ECI by 0.072 and 0.078, respectively. In
cross domain evaluation, it achieves up to 78.9% accuracy and 76.1% F1-score,
demonstrating strong generalization. Ablation shows a 3.8% drop in accuracy
without alignment or contrastive loss.

### 2. [Data Augmentation for Cognitive Behavioral Therapy: Leveraging ERNIE Language Models using Artificial Intelligence](http://arxiv.org/pdf/2506.23503v1)

Authors: Bosubabu Sambana, Kondreddygari Archana, Suram Indhra Sena Reddy, Shaik Meethaigar Jameer Basha, Shaik Karishma

Cognitive Behavioral Therapy (CBT) is a proven approach for addressing the
irrational thought patterns associated with mental health disorders, but its
effectiveness relies on accurately identifying cognitive pathways to provide
targeted treatment. In today's digital age, individuals often express negative
emotions on social media, where they may reveal cognitive distortions, and in
severe cases, exhibit suicidal tendencies. However, there is a significant gap
in methodologies designed to analyze these cognitive pathways, which could be
critical for psychotherapists aiming to deliver timely and effective
interventions in online environments. Cognitive Behavioral Therapy (CBT)
framework leveraging acceptance, commitment and data augmentation to categorize
and address both textual and visual content as positive or negative.
Specifically, the system employs BERT, RoBERTa for Sentiment Analysis and T5,
PEGASUS for Text Summarization, mT5 for Text Translation in Multiple Languages
focusing on detecting negative emotions and cognitive distortions within social
media data. While existing models are primarily designed to identify negative
thoughts, the proposed system goes beyond this by predicting additional
negative side effects and other potential mental health disorders likes
Phobias, Eating Disorders. This enhancement allows for a more comprehensive
understanding and intervention strategy, offering psychotherapists a powerful
tool for early detection and treatment of various psychological issues.

### 3. [Hybrid Approach for Electricity Price Forecasting using AlexNet and LSTM](http://arxiv.org/pdf/2506.23504v1)

Authors: Bosubabu Sambana, Kotamsetty Geethika Devi, Bandi Rajeswara Reddy, Galeti Mohammad Hussain, Gownivalla Siddartha

The recent development of advanced machine learning methods for hybrid models
has greatly addressed the need for the correct prediction of electrical prices.
This method combines AlexNet and LSTM algorithms, which are used to introduce a
new model with higher accuracy in price forecasting. Despite RNN and ANN being
effective, they often fail to deal with forex time sequence data. The
traditional methods do not accurately forecast the prices. These traditional
methods only focus on demand and price which leads to insufficient analysis of
data. To address this issue, using the hybrid approach, which focuses on
external variables that also effect the predicted prices. Nevertheless, due to
AlexNet's excellent feature extraction and LSTM's learning sequential patterns,
the prediction accuracy is vastly increased. The model is built on the past
data, which has been supplied with the most significant elements like demand,
temperature, sunlight, and rain. For example, the model applies methods, such
as minimum-maximum scaling and a time window, to predict the electricity prices
of the future. The results show that this hybrid model is good than the
standalone ones in terms of accuracy. Although we got our accuracy rating of
97.08, it shows higher accompaniments than remaining models RNN and ANN with
accuracies of 96.64 and 96.63 respectively.

### 4. [ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data](http://arxiv.org/pdf/2506.23520v1)

Authors: Yu Zhang, Ruijie Yu, Jidong Tian, Feng Zhu, Jiapeng Liu, Xiaokang Yang, Yaohui Jin, Yanyan Xu

With the increasing interest in robotic synthesis in the context of organic
chemistry, the automated extraction of chemical procedures from literature is
critical. However, this task remains challenging due to the inherent ambiguity
of chemical language and the high cost of human annotation required for
developing reliable computer-aided extraction protocols. Here, we present
ChemActor, a fully fine-tuned large language model (LLM), as a chemical
executor to convert between unstructured experimental procedures and structured
action sequences. We propose a sequential LLM-generated data framework to
address the challenges of insufficient and low-quality annotated data. This
framework integrates a data selection module that selects data based on
distribution divergence, with a general-purpose LLM, to generate
machine-executable actions from a single molecule input. Additionally, we
introduce a novel multi-round LLMs circle review metric, which reflects the
model's advanced understanding of chemical experimental procedures. Extensive
experiments on reaction-to-description (R2D) and description-to-action (D2A)
tasks demonstrate that ChemActor, augmented by LLM-generated data, achieves
state-of-the-art performance, outperforming the baseline model by 10%. The code
is available at: https://github.com/Zhanghahah/ChemActor.

### 5. [Evaluating Multi-Agent Defences Against Jailbreaking Attacks on Large Language Models](http://arxiv.org/pdf/2506.23576v1)

Authors: Maria Carolina Cornelia Wit, Jun Pang

Recent advances in large language models (LLMs) have raised concerns about
jailbreaking attacks, i.e., prompts that bypass safety mechanisms. This paper
investigates the use of multi-agent LLM systems as a defence against such
attacks. We evaluate three jailbreaking strategies, including the original
AutoDefense attack and two from Deepleaps: BetterDan and JB. Reproducing the
AutoDefense framework, we compare single-agent setups with two- and three-agent
configurations. Our results show that multi-agent systems enhance resistance to
jailbreaks, especially by reducing false negatives. However, its effectiveness
varies by attack type, and it introduces trade-offs such as increased false
positives and computational overhead. These findings point to the limitations
of current automated defences and suggest directions for improving alignment
robustness in future LLM systems.

### 6. [Self-correcting Reward Shaping via Language Models for Reinforcement Learning Agents in Games](http://arxiv.org/pdf/2506.23626v1)

Authors: António Afonso, Iolanda Leite, Alessandro Sestini, Florian Fuchs, Konrad Tollmar, Linus Gisslén

Reinforcement Learning (RL) in games has gained significant momentum in
recent years, enabling the creation of different agent behaviors that can
transform a player's gaming experience. However, deploying RL agents in
production environments presents two key challenges: (1) designing an effective
reward function typically requires an RL expert, and (2) when a game's content
or mechanics are modified, previously tuned reward weights may no longer be
optimal. Towards the latter challenge, we propose an automated approach for
iteratively fine-tuning an RL agent's reward function weights, based on a
user-defined language based behavioral goal. A Language Model (LM) proposes
updated weights at each iteration based on this target behavior and a summary
of performance statistics from prior training rounds. This closed-loop process
allows the LM to self-correct and refine its output over time, producing
increasingly aligned behavior without the need for manual reward engineering.
We evaluate our approach in a racing task and show that it consistently
improves agent performance across iterations. The LM-guided agents show a
significant increase in performance from $9\%$ to $74\%$ success rate in just
one iteration. We compare our LM-guided tuning against a human expert's manual
weight design in the racing task: by the final iteration, the LM-tuned agent
achieved an $80\%$ success rate, and completed laps in an average of $855$ time
steps, a competitive performance against the expert-tuned agent's peak $94\%$
success, and $850$ time steps.

### 7. [HASD: Hierarchical Adaption for pathology Slide-level Domain-shift](http://arxiv.org/pdf/2506.23673v1)

Authors: Jingsong Liu, Han Li, Chen Yang, Michael Deutges, Ario Sadafi, Xin You, Katharina Breininger, Nassir Navab, Peter J. Schüffler

Domain shift is a critical problem for pathology AI as pathology data is
heavily influenced by center-specific conditions. Current pathology domain
adaptation methods focus on image patches rather than WSI, thus failing to
capture global WSI features required in typical clinical scenarios. In this
work, we address the challenges of slide-level domain shift by proposing a
Hierarchical Adaptation framework for Slide-level Domain-shift (HASD). HASD
achieves multi-scale feature consistency and computationally efficient
slide-level domain adaptation through two key components: (1) a hierarchical
adaptation framework that integrates a Domain-level Alignment Solver for
feature alignment, a Slide-level Geometric Invariance Regularization to
preserve the morphological structure, and a Patch-level Attention Consistency
Regularization to maintain local critical diagnostic cues; and (2) a prototype
selection mechanism that reduces computational overhead. We validate our method
on two slide-level tasks across five datasets, achieving a 4.1\% AUROC
improvement in a Breast Cancer HER2 Grading cohort and a 3.9\% C-index gain in
a UCEC survival prediction cohort. Our method provides a practical and reliable
slide-level domain adaption solution for pathology institutions, minimizing
both computational and annotation costs.

### 8. [Agent4S: The Transformation of Research Paradigms from the Perspective of Large Language Models](http://arxiv.org/pdf/2506.23692v1)

Authors: Boyuan Zheng, Zerui Fang, Zhe Xu, Rui Wang, Yiwen Chen, Cunshi Wang, Mengwei Qu, Lei Lei, Zhen Feng, Yan Liu, Yuyang Li, Mingzhou Tan, Jiaji Wu, Jianwei Shuai, Jia Li, Fangfu Ye

While AI for Science (AI4S) serves as an analytical tool in the current
research paradigm, it doesn't solve its core inefficiency. We propose "Agent
for Science" (Agent4S)-the use of LLM-driven agents to automate the entire
research workflow-as the true Fifth Scientific Paradigm. This paper introduces
a five-level classification for Agent4S, outlining a clear roadmap from simple
task automation to fully autonomous, collaborative "AI Scientists." This
framework defines the next revolutionary step in scientific discovery.

### 9. [A New Perspective On AI Safety Through Control Theory Methodologies](http://arxiv.org/pdf/2506.23703v1)

Authors: Lars Ullrich, Walter Zimmer, Ross Greer, Knut Graichen, Alois C. Knoll, Mohan Trivedi

While artificial intelligence (AI) is advancing rapidly and mastering
increasingly complex problems with astonishing performance, the safety
assurance of such systems is a major concern. Particularly in the context of
safety-critical, real-world cyber-physical systems, AI promises to achieve a
new level of autonomy but is hampered by a lack of safety assurance. While
data-driven control takes up recent developments in AI to improve control
systems, control theory in general could be leveraged to improve AI safety.
Therefore, this article outlines a new perspective on AI safety based on an
interdisciplinary interpretation of the underlying data-generation process and
the respective abstraction by AI systems in a system theory-inspired and system
analysis-driven manner. In this context, the new perspective, also referred to
as data control, aims to stimulate AI engineering to take advantage of existing
safety analysis and assurance in an interdisciplinary way to drive the paradigm
of data control. Following a top-down approach, a generic foundation for safety
analysis and assurance is outlined at an abstract level that can be refined for
specific AI systems and applications and is prepared for future innovation.

### 10. [A Survey on Autonomy-Induced Security Risks in Large Model-Based Agents](http://arxiv.org/pdf/2506.23844v1)

Authors: Hang Su, Jun Luo, Chang Liu, Xiao Yang, Yichi Zhang, Yinpeng Dong, Jun Zhu

Recent advances in large language models (LLMs) have catalyzed the rise of
autonomous AI agents capable of perceiving, reasoning, and acting in dynamic,
open-ended environments. These large-model agents mark a paradigm shift from
static inference systems to interactive, memory-augmented entities. While these
capabilities significantly expand the functional scope of AI, they also
introduce qualitatively novel security risks - such as memory poisoning, tool
misuse, reward hacking, and emergent misalignment - that extend beyond the
threat models of conventional systems or standalone LLMs. In this survey, we
first examine the structural foundations and key capabilities that underpin
increasing levels of agent autonomy, including long-term memory retention,
modular tool use, recursive planning, and reflective reasoning. We then analyze
the corresponding security vulnerabilities across the agent stack, identifying
failure modes such as deferred decision hazards, irreversible tool chains, and
deceptive behaviors arising from internal state drift or value misalignment.
These risks are traced to architectural fragilities that emerge across
perception, cognition, memory, and action modules. To address these challenges,
we systematically review recent defense strategies deployed at different
autonomy layers, including input sanitization, memory lifecycle control,
constrained decision-making, structured tool invocation, and introspective
reflection. We introduce the Reflective Risk-Aware Agent Architecture (R2A2), a
unified cognitive framework grounded in Constrained Markov Decision Processes
(CMDPs), which incorporates risk-aware world modeling, meta-policy adaptation,
and joint reward-risk optimization to enable principled, proactive safety
across the agent's decision-making loop.

### Hardware Architecture

### 1. [Sustainable operation of research infrastructure for novel computing](http://arxiv.org/pdf/2506.23901v1)

Authors: Yannik Stradmann, Joscha Ilmberger, Eric Müller, Johannes Schemmel

Novel compute systems are an emerging research topic, aiming towards building
next-generation compute platforms. For these systems to thrive, they need to be
provided as research infrastructure to allow acceptance and usage by a large
community. By the example of the neuromorphic BrainScaleS-2 system, we showcase
the transformation from a laboratory setup to a sustainable, publicly available
platform. It is embedded into a purpose-built institute, tightly coupling a
conventional cluster with novel compute hardware. The network infrastructure is
optimized for robust operation, even in the case of unintended behavior of
individual devices. The systems themselves are packaged into 19-inch compatible
units to allow for easy maintenance and extension. We operate the platform
using modern CI/CD techniques and continuously assert its health using
automated system monitoring. Finally, we share our lessons learned during the
decade-long endeavor of operating analog neuromorphic systems as a publicly
available research platform.

### 2. [Data-Driven Power Modeling and Monitoring via Hardware Performance Counter Tracking](http://arxiv.org/pdf/2506.23672v1)

Authors: Sergio Mazzola, Gabriele Ara, Thomas Benz, Björn Forsberg, Tommaso Cucinotta, Luca Benini

Energy-centric design is paramount in the current embedded computing era: use
cases require increasingly high performance at an affordable power budget,
often under real-time constraints. Hardware heterogeneity and parallelism help
address the efficiency challenge, but greatly complicate online power
consumption assessments, which are essential for dynamic hardware and software
stack adaptations. We introduce a novel power modeling methodology with
state-of-the-art accuracy, low overhead, and high responsiveness, whose
implementation does not rely on microarchitectural details. Our methodology
identifies the Performance Monitoring Counters (PMCs) with the highest linear
correlation to the power consumption of each hardware sub-system, for each
Dynamic Voltage and Frequency Scaling (DVFS) state. The individual, simple
models are composed into a complete model that effectively describes the power
consumption of the whole system, achieving high accuracy and low overhead. Our
evaluation reports an average estimation error of 7.5% for power consumption
and 1.3% for energy. We integrate these models in the Linux kernel with
Runmeter, an open-source, PMC-based monitoring framework. Runmeter manages PMC
sampling and processing, enabling the execution of our power models at runtime.
With a worst-case time overhead of only 0.7%, Runmeter provides responsive and
accurate power measurements directly in the kernel. This information can be
employed for actuation policies in workload-aware DVFS and power-aware,
closed-loop task scheduling.

### 3. [Not quite a piece of CHERI-cake: Are new digital security by design architectures usable?](http://arxiv.org/pdf/2506.23682v1)

Authors: Maysara Alhindi, Joseph Hallett

A digital security-by-design computer architecture, like CHERI, lets you
program without fear of buffer overflows or other memory safety errors, but
CHERI also rewrites some of the assumptions about how C works and how
fundamental types (such as pointers) are implemented in hardware. We conducted
a usability study to examine how developers react to the changes required by
CHERI when porting software to run on it. We find that developers struggle with
CHERI's display of warnings and errors and a lack of diverse documentation.

### Computational Complexity

### 1. [Fantastic Flips and Where to Find Them: A General Framework for Parameterized Local Search on Partitioning Problem](http://arxiv.org/pdf/2506.24001v1)

Authors: Niels Grüttemeier, Nils Morawietz, Frank Sommer

Parameterized local search combines classic local search heuristics with the
paradigm of parameterized algorithmics. While most local search algorithms aim
to improve given solutions by performing one single operation on a given
solution, the parameterized approach aims to improve a solution by performing
$k$ simultaneous operations. Herein, $k$ is a parameter called search radius
for which the value can be chosen by a user. One major goal in the field of
parameterized local search is to outline the trade-off between the size of $k$
and the running time of the local search step. In this work, we introduce an
abstract framework that generalizes natural parameterized local search
approaches for a large class of partitioning problems: Given $n$ items that are
partitioned into $b$ bins and a target function that evaluates the quality of
the current partition, one asks whether it is possible to improve the solution
by removing up to $k$ items from their current bins and reassigning them to
other bins. Among others, our framework applies for the local search versions
of problems like Cluster Editing, Vector Bin Packing, and Nash Social Welfare.
Motivated by a real-world application of the problem Vector Bin Packing, we
introduce a parameter called number of types $\tau \le n$ and show that all
problems fitting in our framework can be solved in $\tau^k 2^{O(k)} |I|^{O(1)}$
time, where $|I|$ denotes the total input size. In case of Cluster Editing, the
parameter $\tau$ generalizes the well-known parameter neighborhood diversity of
the input graph. We complement this by showing that for all considered
problems, an algorithm significantly improving over our algorithm with running
time $\tau^k 2^{O(k)} |I|^{O(1)}$ would contradict the ETH. Additionally, we
show that even on very restricted instances, all considered problems are
W[1]-hard when parameterized by the search radius $k$ alone.

### 2. [Dominating Set Knapsack: Profit Optimization on Dominating Sets](http://arxiv.org/pdf/2506.24032v1)

Authors: Sipra Singh

In a large-scale network, we want to choose some influential nodes to make a
profit by paying some cost within a limited budget so that we do not have to
spend more budget on some nodes adjacent to the chosen nodes; our problem is
the graph-theoretic representation of it. We define our problem Dominating Set
Knapsack by attaching Knapsack Problem with Dominating Set on graphs. Each
vertex is associated with a cost factor and a profit amount. We aim to choose
some vertices within a fixed budget that gives maximum profit so that we do not
need to choose their 1-hop neighbors. We show that the Dominating Set Knapsack
problem is strongly NP-complete even when restricted to Bipartite graphs but
weakly NP-complete for Star graphs. We present a pseudo-polynomial time
algorithm for Trees in time $O(n\cdot min\{s^2, (\alpha(V))^2\})$. We show that
Dominating Set Knapsack is very unlikely to be Fixed Parameter Tractable(FPT)
by proving that it is in W[2]-hard parameterized by the solution size. We
developed FPT algorithms with running time $O(4^{tw}\cdot n^{O(1)} \cdot
min\{s^2,{\alpha(V)}^2\})$ and $O(2^{vck-1}\cdot n^{O(1)} \cdot
min\{s^2,{\alpha(V)}^2\})$, where $tw$ represents the treewidth of the given
graph, $vck$ is the solution size of the Vertex Cover Knapsack, $s$ is the size
of the knapsack and $\alpha(V)=\sum_{v\in V}\alpha(v)$.

### 3. [A Graph Width Perspective on Partially Ordered Hamiltonian Paths and Cycles I: Treewidth, Pathwidth, and Grid Graphs](http://arxiv.org/pdf/2506.23790v1)

Authors: Jesse Beisegel, Katharina Klost, Kristin Knorr, Fabienne Ratajczak, Robert Scheffler

We consider the problem of finding a Hamiltonian path or a Hamiltonian cycle
with precedence constraints in the form of a partial order on the vertex set.
We show that the path problem is $\mathsf{NP}$-complete for graphs of pathwidth
4 while the cycle problem is $\mathsf{NP}$-complete on graphs of pathwidth 5.
We complement these results by giving polynomial-time algorithms for graphs of
pathwidth 3 and treewidth 2 for Hamiltonian paths as well as pathwidth 4 and
treewidth 3 for Hamiltonian cycles. Furthermore, we study the complexity of the
path and cycle problems on rectangular grid graphs of bounded height. For
these, we show that the path and cycle problems are $\mathsf{NP}$-complete when
the height of the grid is greater or equal to 7 and 9, respectively. In the
variant where we look for minimum edge-weighted Hamiltonian paths and cycles,
the problems are $\mathsf{NP}$-hard for heights 5 and 6, respectively.

### 4. [Segmented Operations using Matrix Multiplications](http://arxiv.org/pdf/2506.23906v1)

Authors: Aleksandros Sobczyk, Giuseppe Sorrentino, Anastasios Zouzias

Specialized computational units that perform small matrix multiplications as
primitive operations are typically present in modern accelerators. However,
these units are often underutilized for many fundamental operations besides
dense matrix multiplications. The analysis of algorithms for such architectures
is currently stagnated due to the lack of a rigorous theoretical model of
computation that captures their characteristics. In this work, we propose
MMV-RAM, a computational model tailored to matrix multiplication accelerators.
MMV-RAM judiciously extends the Vector-RAM model with an additional processing
unit that multiplies two matrices of sizes $n\times s$ and $s\times s$ in a
single parallel step, where $s$ is a model parameter. We provide a detailed
theoretical analysis of the model, and carefully balance the computational
power between the matrix and vector units, guided by the circuit complexity
lower bound that parity is not in AC[0].
  In MMV-RAM, we study algorithms for segmented scan and sum, two fundamental
parallel primitives. We propose a segmented scan algorithm that uses matrix
multiplications to perform speculative block-scan computations, which runs in
$O(\log_s(n))$ steps. In contrast, we show that any algorithm that uses only
the vector unit of MMV-RAM requires
$\Omega\left(\frac{\log_2(n)}{\log_2\log_2(n)}\right)$ steps. We further apply
these techniques to obtain similar theoretical speedups for element-wise vector
multiplication and matrix multiplication. Beyond the worst-case complexity
analysis, we propose algorithms for segmented operations that could lead to
highly efficient and pragmatic implementations. For example, we observe that
segmented sum is a combination of three elementary parallel primitives: scan,
compress, and vector differentiation. As a case study, we implement...

### 5. [Factorization norms and an inverse theorem for MaxCut](http://arxiv.org/pdf/2506.23989v1)

Authors: Igor Balla, Lianna Hambardzumyan, István Tomon

We prove that Boolean matrices with bounded $\gamma_2$-norm or bounded
normalized trace norm must contain a linear-sized all-ones or all-zeros
submatrix, verifying a conjecture of Hambardzumyan, Hatami, and Hatami. We also
present further structural results about Boolean matrices of bounded
$\gamma_2$-norm and discuss applications in communication complexity, operator
theory, spectral graph theory, and extremal combinatorics.
  As a key application, we establish an inverse theorem for MaxCut. A
celebrated result of Edwards states that every graph $G$ with $m$ edges has a
cut of size at least $\frac{m}{2}+\frac{\sqrt{8m+1}-1}{8}$, with equality
achieved by complete graphs with an odd number of vertices. To contrast this,
we prove that if the MaxCut of $G$ is at most $\frac{m}{2}+O(\sqrt{m})$, then
$G$ must contain a clique of size $\Omega(\sqrt{m})$.

### Computational Engineering

### 1. [Immersive Technologies in Training and Healthcare: From Space Missions to Psychophysiological Research](http://arxiv.org/pdf/2506.23545v1)

Authors: Barbara Karpowicz, Maciej Grzeszczuk, Adam Kuzdraliński, Monika Kornacka, Aliaksandr Marozau, Wiktor Stawski, Pavlo Zinevych, Grzegorz Marcin Wójcik, Tomasz Kowalewski, Grzegorz Pochwatko, Wiesław Kopeć

Virtual, Augmented, and eXtended Reality (VR/AR/XR) technologies are
increasingly recognized for their applications in training, diagnostics, and
psychological research, particularly in high-risk and highly regulated
environments. In this panel we discuss how immersive systems enhance human
performance across multiple domains, including clinical psychology, space
exploration, and medical education. In psychological research and training, XR
can offer a controlled yet ecologically valid setting for measuring cognitive
and affective processes. In space exploration, we discuss the development of
VR-based astronaut training and diagnostic systems, allowing astronauts to
perform real-time health assessments. In medical education and rehabilitation,
we cover procedural training and patient engagement. From virtual surgical
simulations to gamified rehabilitation exercises, immersive environments
enhance both learning outcomes and treatment adherence.

### 2. [Validation of AI-Based 3D Human Pose Estimation in a Cyber-Physical Environment](http://arxiv.org/pdf/2506.23739v1)

Authors: Lisa Marie Otto, Michael Kaiser, Daniel Seebacher, Steffen Müller

Ensuring safe and realistic interactions between automated driving systems
and vulnerable road users (VRUs) in urban environments requires advanced
testing methodologies. This paper presents a test environment that combines a
Vehiclein-the-Loop (ViL) test bench with a motion laboratory, demonstrating the
feasibility of cyber-physical (CP) testing of vehicle-pedestrian and
vehicle-cyclist interactions. Building upon previous work focused on pedestrian
localization, we further validate a human pose estimation (HPE) approach
through a comparative analysis of real-world (RW) and virtual representations
of VRUs. The study examines the perception of full-body motion using a
commercial monocular camera-based 3Dskeletal detection AI. The virtual scene is
generated in Unreal Engine 5, where VRUs are animated in real time and
projected onto a screen to stimulate the camera. The proposed stimulation
technique ensures the correct perspective, enabling realistic vehicle
perception. To assess the accuracy and consistency of HPE across RW and CP
domains, we analyze the reliability of detections as well as variations in
movement trajectories and joint estimation stability. The validation includes
dynamic test scenarios where human avatars, both walking and cycling, are
monitored under controlled conditions. Our results show a strong alignment in
HPE between RW and CP test conditions for stable motion patterns, while notable
inaccuracies persist under dynamic movements and occlusions, particularly for
complex cyclist postures. These findings contribute to refining CP testing
approaches for evaluating next-generation AI-based vehicle perception and to
enhancing interaction models of automated vehicles and VRUs in CP environments.

### Computational Geometry

### 1. [Passage-traversing optimal path planning with sampling-based algorithms](http://arxiv.org/pdf/2506.23614v1)

Authors: Jing Huang, Hao Su, Kwok Wai Samuel Au

This paper introduces a new paradigm of optimal path planning, i.e.,
passage-traversing optimal path planning (PTOPP), that optimizes paths'
traversed passages for specified optimization objectives. In particular, PTOPP
is utilized to find the path with optimal accessible free space along its
entire length, which represents a basic requirement for paths in robotics. As
passages are places where free space shrinks and becomes constrained, the core
idea is to leverage the path's passage traversal status to characterize its
accessible free space comprehensively. To this end, a novel passage detection
and free space decomposition method using proximity graphs is proposed,
enabling fast detection of sparse but informative passages and environment
decompositions. Based on this preprocessing, optimal path planning with
accessible free space objectives or constraints is formulated as PTOPP problems
compatible with sampling-based optimal planners. Then, sampling-based
algorithms for PTOPP, including their dependent primitive procedures, are
developed leveraging partitioned environments for fast passage traversal check.
All these methods are implemented and thoroughly tested for effectiveness and
efficiency validation. Compared to existing approaches, such as clearance-based
methods, PTOPP demonstrates significant advantages in configurability, solution
optimality, and efficiency, addressing prior limitations and incapabilities. It
is believed to provide an efficient and versatile solution to accessible free
space optimization over conventional avenues and more generally, to a broad
class of path planning problems that can be formulated as PTOPP.

### 2. [$C_4$-free subgraphs of high degree with geometric applications](http://arxiv.org/pdf/2506.23942v1)

Authors: Zach Hunter, Aleksa Milojević, Istvan Tomon, Benny Sudakov

The Zarankiewicz problem, a cornerstone problem in extremal graph theory,
asks for the maximum number of edges in an $n$-vertex graph that does not
contain the complete bipartite graph $K_{s,s}$. While the problem remains
widely open in the case of general graphs, the past two decades have seen
significant progress on this problem for various restricted graph classes --
particularly those arising from geometric settings -- leading to a deeper
understanding of their structure.
  In this paper, we develop a new structural tool for addressing
Zarankiewicz-type problems. More specifically, we show that for any positive
integer $k$, every graph with average degree $d$ either contains an induced
$C_4$-free subgraph with average degree at least $k$, or it contains a
$d$-vertex subgraph with $\Omega_k(d^2)$ edges. As an application of this
dichotomy, we propose a unified approach to a large number of Zarankiewicz-type
problems in geometry, obtaining optimal bounds in each case.

### 3. [Linear Layouts of Graphs with Priority Queues](http://arxiv.org/pdf/2506.23943v1)

Authors: Emilio Di Giacomo, Walter Didimo, Henry Förster, Torsten Ueckerdt, Johannes Zink

A linear layout of a graph consists of a linear ordering of its vertices and
a partition of its edges into pages such that the edges assigned to the same
page obey some constraint. The two most prominent and widely studied types of
linear layouts are stack and queue layouts, in which any two edges assigned to
the same page are forbidden to cross and nest, respectively. The names of these
two layouts derive from the fact that, when parsing the graph according to the
linear vertex ordering, the edges in a single page can be stored using a single
stack or queue, respectively. Recently, the concepts of stack and queue layouts
have been extended by using a double-ended queue or a restricted-input queue
for storing the edges of a page. We extend this line of study to edge-weighted
graphs by introducing priority queue layouts, that is, the edges on each page
are stored in a priority queue whose keys are the edge weights. First, we show
that there are edge-weighted graphs that require a linear number of priority
queues. Second, we characterize the graphs that admit a priority queue layout
with a single queue, regardless of the edge-weight function, and we provide an
efficient recognition algorithm. Third, we show that the number of priority
queues required independently of the edge-weight function is bounded by the
pathwidth of the graph, but can be arbitrarily large already for graphs of
treewidth two. Finally, we prove that determining the minimum number of
priority queues is NP-complete if the linear ordering of the vertices is fixed.

### Computation and Language

### 1. [What to Keep and What to Drop: Adaptive Table Filtering Framework](http://arxiv.org/pdf/2506.23463v1)

Authors: Jang Won June

Large language models (LLMs) for table-based reasoning often struggle with
large tables due to input length limits. We propose ATF (Adaptive Table
Filtering Framework), a modular and question-aware filtering pipeline that
prunes uninformative columns and rows using LLM-generated column descriptions,
clustering, and sparse-dense alignment scores. ATF integrates seamlessly with
existing models (e.g., TAPAS, TAPEX) without retraining. Experiments show that
ATF reduces table cells by ~70\%, boosting performance on out-of-domain TableQA
tasks while causing slight performance drops on Table Fact Verification, where
full-table context is more critical. These results highlight ATF's ability to
adaptively balance informativeness and minimalism across tasks.

### 2. [On Recipe Memorization and Creativity in Large Language Models: Is Your Model a Creative Cook, a Bad Cook, or Merely a Plagiator?](http://arxiv.org/pdf/2506.23527v1)

Authors: Jan Kvapil, Martin Fajcik

This work-in-progress investigates the memorization, creativity, and nonsense
found in cooking recipes generated from Large Language Models (LLMs).
Precisely, we aim (i) to analyze memorization, creativity, and non-sense in
LLMs using a small, high-quality set of human judgments and (ii) to evaluate
potential approaches to automate such a human annotation in order to scale our
study to hundreds of recipes. To achieve (i), we conduct a detailed human
annotation on 20 preselected recipes generated by LLM (Mixtral), extracting
each recipe's ingredients and step-by-step actions to assess which elements are
memorized--i.e., directly traceable to online sources possibly seen during
training--and which arise from genuine creative synthesis or outright nonsense.
We find that Mixtral consistently reuses ingredients that can be found in
online documents, potentially seen during model training, suggesting strong
reliance on memorized content. To achieve aim (ii) and scale our analysis
beyond small sample sizes and single LLM validation, we design an
``LLM-as-judge'' pipeline that automates recipe generation, nonsense detection,
parsing ingredients and recipe steps, and their annotation. For instance,
comparing its output against human annotations, the best ingredient extractor
and annotator is Llama 3.1+Gemma 2 9B, achieving up to 78% accuracy on
ingredient matching. This automated framework enables large-scale
quantification of memorization, creativity, and nonsense in generated recipes,
providing rigorous evidence of the models' creative capacities.

### 3. [Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack](http://arxiv.org/pdf/2506.23661v1)

Authors: Arnisa Fazla, Lucas Krauter, David Guzman Piedrahita, Andrianos Michail

We extend BeamAttack, an adversarial attack algorithm designed to evaluate
the robustness of text classification systems through word-level modifications
guided by beam search. Our extensions include support for word deletions and
the option to skip substitutions, enabling the discovery of minimal
modifications that alter model predictions. We also integrate LIME to better
prioritize word replacements. Evaluated across multiple datasets and victim
models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA
framework, our approach achieves over a 99\% attack success rate while
preserving the semantic and lexical similarity of the original texts. Through
both quantitative and qualitative analysis, we highlight BeamAttack's
effectiveness and its limitations. Our implementation is available at
https://github.com/LucK1Y/BeamAttack

### 4. [L0: Reinforcement Learning to Become General Agents](http://arxiv.org/pdf/2506.23667v1)

Authors: Junjie Zhang, Jingyi Xi, Zhuoyang Song, Junyu Lu, Yuhua Ke, Ting Sun, Yukun Yang, Jiaxing Zhang, Songxin Zhang, Zejian Xie

Training large language models (LLMs) to act as autonomous agents for
multi-turn, long-horizon tasks remains significant challenges in scalability
and training efficiency. To address this, we introduce L-Zero (L0), a scalable,
end-to-end training pipeline for general-purpose agents. Featuring a low-cost,
extensible, and sandboxed concurrent agent worker pool, L0 lowers the barrier
for applying reinforcement learning in complex environments. We also introduce
NB-Agent, the agent scaffold within L0, which operates in a "code-as-action"
fashion via a Read-Eval-Print-Loop (REPL). We evaluate L0 on factuality
question-answering benchmarks. Our experiments demonstrate that a base model
can develop robust problem-solving skills using solely Reinforcement Learning
with Verifiable Rewards (RLVR). On the Qwen2.5-7B-Instruct model, our method
boosts accuracy on SimpleQA from 30 % to 80 % and on HotpotQA from 22 % to 41
%. We have open-sourced the entire L0 system, including our L0 series models,
the NB-Agent, a complete training pipeline, and the corresponding training
recipes on (https://github.com/cmriat/l0).

### 5. [Positional Bias in Binary Question Answering: How Uncertainty Shapes Model Preferences](http://arxiv.org/pdf/2506.23743v1)

Authors: Tiziano Labruna, Simone Gallo, Giovanni Da San Martino

Positional bias in binary question answering occurs when a model
systematically favors one choice over another based solely on the ordering of
presented options. In this study, we quantify and analyze positional bias
across five large language models under varying degrees of answer uncertainty.
We re-adapted the SQuAD-it dataset by adding an extra incorrect answer option
and then created multiple versions with progressively less context and more
out-of-context answers, yielding datasets that range from low to high
uncertainty. Additionally, we evaluate two naturally higher-uncertainty
benchmarks: (1) WebGPT - question pairs with unequal human-assigned quality
scores, and (2) Winning Arguments - where models predict the more persuasive
argument in Reddit's r/ChangeMyView exchanges. Across each dataset, the order
of the "correct" (or higher-quality/persuasive) option is systematically
flipped (first placed in position 1, then in position 2) to compute both
Preference Fairness and Position Consistency. We observe that positional bias
is nearly absent under low-uncertainty conditions, but grows exponentially when
it becomes doubtful to decide which option is correct.

### 6. [Garbage In, Reasoning Out? Why Benchmark Scores are Unreliable and What to Do About It](http://arxiv.org/pdf/2506.23864v1)

Authors: Seyed Mahed Mousavi, Edoardo Cecchinato, Lucia Hornikova, Giuseppe Riccardi

We conduct a systematic audit of three widely used reasoning benchmarks,
SocialIQa, FauxPas-EAI, and ToMi, and uncover pervasive flaws in both benchmark
items and evaluation methodology. Using five LLMs (GPT-{3, 3.5, 4, o1}, and
LLaMA 3.1) as diagnostic tools, we identify structural, semantic, and pragmatic
issues in benchmark design (e.g., duplicated items, ambiguous wording, and
implausible answers), as well as scoring procedures that prioritize output form
over reasoning process. Through systematic human annotation and re-evaluation
on cleaned benchmark subsets, we find that model scores often improve not due
to due to erratic surface wording variations and not to improved reasoning.
Infact, further analyses show that model performance is highly sensitive to
minor input variations such as context availability and phrasing, revealing
that high scores may reflect alignment with format-specific cues rather than
consistent inference based on the input. These findings challenge the validity
of current benchmark-based claims about reasoning in LLMs, and highlight the
need for evaluation protocols that assess reasoning as a process of drawing
inference from available information, rather than as static output selection.
We release audited data and evaluation tools to support more interpretable and
diagnostic assessments of model reasoning.

### 7. [Advancing Multi-Step Mathematical Reasoning in Large Language Models through Multi-Layered Self-Reflection with Auto-Prompting](http://arxiv.org/pdf/2506.23888v1)

Authors: André de Souza Loureiro, Jorge Valverde-Rebaza, Julieta Noguez, David Escarcega, Ricardo Marcacini

Recent advancements in Large Language Models (LLMs) have significantly
improved their problem-solving capabilities. However, these models still
struggle when faced with complex multi-step reasoning tasks. In this paper, we
propose the Multi-Layered Self-Reflection with Auto-Prompting (MAPS) framework,
a novel approach designed to enhance multi-step mathematical reasoning in LLMs
by integrating techniques such as Chain of Thought (CoT), Self-Reflection, and
Auto-Prompting. Unlike traditional static prompting methods, MAPS employs an
iterative refinement process. Initially, the model generates a solution using
CoT prompting. When errors are detected, an adaptive self-reflection mechanism
identifies and analyzes them, generating tailored prompts to guide corrections.
These dynamically adjusted prompts enable the model to iteratively refine its
reasoning. Experiments on four well-established benchmarks across multiple LLMs
show that MAPS significantly outperforms standard CoT and achieves competitive
results with reasoning-optimized models. In addition, MAPS enables
general-purpose LLMs to reach performance levels comparable to specialized
reasoning models. While deeper reflection layers improve accuracy, they also
increase token usage and costs. To balance this trade-off, MAPS strategically
limits reflection depth, ensuring an optimal balance between cost and reasoning
performance.

### 8. [IMPACT: Inflectional Morphology Probes Across Complex Typologies](http://arxiv.org/pdf/2506.23929v1)

Authors: Mohammed J. Saeed, Tommi Vehvilainen, Evgeny Fedoseev, Sevil Caliskan, Tatiana Vodolazova

Large Language Models (LLMs) have shown significant progress on various
multilingual benchmarks and are increasingly used to generate and evaluate text
in non-English languages. However, while they may produce fluent outputs, it
remains unclear to what extent these models truly grasp the underlying
linguistic complexity of those languages, particularly in morphology. To
investigate this, we introduce IMPACT, a synthetically generated evaluation
framework focused on inflectional morphology, which we publicly release,
designed to evaluate LLM performance across five morphologically rich
languages: Arabic, Russian, Finnish, Turkish, and Hebrew. IMPACT includes
unit-test-style cases covering both shared and language-specific phenomena,
from basic verb inflections (e.g., tense, number, gender) to unique features
like Arabic's reverse gender agreement and vowel harmony in Finnish and
Turkish. We assess eight multilingual LLMs that, despite strong English
performance, struggle with other languages and uncommon morphological patterns,
especially when judging ungrammatical examples. We also show that Chain of
Thought and Thinking Models can degrade performance. Our work exposes gaps in
LLMs' handling of linguistic complexity, pointing to clear room for
improvement. To support further research, we publicly release the IMPACT
framework.

### 9. [Graft: Integrating the Domain Knowledge via Efficient Parameter Synergy for MLLMs](http://arxiv.org/pdf/2506.23940v1)

Authors: Yang Dai, Jianxiang An, Tianwei Lin, Hongyang He, Hongzhe Huang, Wenqiao Zhang, Zheqi Lv, Siliang Tang, Yueting Zhuang

Multimodal Large Language Models (MLLMs) have achieved success across various
domains. However, their applicability tends to degrade when confronted with
different types of data inputs, especially for MLLMs that have been fine-tuned
for specific tasks. Despite its importance, the study of knowledge sharing
among domain-specific MLLMs--such as those trained for mathematics or
code--remains largely underexplored. To address the fragmentation of knowledge
across domain-specialized MLLMs, we propose a unified parameter integration
framework that enables modular composition of expert capabilities. Our method
is grounded in a novel Compatibility-Aware Parameter Splicing (CAPS) strategy,
which leverages both local functional attribution and global
information-theoretic signals to guide selective parameter fusion. By extending
this mechanism to the low-rank adaptation layer granularity, we ensure
efficient integration with minimal inference overhead. Furthermore, we
introduce a domain compatibility scoring mechanism that quantifies inter-expert
alignment at the activation level and correlates with downstream task utility.
This principled fusion protocol allows the final model to synergize
heterogeneous expertise while preserving structural modularity. Extensive
evaluations across diverse multimodal benchmarks validate the effectiveness of
our framework, offering a scalable path toward compositional, domain-adaptive
MLLMs.

### 10. [Unveiling Decision-Making in LLMs for Text Classification : Extraction of influential and interpretable concepts with Sparse Autoencoders](http://arxiv.org/pdf/2506.23951v1)

Authors: Mathis Le Bail, Jérémie Dentan, Davide Buscaldi, Sonia Vanier

Sparse Autoencoders (SAEs) have been successfully used to probe Large
Language Models (LLMs) and extract interpretable concepts from their internal
representations. These concepts are linear combinations of neuron activations
that correspond to human-interpretable features. In this paper, we investigate
the effectiveness of SAE-based explainability approaches for sentence
classification, a domain where such methods have not been extensively explored.
We present a novel SAE-based architecture tailored for text classification,
leveraging a specialized classifier head and incorporating an activation rate
sparsity loss. We benchmark this architecture against established methods such
as ConceptShap, Independent Component Analysis, and other SAE-based concept
extraction techniques. Our evaluation covers two classification benchmarks and
four fine-tuned LLMs from the Pythia family. We further enrich our analysis
with two novel metrics for measuring the precision of concept-based
explanations, using an external sentence encoder. Our empirical results show
that our architecture improves both the causality and interpretability of the
extracted features.

### Cryptography and Security

### 1. [A Large-Scale Evolvable Dataset for Model Context Protocol Ecosystem and Security Analysis](http://arxiv.org/pdf/2506.23474v1)

Authors: Zhiwei Lin, Bonan Ruan, Jiahao Liu, Weibo Zhao

The Model Context Protocol (MCP) has recently emerged as a standardized
interface for connecting language models with external tools and data. As the
ecosystem rapidly expands, the lack of a structured, comprehensive view of
existing MCP artifacts presents challenges for research. To bridge this gap, we
introduce MCPCorpus, a large-scale dataset containing around 14K MCP servers
and 300 MCP clients. Each artifact is annotated with 20+ normalized attributes
capturing its identity, interface configuration, GitHub activity, and metadata.
MCPCorpus provides a reproducible snapshot of the real-world MCP ecosystem,
enabling studies of adoption trends, ecosystem health, and implementation
diversity. To keep pace with the rapid evolution of the MCP ecosystem, we
provide utility tools for automated data synchronization, normalization, and
inspection. Furthermore, to support efficient exploration and exploitation, we
release a lightweight web-based search interface. MCPCorpus is publicly
available at: https://github.com/Snakinya/MCPCorpus.

### 2. [Cybersecurity AI: The Dangerous Gap Between Automation and Autonomy](http://arxiv.org/pdf/2506.23592v1)

Authors: Víctor Mayoral-Vilches

The cybersecurity industry combines "automated" and "autonomous" AI, creating
dangerous misconceptions about system capabilities. Recent milestones like XBOW
topping HackerOne's leaderboard showcase impressive progress, yet these systems
remain fundamentally semi-autonomous--requiring human oversight. Drawing from
robotics principles, where the distinction between automation and autonomy is
well-established, I take inspiration from prior work and establish a 6-level
taxonomy (Level 0-5) distinguishing automation from autonomy in Cybersecurity
AI. Current "autonomous" pentesters operate at Level 3-4: they execute complex
attack sequences but need human review for edge cases and strategic decisions.
True Level 5 autonomy remains aspirational. Organizations deploying
mischaracterized "autonomous" tools risk reducing oversight precisely when it's
most needed, potentially creating new vulnerabilities. The path forward
requires precise terminology, transparent capabilities disclosure, and human-AI
partnership-not replacement.

### 3. [Privacy-Preserving Federated Learning Scheme with Mitigating Model Poisoning Attacks: Vulnerabilities and Countermeasures](http://arxiv.org/pdf/2506.23622v1)

Authors: Jiahui Wu, Fucai Luo, Tiecheng Sun, Haiyan Wang, Weizhe Zhang

The privacy-preserving federated learning schemes based on the setting of two
honest-but-curious and non-colluding servers offer promising solutions in terms
of security and efficiency. However, our investigation reveals that these
schemes still suffer from privacy leakage when considering model poisoning
attacks from malicious users. Specifically, we demonstrate that the
privacy-preserving computation process for defending against model poisoning
attacks inadvertently leaks privacy to one of the honest-but-curious servers,
enabling it to access users' gradients in plaintext. To address both privacy
leakage and model poisoning attacks, we propose an enhanced privacy-preserving
and Byzantine-robust federated learning (PBFL) scheme, comprising three
components: (1) a two-trapdoor fully homomorphic encryption (FHE) scheme to
bolster users' privacy protection; (2) a novel secure normalization judgment
method to preemptively thwart gradient poisoning; and (3) an innovative secure
cosine similarity measurement method for detecting model poisoning attacks
without compromising data privacy. Our scheme guarantees privacy preservation
and resilience against model poisoning attacks, even in scenarios with
heterogeneous, non-IID (Independently and Identically Distributed) datasets.
Theoretical analyses substantiate the security and efficiency of our scheme,
and extensive experiments corroborate the efficacy of our private attacks.
Furthermore, the experimental results demonstrate that our scheme accelerates
training speed while reducing communication overhead compared to the
state-of-the-art PBFL schemes.

### 4. [Breaking Out from the TESSERACT: Reassessing ML-based Malware Detection under Spatio-Temporal Drift](http://arxiv.org/pdf/2506.23814v1)

Authors: Theo Chow, Mario D'Onghia, Lorenz Linhardt, Zeliang Kan, Daniel Arp, Lorenzo Cavallaro, Fabio Pierazzi

Several recent works focused on the best practices for applying machine
learning to cybersecurity. In the context of malware, TESSERACT highlighted the
impact of concept drift on detection performance and suggested temporal and
spatial constraints to be enforced to ensure realistic time-aware evaluations,
which have been adopted by the community. In this paper, we demonstrate
striking discrepancies in the performance of learning-based malware detection
across the same time frame when evaluated on two representative Android malware
datasets used in top-tier security conferences, both adhering to established
sampling and evaluation guidelines. This questions our ability to understand
how current state-of-the-art approaches would perform in realistic scenarios.
To address this, we identify five novel temporal and spatial bias factors that
affect realistic evaluations. We thoroughly evaluate the impact of these
factors in the Android malware domain on two representative datasets and five
Android malware classifiers used or proposed in top-tier security conferences.
For each factor, we provide practical and actionable recommendations that the
community should integrate in their methodology for more realistic and
reproducible settings.

### 5. [Poisoning Attacks to Local Differential Privacy for Ranking Estimation](http://arxiv.org/pdf/2506.24033v1)

Authors: Pei Zhan, Peng Tang, Yangzhuo Li, Puwen Wei, Shanqing Guo

Local differential privacy (LDP) involves users perturbing their inputs to
provide plausible deniability of their data. However, this also makes LDP
vulnerable to poisoning attacks. In this paper, we first introduce novel
poisoning attacks for ranking estimation. These attacks are intricate, as fake
attackers do not merely adjust the frequency of target items. Instead, they
leverage a limited number of fake users to precisely modify frequencies,
effectively altering item rankings to maximize gains. To tackle this challenge,
we introduce the concepts of attack cost and optimal attack item (set), and
propose corresponding strategies for kRR, OUE, and OLH protocols. For kRR, we
iteratively select optimal attack items and allocate suitable fake users. For
OUE, we iteratively determine optimal attack item sets and consider the
incremental changes in item frequencies across different sets. Regarding OLH,
we develop a harmonic cost function based on the pre-image of a hash to select
that supporting a larger number of effective attack items. Lastly, we present
an attack strategy based on confidence levels to quantify the probability of a
successful attack and the number of attack iterations more precisely. We
demonstrate the effectiveness of our attacks through theoretical and empirical
evidence, highlighting the necessity for defenses against these attacks. The
source code and data have been made available at
https://github.com/LDP-user/LDP-Ranking.git.

### 6. [All Proof of Work But No Proof of Play](http://arxiv.org/pdf/2506.23435v1)

Authors: Hayder Tirmazi

Speedrunning is a competition that emerged from communities of early video
games such as Doom (1993). Speedrunners try to finish a game in minimal time.
Provably verifying the authenticity of submitted speedruns is an open problem.
Traditionally, best-effort speedrun verification is conducted by on-site human
observers, forensic audio analysis, or a rigorous mathematical analysis of the
game mechanics. Such methods are tedious, fallible, and, perhaps worst of all,
not cryptographic. Motivated by naivety and the Dunning-Kruger effect, we
attempt to build a system that cryptographically proves the authenticity of
speedruns. This paper describes our attempted solutions and ways to circumvent
them. Through a narration of our failures, we attempt to demonstrate the
difficulty of authenticating live and interactive human input in untrusted
environments, as well as the limits of signature schemes, game integrity, and
provable play.

### 7. [Unbounded knapsack problem and double partitions](http://arxiv.org/pdf/2506.23499v1)

Authors: Boris Y. Rubinstein

The unbounded knapsack problem can be considered as a particular case of the
double partition problem that asks for a number of nonnegative integer
solutions to a system of two linear Diophantine equations with integer
coefficients. In the middle of 19th century Sylvester and Cayley suggested an
approach based on the variable elimination allowing a reduction of a double
partition to a sum of scalar partitions. This manuscript discusses a geometric
interpretation of this method and its application to the knapsack problem.

### 8. [SoK: Semantic Privacy in Large Language Models](http://arxiv.org/pdf/2506.23603v1)

Authors: Baihe Ma, Yanna Jiang, Xu Wang, Guangshen Yu, Qin Wang, Caijun Sun, Chen Li, Xuelei Qi, Ying He, Wei Ni, Ren Ping Liu

As Large Language Models (LLMs) are increasingly deployed in sensitive
domains, traditional data privacy measures prove inadequate for protecting
information that is implicit, contextual, or inferable - what we define as
semantic privacy. This Systematization of Knowledge (SoK) introduces a
lifecycle-centric framework to analyze how semantic privacy risks emerge across
input processing, pretraining, fine-tuning, and alignment stages of LLMs. We
categorize key attack vectors and assess how current defenses, such as
differential privacy, embedding encryption, edge computing, and unlearning,
address these threats. Our analysis reveals critical gaps in semantic-level
protection, especially against contextual inference and latent representation
leakage. We conclude by outlining open challenges, including quantifying
semantic leakage, protecting multimodal inputs, balancing de-identification
with generation quality, and ensuring transparency in privacy enforcement. This
work aims to inform future research on designing robust, semantically aware
privacy-preserving techniques for LLMs.

### 9. [gMBA: Expression Semantic Guided Mixed Boolean-Arithmetic Deobfuscation Using Transformer Architectures](http://arxiv.org/pdf/2506.23634v1)

Authors: Youjeong Noh, Joon-Young Paik, Jingun Kwon, Eun-Sun Cho

Mixed Boolean-Arithmetic (MBA) obfuscation protects intellectual property by
converting programs into forms that are more complex to analyze. However, MBA
has been increasingly exploited by malware developers to evade detection and
cause significant real-world problems. Traditional MBA deobfuscation methods
often consider these expressions as part of a black box and overlook their
internal semantic information. To bridge this gap, we propose a truth table,
which is an automatically constructed semantic representation of an
expression's behavior that does not rely on external resources. The truth table
is a mathematical form that represents the output of expression for all
possible combinations of input. We also propose a general and extensible guided
MBA deobfuscation framework (gMBA) that modifies a Transformer-based neural
encoder-decoder Seq2Seq architecture to incorporate this semantic guidance.
Experimental results and in-depth analysis show that integrating expression
semantics significantly improves performance and highlights the importance of
internal semantic expressions in recovering obfuscated code to its original
form.

### 10. [An ontological lens on attack trees: Toward adequacy and interoperability](http://arxiv.org/pdf/2506.23841v1)

Authors: Ítalo Oliveira, Stefano M. Nicoletti, Gal Engelberg, Mattia Fumagalli, Dan Klein, Giancarlo Guizzardi

Attack Trees (AT) are a popular formalism for security analysis. They are
meant to display an attacker's goal decomposed into attack steps needed to
achieve it and compute certain security metrics (e.g., attack cost,
probability, and damage). ATs offer three important services: (a) conceptual
modeling capabilities for representing security risk management scenarios, (b)
a qualitative assessment to find root causes and minimal conditions of
successful attacks, and (c) quantitative analyses via security metrics
computation under formal semantics, such as minimal time and cost among all
attacks. Still, the AT language presents limitations due to its lack of
ontological foundations, thus compromising associated services. Via an
ontological analysis grounded in the Common Ontology of Value and Risk (COVER)
-- a reference core ontology based on the Unified Foundational Ontology (UFO)
-- we investigate the ontological adequacy of AT and reveal four significant
shortcomings: (1) ambiguous syntactical terms that can be interpreted in
various ways; (2) ontological deficit concerning crucial domain-specific
concepts; (3) lacking modeling guidance to construct ATs decomposing a goal;
(4) lack of semantic interoperability, resulting in ad hoc stand-alone tools.
We also discuss existing incremental solutions and how our analysis paves the
way for overcoming those issues through a broader approach to risk management
modeling.

### Computer Vision and Pattern Recognition

### 1. [PathDiff: Histopathology Image Synthesis with Unpaired Text and Mask Conditions](http://arxiv.org/pdf/2506.23440v1)

Authors: Mahesh Bhosale, Abdul Wasi, Yuanhao Zhai, Yunjie Tian, Samuel Border, Nan Xi, Pinaki Sarder, Junsong Yuan, David Doermann, Xuan Gong

Diffusion-based generative models have shown promise in synthesizing
histopathology images to address data scarcity caused by privacy constraints.
Diagnostic text reports provide high-level semantic descriptions, and masks
offer fine-grained spatial structures essential for representing distinct
morphological regions. However, public datasets lack paired text and mask data
for the same histopathological images, limiting their joint use in image
generation. This constraint restricts the ability to fully exploit the benefits
of combining both modalities for enhanced control over semantics and spatial
details. To overcome this, we propose PathDiff, a diffusion framework that
effectively learns from unpaired mask-text data by integrating both modalities
into a unified conditioning space. PathDiff allows precise control over
structural and contextual features, generating high-quality, semantically
accurate images. PathDiff also improves image fidelity, text-image alignment,
and faithfulness, enhancing data augmentation for downstream tasks like nuclei
segmentation and classification. Extensive experiments demonstrate its
superiority over existing methods.

### 2. [Contrastive Learning with Diffusion Features for Weakly Supervised Medical Image Segmentation](http://arxiv.org/pdf/2506.23460v1)

Authors: Dewen Zeng, Xinrong Hu, Yu-Jen Chen, Yawen Wu, Xiaowei Xu, Yiyu Shi

Weakly supervised semantic segmentation (WSSS) methods using class labels
often rely on class activation maps (CAMs) to localize objects. However,
traditional CAM-based methods struggle with partial activations and imprecise
object boundaries due to optimization discrepancies between classification and
segmentation. Recently, the conditional diffusion model (CDM) has been used as
an alternative for generating segmentation masks in WSSS, leveraging its strong
image generation capabilities tailored to specific class distributions. By
modifying or perturbing the condition during diffusion sampling, the related
objects can be highlighted in the generated images. Yet, the saliency maps
generated by CDMs are prone to noise from background alterations during reverse
diffusion. To alleviate the problem, we introduce Contrastive Learning with
Diffusion Features (CLDF), a novel method that uses contrastive learning to
train a pixel decoder to map the diffusion features from a frozen CDM to a
low-dimensional embedding space for segmentation. Specifically, we integrate
gradient maps generated from CDM external classifier with CAMs to identify
foreground and background pixels with fewer false positives/negatives for
contrastive learning, enabling robust pixel embedding learning. Experimental
results on four segmentation tasks from two public medical datasets demonstrate
that our method significantly outperforms existing baselines.

### 3. [NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](http://arxiv.org/pdf/2506.23468v1)

Authors: Xuan Yao, Junyu Gao, Changsheng Xu

Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires
agents to execute sequential navigation actions in complex environments guided
by natural language instructions. Current approaches often struggle with
generalizing to novel environments and adapting to ongoing changes during
navigation. Inspired by human cognition, we present NavMorph, a self-evolving
world model framework that enhances environmental understanding and
decision-making in VLN-CE tasks. NavMorph employs compact latent
representations to model environmental dynamics, equipping agents with
foresight for adaptive planning and policy refinement. By integrating a novel
Contextual Evolution Memory, NavMorph leverages scene-contextual information to
support effective navigation while maintaining online adaptability. Extensive
experiments demonstrate that our method achieves notable performance
improvements on popular VLN-CE benchmarks. Code is available at
\href{https://github.com/Feliciaxyao/NavMorph}{this https URL}.

### 4. [Interactive Interface For Semantic Segmentation Dataset Synthesis](http://arxiv.org/pdf/2506.23470v1)

Authors: Ngoc-Do Tran, Minh-Tuan Huynh, Tam V. Nguyen, Minh-Triet Tran, Trung-Nghia Le

The rapid advancement of AI and computer vision has significantly increased
the demand for high-quality annotated datasets, particularly for semantic
segmentation. However, creating such datasets is resource-intensive, requiring
substantial time, labor, and financial investment, and often raises privacy
concerns due to the use of real-world data. To mitigate these challenges, we
present SynthLab, consisting of a modular platform for visual data synthesis
and a user-friendly interface. The modular architecture of SynthLab enables
easy maintenance, scalability with centralized updates, and seamless
integration of new features. Each module handles distinct aspects of computer
vision tasks, enhancing flexibility and adaptability. Meanwhile, its
interactive, user-friendly interface allows users to quickly customize their
data pipelines through drag-and-drop actions. Extensive user studies involving
a diverse range of users across different ages, professions, and expertise
levels, have demonstrated flexible usage, and high accessibility of SynthLab,
enabling users without deep technical expertise to harness AI for real-world
applications.

### 5. [GeoCD: A Differential Local Approximation for Geodesic Chamfer Distance](http://arxiv.org/pdf/2506.23478v1)

Authors: Pedro Alonso, Tianrui Li, Chongshou Li

Chamfer Distance (CD) is a widely adopted metric in 3D point cloud learning
due to its simplicity and efficiency. However, it suffers from a fundamental
limitation: it relies solely on Euclidean distances, which often fail to
capture the intrinsic geometry of 3D shapes. To address this limitation, we
propose GeoCD, a topology-aware and fully differentiable approximation of
geodesic distance designed to serve as a metric for 3D point cloud learning.
Our experiments show that GeoCD consistently improves reconstruction quality
over standard CD across various architectures and datasets. We demonstrate this
by fine-tuning several models, initially trained with standard CD, using GeoCD.
Remarkably, fine-tuning for a single epoch with GeoCD yields significant gains
across multiple evaluation metrics.

### 6. [Instant GaussianImage: A Generalizable and Self-Adaptive Image Representation via 2D Gaussian Splatting](http://arxiv.org/pdf/2506.23479v1)

Authors: Zhaojie Zeng, Yuesong Wang, Chao Yang, Tao Guan, Lili Ju

Implicit Neural Representation (INR) has demonstrated remarkable advances in
the field of image representation but demands substantial GPU resources.
GaussianImage recently pioneered the use of Gaussian Splatting to mitigate this
cost, however, the slow training process limits its practicality, and the fixed
number of Gaussians per image limits its adaptability to varying information
entropy. To address these issues, we propose in this paper a generalizable and
self-adaptive image representation framework based on 2D Gaussian Splatting.
Our method employs a network to quickly generate a coarse Gaussian
representation, followed by minimal fine-tuning steps, achieving comparable
rendering quality of GaussianImage while significantly reducing training time.
Moreover, our approach dynamically adjusts the number of Gaussian points based
on image complexity to further enhance flexibility and efficiency in practice.
Experiments on DIV2K and Kodak datasets show that our method matches or exceeds
GaussianImage's rendering performance with far fewer iterations and shorter
training times. Specifically, our method reduces the training time by up to one
order of magnitude while achieving superior rendering performance with the same
number of Gaussians.

### 7. [MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](http://arxiv.org/pdf/2506.23482v1)

Authors: Jun Huang, Ting Liu, Yihang Wu, Xiaochao Qu, Luoqi Liu, Xiaolin Hu

Advancements in generative models have enabled image inpainting models to
generate content within specific regions of an image based on provided prompts
and masks. However, existing inpainting methods often suffer from problems such
as semantic misalignment, structural distortion, and style inconsistency. In
this work, we present MTADiffusion, a Mask-Text Alignment diffusion model
designed for object inpainting. To enhance the semantic capabilities of the
inpainting model, we introduce MTAPipeline, an automatic solution for
annotating masks with detailed descriptions. Based on the MTAPipeline, we
construct a new MTADataset comprising 5 million images and 25 million mask-text
pairs. Furthermore, we propose a multi-task training strategy that integrates
both inpainting and edge prediction tasks to improve structural stability. To
promote style consistency, we present a novel inpainting style-consistency loss
using a pre-trained VGG network and the Gram matrix. Comprehensive evaluations
on BrushBench and EditBench demonstrate that MTADiffusion achieves
state-of-the-art performance compared to other methods.

### 8. [LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching](http://arxiv.org/pdf/2506.23502v1)

Authors: Mengxiao Tian, Xinxiao Wu, Shuo Yang

Driven by large-scale contrastive vision-language pre-trained models such as
CLIP, recent advancements in the image-text matching task have achieved
remarkable success in representation learning. Due to image-level
visual-language alignment, CLIP falls short in understanding fine-grained
details such as object attributes and spatial relationships between objects.
Recent efforts have attempted to compel CLIP to acquire structured visual
representations by introducing prompt learning to achieve object-level
alignment. While achieving promising results, they still lack the capability to
perceive actions, which are crucial for describing the states or relationships
between objects. Therefore, we propose to endow CLIP with fine-grained
action-level understanding by introducing an LLM-enhanced action-aware
multi-modal prompt-tuning method, incorporating the action-related external
knowledge generated by large language models (LLMs). Specifically, we design an
action triplet prompt and an action state prompt to exploit compositional
semantic knowledge and state-related causal knowledge implicitly stored in
LLMs. Subsequently, we propose an adaptive interaction module to aggregate
attentive visual features conditioned on action-aware prompted knowledge for
establishing discriminative and action-aware visual representations, which
further improves the performance. Comprehensive experimental results on two
benchmark datasets demonstrate the effectiveness of our method.

### 9. [Improve Underwater Object Detection through YOLOv12 Architecture and Physics-informed Augmentation](http://arxiv.org/pdf/2506.23505v1)

Authors: Tinh Nguyen

Underwater object detection is crucial for autonomous navigation,
environmental monitoring, and marine exploration, but it is severely hampered
by light attenuation, turbidity, and occlusion. Current methods balance
accuracy and computational efficiency, but they have trouble deploying in
real-time under low visibility conditions. Through the integration of
physics-informed augmentation techniques with the YOLOv12 architecture, this
study advances underwater detection. With Residual ELAN blocks to preserve
structural features in turbid waters and Area Attention to maintain large
receptive fields for occluded objects while reducing computational complexity.
Underwater optical properties are addressed by domain-specific augmentations
such as turbulence adaptive blurring, biologically grounded occlusion
simulation, and spectral HSV transformations for color distortion. Extensive
tests on four difficult datasets show state-of-the-art performance, with
Brackish data registering 98.30% mAP at 142 FPS. YOLOv12 improves occlusion
robustness by 18.9%, small-object recall by 22.4%, and detection precision by
up to 7.94% compared to previous models. The crucial role of augmentation
strategy is validated by ablation studies. This work offers a precise and
effective solution for conservation and underwater robotics applications.

### 10. [ViewPoint: Panoramic Video Generation with Pretrained Diffusion Models](http://arxiv.org/pdf/2506.23513v1)

Authors: Zixun Fang, Kai Zhu, Zhiheng Liu, Yu Liu, Wei Zhai, Yang Cao, Zheng-Jun Zha

Panoramic video generation aims to synthesize 360-degree immersive videos,
holding significant importance in the fields of VR, world models, and spatial
intelligence. Existing works fail to synthesize high-quality panoramic videos
due to the inherent modality gap between panoramic data and perspective data,
which constitutes the majority of the training data for modern diffusion
models. In this paper, we propose a novel framework utilizing pretrained
perspective video models for generating panoramic videos. Specifically, we
design a novel panorama representation named ViewPoint map, which possesses
global spatial continuity and fine-grained visual details simultaneously. With
our proposed Pano-Perspective attention mechanism, the model benefits from
pretrained perspective priors and captures the panoramic spatial correlations
of the ViewPoint map effectively. Extensive experiments demonstrate that our
method can synthesize highly dynamic and spatially consistent panoramic videos,
achieving state-of-the-art performance and surpassing previous methods.

### Computers and Society

### 1. [Evaluating the Simulation of Human Personality-Driven Susceptibility to Misinformation with LLMs](http://arxiv.org/pdf/2506.23610v1)

Authors: Manuel Pratelli, Marinella Petrocchi

Large language models (LLMs) make it possible to generate synthetic
behavioural data at scale, offering an ethical and low-cost alternative to
human experiments. Whether such data can faithfully capture psychological
differences driven by personality traits, however, remains an open question. We
evaluate the capacity of LLM agents, conditioned on Big-Five profiles, to
reproduce personality-based variation in susceptibility to misinformation,
focusing on news discernment, the ability to judge true headlines as true and
false headlines as false. Leveraging published datasets in which human
participants with known personality profiles rated headline accuracy, we create
matching LLM agents and compare their responses to the original human patterns.
Certain trait-misinformation associations, notably those involving
Agreeableness and Conscientiousness, are reliably replicated, whereas others
diverge, revealing systematic biases in how LLMs internalize and express
personality. The results underscore both the promise and the limits of
personality-aligned LLMs for behavioral simulation, and offer new insight into
modeling cognitive diversity in artificial agents.

### 2. [Leveraging a Multi-Agent LLM-Based System to Educate Teachers in Hate Incidents Management](http://arxiv.org/pdf/2506.23774v1)

Authors: Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jarosław A. Chudziak

Computer-aided teacher training is a state-of-the-art method designed to
enhance teachers' professional skills effectively while minimising concerns
related to costs, time constraints, and geographical limitations. We
investigate the potential of large language models (LLMs) in teacher education,
using a case of teaching hate incidents management in schools. To this end, we
create a multi-agent LLM-based system that mimics realistic situations of hate,
using a combination of retrieval-augmented prompting and persona modelling. It
is designed to identify and analyse hate speech patterns, predict potential
escalation, and propose effective intervention strategies. By integrating
persona modelling with agentic LLMs, we create contextually diverse simulations
of hate incidents, mimicking real-life situations. The system allows teachers
to analyse and understand the dynamics of hate incidents in a safe and
controlled environment, providing valuable insights and practical knowledge to
manage such situations confidently in real life. Our pilot evaluation
demonstrates teachers' enhanced understanding of the nature of annotator
disagreements and the role of context in hate speech interpretation, leading to
the development of more informed and effective strategies for addressing hate
in classroom settings.

### 3. [Beyond Distance: Mobility Neural Embeddings Reveal Visible and Invisible Barriers in Urban Space](http://arxiv.org/pdf/2506.24061v1)

Authors: Guangyuan Weng, Minsuk Kim, Yong-Yeol Ahn, Esteban Moro

Human mobility in cities is shaped not only by visible structures such as
highways, rivers, and parks but also by invisible barriers rooted in
socioeconomic segregation, uneven access to amenities, and administrative
divisions. Yet identifying and quantifying these barriers at scale and their
relative importance on people's movements remains a major challenge. Neural
embedding models, originally developed for language, offer a powerful way to
capture the complexity of human mobility from large-scale data. Here, we apply
this approach to 25.4 million observed trajectories across 11 major U.S.
cities, learning mobility embeddings that reveal how people move through urban
space. These mobility embeddings define a functional distance between places,
one that reflects behavioral rather than physical proximity, and allow us to
detect barriers between neighborhoods that are geographically close but
behaviorally disconnected. We find that the strongest predictors of these
barriers are differences in access to amenities, administrative borders, and
residential segregation by income and race. These invisible borders are
concentrated in urban cores and persist across cities, spatial scales, and time
periods. Physical infrastructure, such as highways and parks, plays a secondary
but still significant role, especially at short distances. We also find that
individuals who cross barriers tend to do so outside of traditional commuting
hours and are more likely to live in areas with greater racial diversity, and
higher transit use or income. Together, these findings reveal how spatial,
social, and behavioral forces structure urban accessibility and provide a
scalable framework to detect and monitor barriers in cities, with applications
in planning, policy evaluation, and equity analysis.

### 4. [Scaling Human Judgment in Community Notes with LLMs](http://arxiv.org/pdf/2506.24118v1)

Authors: Haiwen Li, Soham De, Manon Revel, Andreas Haupt, Brad Miller, Keith Coleman, Jay Baxter, Martin Saveski, Michiel A. Bakker

This paper argues for a new paradigm for Community Notes in the LLM era: an
open ecosystem where both humans and LLMs can write notes, and the decision of
which notes are helpful enough to show remains in the hands of humans. This
approach can accelerate the delivery of notes, while maintaining trust and
legitimacy through Community Notes' foundational principle: A community of
diverse human raters collectively serve as the ultimate evaluator and arbiter
of what is helpful. Further, the feedback from this diverse community can be
used to improve LLMs' ability to produce accurate, unbiased, broadly helpful
notes--what we term Reinforcement Learning from Community Feedback (RLCF). This
becomes a two-way street: LLMs serve as an asset to humans--helping deliver
context quickly and with minimal effort--while human feedback, in turn,
enhances the performance of LLMs. This paper describes how such a system can
work, its benefits, key new risks and challenges it introduces, and a research
agenda to solve those challenges and realize the potential of this approach.

### 5. [Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts](http://arxiv.org/pdf/2506.23845v1)

Authors: Kenny Peng, Rajiv Movva, Jon Kleinberg, Emma Pierson, Nikhil Garg

While sparse autoencoders (SAEs) have generated significant excitement, a
series of negative results have added to skepticism about their usefulness.
Here, we establish a conceptual distinction that reconciles competing
narratives surrounding SAEs. We argue that while SAEs may be less effective for
acting on known concepts, SAEs are powerful tools for discovering unknown
concepts. This distinction cleanly separates existing negative and positive
results, and suggests several classes of SAE applications. Specifically, we
outline use cases for SAEs in (i) ML interpretability, explainability,
fairness, auditing, and safety, and (ii) social and health sciences.

### 6. [Comparative Studies: Cloud-Enabled Adaptive Learning System for Scalable Education in Sub-Saharan](http://arxiv.org/pdf/2506.23851v1)

Authors: Israel Fianyi, Soonja Yeom, Ju-Hyun Shin

The integration of cloud computing in education can revolutionise learning in
advanced (Australia & South Korea) and middle-income (Ghana & Nigeria)
countries, while offering scalable, cost-effective and equitable access to
adaptive learning systems. This paper explores how cloud computing and adaptive
learning technologies are deployed across different socio-economic and
infrastructure contexts. The study identifies enabling factors and systematic
challenges, providing insights into how cloud-based education can be tailored
to bridge the digital and educational divide globally.

### 7. [Exploring Privacy and Security as Drivers for Environmental Sustainability in Cloud-Based Office Solutions](http://arxiv.org/pdf/2506.23866v1)

Authors: Jason Kayembe, Iness Ben Guirat, Jan Tobias Mühlberg

In this paper, we explore the intersection of privacy, security, and
environmental sustainability in cloud-based office solutions, focusing on
quantifying user- and network-side energy use and associated carbon emissions.
We hypothesise that privacy-focused services are typically more
energy-efficient than those funded through data collection and advertising. To
evaluate this, we propose a framework that systematically measures
environmental costs based on energy usage and network data traffic during
well-defined, automated usage scenarios. To test our hypothesis, we first
analyse how underlying architectures and business models, such as monetisation
through personalised advertising, contribute to the environmental footprint of
these services. We then explore existing methodologies and tools for software
environmental impact assessment. We apply our framework to three mainstream
email services selected to reflect different privacy policies, from
ad-supported tracking-intensive models to privacy-focused designs: Microsoft
Outlook, Google Mail (Gmail), and Proton Mail. We extend this comparison to a
self-hosted email solution, evaluated with and without end-to-end encryption.
We show that the self-hosted solution, even with 14% of device energy and 15%
of emissions overheads from PGP encryption, remains the most energy-efficient,
saving up to 33% of emissions per session compared to Gmail. Among commercial
providers, Proton Mail is the most efficient, saving up to 0.1 gCO2 e per
session compared to Outlook, whose emissions can be further reduced by 2%
through ad-blocking.

### 8. [AI Risk-Management Standards Profile for General-Purpose AI (GPAI) and Foundation Models](http://arxiv.org/pdf/2506.23949v1)

Authors: Anthony M. Barrett, Jessica Newman, Brandie Nonnecke, Nada Madkour, Dan Hendrycks, Evan R. Murphy, Krystal Jackson, Deepika Raman

Increasingly multi-purpose AI models, such as cutting-edge large language
models or other 'general-purpose AI' (GPAI) models, 'foundation models,'
generative AI models, and 'frontier models' (typically all referred to
hereafter with the umbrella term 'GPAI/foundation models' except where greater
specificity is needed), can provide many beneficial capabilities but also risks
of adverse events with profound consequences. This document provides
risk-management practices or controls for identifying, analyzing, and
mitigating risks of GPAI/foundation models. We intend this document primarily
for developers of large-scale, state-of-the-art GPAI/foundation models; others
that can benefit from this guidance include downstream developers of end-use
applications that build on a GPAI/foundation model. This document facilitates
conformity with or use of leading AI risk management-related standards,
adapting and building on the generic voluntary guidance in the NIST AI Risk
Management Framework and ISO/IEC 23894, with a focus on the unique issues faced
by developers of GPAI/foundation models.

### 9. [Green Metrics Tool: Measuring for fun and profit](http://arxiv.org/pdf/2506.23967v1)

Authors: Geerd-Dietger Hoffmann, Verena Majuntke

The environmental impact of software is gaining increasing attention as the
demand for computational resources continues to rise. In order to optimize
software resource consumption and reduce carbon emissions, measuring and
evaluating software is a first essential step. In this paper we discuss what
metrics are important for fact base decision making. We introduce the Green
Metrics Tool (GMT), a novel framework for accurately measuring the resource
consumption of software. The tool provides a containerized, controlled, and
reproducible life cycle-based approach, assessing the resource use of software
during key phases. Finally, we discuss GMT features like visualization,
comparability and rule- and LLM-based optimisations highlighting its potential
to guide developers and researchers in reducing the environmental impact of
their software.

### 10. [LLM Agents Are the Antidote to Walled Gardens](http://arxiv.org/pdf/2506.23978v1)

Authors: Samuele Marro, Philip Torr

While the Internet's core infrastructure was designed to be open and
universal, today's application layer is dominated by closed, proprietary
platforms. Open and interoperable APIs require significant investment, and
market leaders have little incentive to enable data exchange that could erode
their user lock-in. We argue that LLM-based agents fundamentally disrupt this
status quo. Agents can automatically translate between data formats and
interact with interfaces designed for humans: this makes interoperability
dramatically cheaper and effectively unavoidable. We name this shift universal
interoperability: the ability for any two digital services to exchange data
seamlessly using AI-mediated adapters. Universal interoperability undermines
monopolistic behaviours and promotes data portability. However, it can also
lead to new security risks and technical debt. Our position is that the ML
community should embrace this development while building the appropriate
frameworks to mitigate the downsides. By acting now, we can harness AI to
restore user freedom and competitive markets without sacrificing security.

### Databases

### 1. [Lock Prediction for Zero-Downtime Database Encryption](http://arxiv.org/pdf/2506.23985v1)

Authors: Mohamed Sami Rakha, Adam Sorrenti, Greg Stager, Walid Rjaibi, Andriy Miranskyy

Modern enterprise database systems face significant challenges in balancing
data security and performance. Ensuring robust encryption for sensitive
information is critical for systems' compliance with security standards.
Although holistic database encryption provides strong protection, existing
database systems often require a complete backup and restore cycle, resulting
in prolonged downtime and increased storage usage. This makes it difficult to
implement online encryption techniques in high-throughput environments without
disrupting critical operations.
  To address this challenge, we envision a solution that enables online
database encryption aligned with system activity, eliminating the need for
downtime, storage overhead, or full-database reprocessing. Central to this
vision is the ability to predict which parts of the database will be accessed
next, allowing encryption to be applied online. As a step towards this
solution, this study proposes a predictive approach that leverages deep
learning models to forecast database lock sequences, using IBM Db2 as the
database system under study. In this study, we collected a specialized dataset
from TPC-C benchmark workloads, leveraging lock event logs for model training
and evaluation. We applied deep learning architectures, such as Transformer and
LSTM, to evaluate models for various table-level and page-level lock
predictions. We benchmark the accuracy of the trained models versus a Naive
Baseline across different prediction horizons and timelines.
  The study experiments demonstrate that the proposed deep learning-based
models achieve up to 49% average accuracy for table-level and 66% for
page-level predictions, outperforming a Naive Baseline. By anticipating which
tables and pages will be locked next, the proposed approach is a step toward
online encryption, offering a practical path toward secure, low-overhead
database systems.

### Distributed, Parallel, and Cluster Computing

### 1. [Large-scale Neural Network Quantum States for ab initio Quantum Chemistry Simulations on Fugaku](http://arxiv.org/pdf/2506.23809v1)

Authors: Hongtao Xu, Zibo Wu, Mingzhen Li, Weile Jia

Solving quantum many-body problems is one of the fundamental challenges in
quantum chemistry. While neural network quantum states (NQS) have emerged as a
promising computational tool, its training process incurs exponentially growing
computational demands, becoming prohibitively expensive for large-scale
molecular systems and creating fundamental scalability barriers for real-world
applications. To address above challenges, we present \ours, a high-performance
NQS training framework for \textit{ab initio} electronic structure
calculations. First, we propose a scalable sampling parallelism strategy with
multi-layers workload division and hybrid sampling scheme, which break the
scalability barriers for large-scale NQS training. Then, we introduce
multi-level parallelism local energy parallelism, enabling more efficient local
energy computation. Last, we employ cache-centric optimization for
transformer-based \textit{ansatz} and incorporate it with sampling parallelism
strategy, which further speedup up the NQS training and achieve stable memory
footprint at scale. Experiments demonstrate that \ours accelerate NQS training
with up to 8.41x speedup and attains a parallel efficiency up to 95.8\% when
scaling to 1,536 nodes.

### 2. [Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC](http://arxiv.org/pdf/2506.24045v1)

Authors: Xinming Wei, Jiahao Zhang, Haoran Li, Jiayu Chen, Rui Qu, Maoliang Li, Xiang Chen, Guojie Luo

The proliferation of agentic Large Language Models (LLMs) on personal devices
introduces a new class of workloads characterized by a dichotomy of objectives.
Reactive tasks, initiated by users, demand immediate, low-latency responses,
while proactive tasks operate invisibly and prioritize throughput. Existing
on-device LLM engines, designed for isolated inferences, fail to efficiently
manage these concurrent and conflicting requests on consumer-grade
heterogeneous SoCs with CPU, integrated GPU, and NPU. This paper introduces
Agent.xpu, an efficient serving system for agentic LLM workloads on
memory-unified heterogeneous SoCs. With dedicated offline profiling, Agent.xpu
first constructs a heterogeneous execution graph, which fuses and chunks model
kernels for affinity-guided, elastic accelerator mapping with predictive kernel
annotation. At runtime, its online scheduler enables fine-grained, kernel-level
preemption to guarantee the responsiveness of reactive tasks. To maximize SoC
utilization, it adopts slack-aware kernel backfill to opportunistically append
proactive tasks, and mitigates NPU-iGPU contention via bandwidth-aware
dispatch. Evaluation on an Intel Core Ultra SoC shows that Agent.xpu achieves
4.6$\times$ lower latency for reactive tasks and sustains
1.6$\times$-6.8$\times$ higher throughput for proactive tasks compared to
state-of-the-art inference engines.

### 3. [Detect \& Score: Privacy-Preserving Misbehaviour Detection and Contribution Evaluation in Federated Learning](http://arxiv.org/pdf/2506.23583v1)

Authors: Marvin Xhemrishi, Alexandre Graell i Amat, Balázs Pejó

Federated learning with secure aggregation enables private and collaborative
learning from decentralised data without leaking sensitive client information.
However, secure aggregation also complicates the detection of malicious client
behaviour and the evaluation of individual client contributions to the
learning. To address these challenges, QI (Pejo et al.) and FedGT (Xhemrishi et
al.) were proposed for contribution evaluation (CE) and misbehaviour detection
(MD), respectively. QI, however, lacks adequate MD accuracy due to its reliance
on the random selection of clients in each training round, while FedGT lacks
the CE ability. In this work, we combine the strengths of QI and FedGT to
achieve both robust MD and accurate CE. Our experiments demonstrate superior
performance compared to using either method independently.

### 4. [Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model](http://arxiv.org/pdf/2506.23635v1)

Authors: Mu-Chi Chen, Po-Hsuan Huang, Xiangrui Ke, Chia-Heng Tu, Chun Jason Xue, Shih-Hao Hung

Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI)
with significant advancements such as OpenAI's ChatGPT, Meta's Llama, and
Databricks' DBRX. This paper addresses the cost and scalability challenges
encountered when constructing private LLM systems for personal or small group
services, as aimed by Apple Intelligence. A Mac Studio cluster with Apple's M2
Ultra chips is established as a cost-efficient solution to host and accelerate
the pretrained DBRX model with the Mixture-of-Experts (MoE) architecture. Our
performance analysis reveal that parallel execution of the model's experts
across two to four machine nodes significantly reduces inference time. We find
that computation time for the experts is comparable to the communication time
for exchanging their outputs, emphasizing the importance of network latency
over bandwidth. We also observe significant management overhead due to Apple
software stack's memory management logic. Based on these findings, we develop
optimization schemes to eliminate the memory management overhead. As a result,
the Mac Studio cluster is 1.15 times more cost-efficient than the
state-of-the-art AI supercomputer with NVIDIA H100 GPUs. In addition, we
construct a performance model to estimate system performance under varying
configurations, and the model provides valuable insights for designing private
LLM systems.

### 5. [Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction](http://arxiv.org/pdf/2506.23836v1)

Authors: Alexander Tyurin

We consider centralized distributed optimization in the classical federated
learning setup, where $n$ workers jointly find an $\varepsilon$-stationary
point of an $L$-smooth, $d$-dimensional nonconvex function $f$, having access
only to unbiased stochastic gradients with variance $\sigma^2$. Each worker
requires at most $h$ seconds to compute a stochastic gradient, and the
communication times from the server to the workers and from the workers to the
server are $\tau_{s}$ and $\tau_{w}$ seconds per coordinate, respectively. One
of the main motivations for distributed optimization is to achieve scalability
with respect to $n$. For instance, it is well known that the distributed
version of SGD has a variance-dependent runtime term $\frac{h \sigma^2 L
\Delta}{n \varepsilon^2},$ which improves with the number of workers $n,$ where
$\Delta = f(x^0) - f^*,$ and $x^0 \in R^d$ is the starting point. Similarly,
using unbiased sparsification compressors, it is possible to reduce both the
variance-dependent runtime term and the communication runtime term. However,
once we account for the communication from the server to the workers
$\tau_{s}$, we prove that it becomes infeasible to design a method using
unbiased random sparsification compressors that scales both the server-side
communication runtime term $\tau_{s} d \frac{L \Delta}{\varepsilon}$ and the
variance-dependent runtime term $\frac{h \sigma^2 L \Delta}{\varepsilon^2},$
better than poly-logarithmically in $n$, even in the homogeneous (i.i.d.) case,
where all workers access the same distribution. To establish this result, we
construct a new "worst-case" function and develop a new lower bound framework
that reduces the analysis to the concentration of a random sum, for which we
prove a concentration bound. These results reveal fundamental limitations in
scaling distributed optimization, even under the homogeneous assumption.

### 6. [Segmented Operations using Matrix Multiplications](http://arxiv.org/pdf/2506.23906v1)

Authors: Aleksandros Sobczyk, Giuseppe Sorrentino, Anastasios Zouzias

Specialized computational units that perform small matrix multiplications as
primitive operations are typically present in modern accelerators. However,
these units are often underutilized for many fundamental operations besides
dense matrix multiplications. The analysis of algorithms for such architectures
is currently stagnated due to the lack of a rigorous theoretical model of
computation that captures their characteristics. In this work, we propose
MMV-RAM, a computational model tailored to matrix multiplication accelerators.
MMV-RAM judiciously extends the Vector-RAM model with an additional processing
unit that multiplies two matrices of sizes $n\times s$ and $s\times s$ in a
single parallel step, where $s$ is a model parameter. We provide a detailed
theoretical analysis of the model, and carefully balance the computational
power between the matrix and vector units, guided by the circuit complexity
lower bound that parity is not in AC[0].
  In MMV-RAM, we study algorithms for segmented scan and sum, two fundamental
parallel primitives. We propose a segmented scan algorithm that uses matrix
multiplications to perform speculative block-scan computations, which runs in
$O(\log_s(n))$ steps. In contrast, we show that any algorithm that uses only
the vector unit of MMV-RAM requires
$\Omega\left(\frac{\log_2(n)}{\log_2\log_2(n)}\right)$ steps. We further apply
these techniques to obtain similar theoretical speedups for element-wise vector
multiplication and matrix multiplication. Beyond the worst-case complexity
analysis, we propose algorithms for segmented operations that could lead to
highly efficient and pragmatic implementations. For example, we observe that
segmented sum is a combination of three elementary parallel primitives: scan,
compress, and vector differentiation. As a case study, we implement...

### 7. [QPART: Adaptive Model Quantization and Dynamic Workload Balancing for Accuracy-aware Edge Inference](http://arxiv.org/pdf/2506.23934v1)

Authors: Xiangchen Li, Saeid Ghafouri, Bo Ji, Hans Vandierendonck, Deepu John, Dimitrios S. Nikolopoulos

As machine learning inferences increasingly move to edge devices, adapting to
diverse computational capabilities, hardware, and memory constraints becomes
more critical. Instead of relying on a pre-trained model fixed for all future
inference queries across diverse edge devices, we argue that planning an
inference pattern with a request-specific model tailored to the device's
computational capacity, accuracy requirements, and time constraints is more
cost-efficient and robust to diverse scenarios. To this end, we propose an
accuracy-aware and workload-balanced inference system that integrates joint
model quantization and inference partitioning. In this approach, the server
dynamically responds to inference queries by sending a quantized model and
adaptively sharing the inference workload with the device. Meanwhile, the
device's computational power, channel capacity, and accuracy requirements are
considered when deciding.
  Furthermore, we introduce a new optimization framework for the inference
system, incorporating joint model quantization and partitioning. Our approach
optimizes layer-wise quantization bit width and partition points to minimize
time consumption and cost while accounting for varying accuracy requirements of
tasks through an accuracy degradation metric in our optimization model. To our
knowledge, this work represents the first exploration of optimizing
quantization layer-wise bit-width in the inference serving system, by
introducing theoretical measurement of accuracy degradation. Simulation results
demonstrate a substantial reduction in overall time and power consumption, with
computation payloads decreasing by over 80% and accuracy degradation kept below
1%.

### Discrete Mathematics

### 1. [Linear Layouts of Graphs with Priority Queues](http://arxiv.org/pdf/2506.23943v1)

Authors: Emilio Di Giacomo, Walter Didimo, Henry Förster, Torsten Ueckerdt, Johannes Zink

A linear layout of a graph consists of a linear ordering of its vertices and
a partition of its edges into pages such that the edges assigned to the same
page obey some constraint. The two most prominent and widely studied types of
linear layouts are stack and queue layouts, in which any two edges assigned to
the same page are forbidden to cross and nest, respectively. The names of these
two layouts derive from the fact that, when parsing the graph according to the
linear vertex ordering, the edges in a single page can be stored using a single
stack or queue, respectively. Recently, the concepts of stack and queue layouts
have been extended by using a double-ended queue or a restricted-input queue
for storing the edges of a page. We extend this line of study to edge-weighted
graphs by introducing priority queue layouts, that is, the edges on each page
are stored in a priority queue whose keys are the edge weights. First, we show
that there are edge-weighted graphs that require a linear number of priority
queues. Second, we characterize the graphs that admit a priority queue layout
with a single queue, regardless of the edge-weight function, and we provide an
efficient recognition algorithm. Third, we show that the number of priority
queues required independently of the edge-weight function is bounded by the
pathwidth of the graph, but can be arbitrarily large already for graphs of
treewidth two. Finally, we prove that determining the minimum number of
priority queues is NP-complete if the linear ordering of the vertices is fixed.

### 2. [Simple Approximations for General Spanner Problems](http://arxiv.org/pdf/2506.23638v1)

Authors: Fritz Bökler, Markus Chimani, Henning Jasper

Consider a graph with n nodes and m edges, independent edge weights and
lengths, and arbitrary distance demands for node pairs. The spanner problem
asks for a minimum-weight subgraph that satisfies these demands via
sufficiently short paths w.r.t. the edge lengths. For multiplicative
alpha-spanners (where demands equal alpha times the original distances) and
assuming that each edge's weight equals its length, the simple Greedy heuristic
by Alth\"ofer et al. (1993) is known to yield strong solutions, both in theory
and practice. To obtain guarantees in more general settings, recent
approximations typically abandon this simplicity and practicality. Still, so
far, there is no known non-trivial approximation algorithm for the spanner
problem in its most general form. We provide two surprisingly simple
approximations algorithms. In general, our Adapted Greedy achieves the first
unconditional approximation ratio of m, which is non-trivial due to the
independence of weights and lengths. Crucially, it maintains all size and
weight guarantees Greedy is known for, i.e., in the aforementioned
multiplicative alpha-spanner scenario and even for additive +beta-spanners.
Further, it generalizes some of these size guarantees to derive new weight
guarantees. Our second approach, Randomized Rounding, establishes a graph
transformation that allows a simple rounding scheme over a standard
multicommodity flow LP. It yields an O(n log n)-approximation, assuming integer
lengths and polynomially bounded distance demands. The only other known
approximation guarantee in this general setting requires several complex
subalgorithms and analyses, yet we match it up to a factor of O(n^{1/5-eps})
using standard tools. Further, on bounded-degree graphs, we yield the first
O(log n) approximation ratio for constant-bounded distance demands (beyond
multiplicative 2-spanners in unit-length graphs).

### 3. [A Graph Width Perspective on Partially Ordered Hamiltonian Paths and Cycles I: Treewidth, Pathwidth, and Grid Graphs](http://arxiv.org/pdf/2506.23790v1)

Authors: Jesse Beisegel, Katharina Klost, Kristin Knorr, Fabienne Ratajczak, Robert Scheffler

We consider the problem of finding a Hamiltonian path or a Hamiltonian cycle
with precedence constraints in the form of a partial order on the vertex set.
We show that the path problem is $\mathsf{NP}$-complete for graphs of pathwidth
4 while the cycle problem is $\mathsf{NP}$-complete on graphs of pathwidth 5.
We complement these results by giving polynomial-time algorithms for graphs of
pathwidth 3 and treewidth 2 for Hamiltonian paths as well as pathwidth 4 and
treewidth 3 for Hamiltonian cycles. Furthermore, we study the complexity of the
path and cycle problems on rectangular grid graphs of bounded height. For
these, we show that the path and cycle problems are $\mathsf{NP}$-complete when
the height of the grid is greater or equal to 7 and 9, respectively. In the
variant where we look for minimum edge-weighted Hamiltonian paths and cycles,
the problems are $\mathsf{NP}$-hard for heights 5 and 6, respectively.

### 4. [Factorization norms and an inverse theorem for MaxCut](http://arxiv.org/pdf/2506.23989v1)

Authors: Igor Balla, Lianna Hambardzumyan, István Tomon

We prove that Boolean matrices with bounded $\gamma_2$-norm or bounded
normalized trace norm must contain a linear-sized all-ones or all-zeros
submatrix, verifying a conjecture of Hambardzumyan, Hatami, and Hatami. We also
present further structural results about Boolean matrices of bounded
$\gamma_2$-norm and discuss applications in communication complexity, operator
theory, spectral graph theory, and extremal combinatorics.
  As a key application, we establish an inverse theorem for MaxCut. A
celebrated result of Edwards states that every graph $G$ with $m$ edges has a
cut of size at least $\frac{m}{2}+\frac{\sqrt{8m+1}-1}{8}$, with equality
achieved by complete graphs with an odd number of vertices. To contrast this,
we prove that if the MaxCut of $G$ is at most $\frac{m}{2}+O(\sqrt{m})$, then
$G$ must contain a clique of size $\Omega(\sqrt{m})$.

### 5. [Translating between the representations of an acyclic convex geometry of bounded degree](http://arxiv.org/pdf/2506.24052v1)

Authors: Oscar Defrain, Arthur Ohana, Simon Vilmin

We consider the problem of enumerating the irreducible closed sets of a
closure system given by an implicational base. In the context of Horn logic,
these correspond to Horn expressions and characteristic models, respectively.
To date, the complexity status of this problem is widely open, and it is
further known to generalize the notorious hypergraph dualization problem, even
in the context of acyclic convex geometries, i.e., closure systems admitting an
acyclic implicational base. This paper studies this later class with a focus on
the degree, which corresponds to the maximal number of implications in which an
element occurs. We show that the problem is tractable for bounded values of
this parameter, even when relaxed to the notions of premise- and
conclusion-degree. Our algorithms rely on structural properties of acyclic
convex geometries and involve various techniques from algorithmic enumeration
such as solution graph traversal, saturation techniques, and a sequential
approach leveraging from acyclicity. They are shown to perform in
incremental-polynomial time for the computation of irreducible closed sets, and
in polynomial time for the construction of an implicational base. Finally, we
argue that our running times cannot be improved to polynomial delay using the
standard framework of flashlight search.

### Data Structures and Algorithms

### 1. [Efficient Resource Allocation under Adversary Attacks: A Decomposition-Based Approach](http://arxiv.org/pdf/2506.23442v1)

Authors: Mansoor Davoodi, Setareh Maghsudi

We address the problem of allocating limited resources in a network under
persistent yet statistically unknown adversarial attacks. Each node in the
network may be degraded, but not fully disabled, depending on its available
defensive resources. The objective is twofold: to minimize total system damage
and to reduce cumulative resource allocation and transfer costs over time. We
model this challenge as a bi-objective optimization problem and propose a
decomposition-based solution that integrates chance-constrained programming
with network flow optimization. The framework separates the problem into two
interrelated subproblems: determining optimal node-level allocations across
time slots, and computing efficient inter-node resource transfers. We
theoretically prove the convergence of our method to the optimal solution that
would be obtained with full statistical knowledge of the adversary. Extensive
simulations demonstrate that our method efficiently learns the adversarial
patterns and achieves substantial gains in minimizing both damage and
operational costs, comparing three benchmark strategies under various parameter
settings.

### 2. [Towards practical FPRAS for #NFA: Exploiting the Power of Dependence](http://arxiv.org/pdf/2506.23561v1)

Authors: Kuldeep S. Meel, Alexis de Colnet

#NFA refers to the problem of counting the words of length $n$ accepted by a
non-deterministic finite automaton. #NFA is #P-hard, and although
fully-polynomial-time randomized approximation schemes (FPRAS) exist, they are
all impractical. The first FPRAS for #NFA had a running time of
$\tilde{O}(n^{17}m^{17}\varepsilon^{-14}\log(\delta^{-1}))$, where $m$ is the
number of states in the automaton, $\delta \in (0,1]$ is the confidence
parameter, and $\varepsilon > 0$ is the tolerance parameter (typically smaller
than $1$). The current best FPRAS achieved a significant improvement in the
time complexity relative to the first FPRAS and obtained FPRAS with time
complexity $\tilde{O}((n^{10}m^2 +
n^6m^3)\varepsilon^{-4}\log^2(\delta^{-1}))$. The complexity of the improved
FPRAS is still too intimidating to attempt any practical implementation.
  In this paper, we pursue the quest for practical FPRAS for #NFA by presenting
a new algorithm with a time complexity of
$O(n^2m^3\log(nm)\varepsilon^{-2}\log(\delta^{-1}))$. Observe that evaluating
whether a word of length $n$ is accepted by an NFA has a time complexity of
$O(nm^2)$. Therefore, our proposed FPRAS achieves sub-quadratic complexity with
respect to membership checks.

### 3. [A Refined Kernel for $d$-Hitting Set](http://arxiv.org/pdf/2506.24114v1)

Authors: Yuxi Liu, Mingyu Xiao

The $d$-Hitting Set problem is a fundamental problem in parameterized
complexity, which asks whether a given hypergraph contains a vertex subset $S$
of size at most $k$ that intersects every hyperedge (i.e., $S \cap e \neq
\emptyset$ for each hyperedge $e$). The best known kernel for this problem,
established by Abu-Khzam [1], has $(2d - 1)k^{d - 1} + k$ vertices. This result
has been very widely used in the literature as many problems can be modeled as
a special $d$-Hitting Set problem. In this work, we present a refinement to
this result by employing linear programming techniques to construct crown
decompositions in hypergraphs. This approach yields a slight but notable
improvement, reducing the size to $(2d - 2)k^{d - 1} + k$ vertices.

### 4. [Optimized methods for composite optimization: a reduction perspective](http://arxiv.org/pdf/2506.23756v1)

Authors: Jinho Bok, Jason M. Altschuler

Recent advances in convex optimization have leveraged computer-assisted
proofs to develop optimized first-order methods that improve over classical
algorithms. However, each optimized method is specially tailored for a
particular problem setting, and it is a well-documented challenge to extend
optimized methods to other settings due to their highly bespoke design and
analysis. We provide a general framework that derives optimized methods for
composite optimization directly from those for unconstrained smooth
optimization. The derived methods naturally extend the original methods,
generalizing how proximal gradient descent extends gradient descent. The key to
our result is certain algebraic identities that provide a unified and
straightforward way of extending convergence analyses from unconstrained to
composite settings. As concrete examples, we apply our framework to establish
(1) the phenomenon of stepsize acceleration for proximal gradient descent; (2)
a convergence rate for the proximal optimized gradient method which is faster
than FISTA; (3) a new method that improves the state-of-the-art rate for
minimizing gradient norm in the composite setting.

### 5. [Fantastic Flips and Where to Find Them: A General Framework for Parameterized Local Search on Partitioning Problem](http://arxiv.org/pdf/2506.24001v1)

Authors: Niels Grüttemeier, Nils Morawietz, Frank Sommer

Parameterized local search combines classic local search heuristics with the
paradigm of parameterized algorithmics. While most local search algorithms aim
to improve given solutions by performing one single operation on a given
solution, the parameterized approach aims to improve a solution by performing
$k$ simultaneous operations. Herein, $k$ is a parameter called search radius
for which the value can be chosen by a user. One major goal in the field of
parameterized local search is to outline the trade-off between the size of $k$
and the running time of the local search step. In this work, we introduce an
abstract framework that generalizes natural parameterized local search
approaches for a large class of partitioning problems: Given $n$ items that are
partitioned into $b$ bins and a target function that evaluates the quality of
the current partition, one asks whether it is possible to improve the solution
by removing up to $k$ items from their current bins and reassigning them to
other bins. Among others, our framework applies for the local search versions
of problems like Cluster Editing, Vector Bin Packing, and Nash Social Welfare.
Motivated by a real-world application of the problem Vector Bin Packing, we
introduce a parameter called number of types $\tau \le n$ and show that all
problems fitting in our framework can be solved in $\tau^k 2^{O(k)} |I|^{O(1)}$
time, where $|I|$ denotes the total input size. In case of Cluster Editing, the
parameter $\tau$ generalizes the well-known parameter neighborhood diversity of
the input graph. We complement this by showing that for all considered
problems, an algorithm significantly improving over our algorithm with running
time $\tau^k 2^{O(k)} |I|^{O(1)}$ would contradict the ETH. Additionally, we
show that even on very restricted instances, all considered problems are
W[1]-hard when parameterized by the search radius $k$ alone.

### 6. [Dominating Set Knapsack: Profit Optimization on Dominating Sets](http://arxiv.org/pdf/2506.24032v1)

Authors: Sipra Singh

In a large-scale network, we want to choose some influential nodes to make a
profit by paying some cost within a limited budget so that we do not have to
spend more budget on some nodes adjacent to the chosen nodes; our problem is
the graph-theoretic representation of it. We define our problem Dominating Set
Knapsack by attaching Knapsack Problem with Dominating Set on graphs. Each
vertex is associated with a cost factor and a profit amount. We aim to choose
some vertices within a fixed budget that gives maximum profit so that we do not
need to choose their 1-hop neighbors. We show that the Dominating Set Knapsack
problem is strongly NP-complete even when restricted to Bipartite graphs but
weakly NP-complete for Star graphs. We present a pseudo-polynomial time
algorithm for Trees in time $O(n\cdot min\{s^2, (\alpha(V))^2\})$. We show that
Dominating Set Knapsack is very unlikely to be Fixed Parameter Tractable(FPT)
by proving that it is in W[2]-hard parameterized by the solution size. We
developed FPT algorithms with running time $O(4^{tw}\cdot n^{O(1)} \cdot
min\{s^2,{\alpha(V)}^2\})$ and $O(2^{vck-1}\cdot n^{O(1)} \cdot
min\{s^2,{\alpha(V)}^2\})$, where $tw$ represents the treewidth of the given
graph, $vck$ is the solution size of the Vertex Cover Knapsack, $s$ is the size
of the knapsack and $\alpha(V)=\sum_{v\in V}\alpha(v)$.

### 7. [Simple Approximations for General Spanner Problems](http://arxiv.org/pdf/2506.23638v1)

Authors: Fritz Bökler, Markus Chimani, Henning Jasper

Consider a graph with n nodes and m edges, independent edge weights and
lengths, and arbitrary distance demands for node pairs. The spanner problem
asks for a minimum-weight subgraph that satisfies these demands via
sufficiently short paths w.r.t. the edge lengths. For multiplicative
alpha-spanners (where demands equal alpha times the original distances) and
assuming that each edge's weight equals its length, the simple Greedy heuristic
by Alth\"ofer et al. (1993) is known to yield strong solutions, both in theory
and practice. To obtain guarantees in more general settings, recent
approximations typically abandon this simplicity and practicality. Still, so
far, there is no known non-trivial approximation algorithm for the spanner
problem in its most general form. We provide two surprisingly simple
approximations algorithms. In general, our Adapted Greedy achieves the first
unconditional approximation ratio of m, which is non-trivial due to the
independence of weights and lengths. Crucially, it maintains all size and
weight guarantees Greedy is known for, i.e., in the aforementioned
multiplicative alpha-spanner scenario and even for additive +beta-spanners.
Further, it generalizes some of these size guarantees to derive new weight
guarantees. Our second approach, Randomized Rounding, establishes a graph
transformation that allows a simple rounding scheme over a standard
multicommodity flow LP. It yields an O(n log n)-approximation, assuming integer
lengths and polynomially bounded distance demands. The only other known
approximation guarantee in this general setting requires several complex
subalgorithms and analyses, yet we match it up to a factor of O(n^{1/5-eps})
using standard tools. Further, on bounded-degree graphs, we yield the first
O(log n) approximation ratio for constant-bounded distance demands (beyond
multiplicative 2-spanners in unit-length graphs).

### 8. [A Graph Width Perspective on Partially Ordered Hamiltonian Paths and Cycles I: Treewidth, Pathwidth, and Grid Graphs](http://arxiv.org/pdf/2506.23790v1)

Authors: Jesse Beisegel, Katharina Klost, Kristin Knorr, Fabienne Ratajczak, Robert Scheffler

We consider the problem of finding a Hamiltonian path or a Hamiltonian cycle
with precedence constraints in the form of a partial order on the vertex set.
We show that the path problem is $\mathsf{NP}$-complete for graphs of pathwidth
4 while the cycle problem is $\mathsf{NP}$-complete on graphs of pathwidth 5.
We complement these results by giving polynomial-time algorithms for graphs of
pathwidth 3 and treewidth 2 for Hamiltonian paths as well as pathwidth 4 and
treewidth 3 for Hamiltonian cycles. Furthermore, we study the complexity of the
path and cycle problems on rectangular grid graphs of bounded height. For
these, we show that the path and cycle problems are $\mathsf{NP}$-complete when
the height of the grid is greater or equal to 7 and 9, respectively. In the
variant where we look for minimum edge-weighted Hamiltonian paths and cycles,
the problems are $\mathsf{NP}$-hard for heights 5 and 6, respectively.

### 9. [Segmented Operations using Matrix Multiplications](http://arxiv.org/pdf/2506.23906v1)

Authors: Aleksandros Sobczyk, Giuseppe Sorrentino, Anastasios Zouzias

Specialized computational units that perform small matrix multiplications as
primitive operations are typically present in modern accelerators. However,
these units are often underutilized for many fundamental operations besides
dense matrix multiplications. The analysis of algorithms for such architectures
is currently stagnated due to the lack of a rigorous theoretical model of
computation that captures their characteristics. In this work, we propose
MMV-RAM, a computational model tailored to matrix multiplication accelerators.
MMV-RAM judiciously extends the Vector-RAM model with an additional processing
unit that multiplies two matrices of sizes $n\times s$ and $s\times s$ in a
single parallel step, where $s$ is a model parameter. We provide a detailed
theoretical analysis of the model, and carefully balance the computational
power between the matrix and vector units, guided by the circuit complexity
lower bound that parity is not in AC[0].
  In MMV-RAM, we study algorithms for segmented scan and sum, two fundamental
parallel primitives. We propose a segmented scan algorithm that uses matrix
multiplications to perform speculative block-scan computations, which runs in
$O(\log_s(n))$ steps. In contrast, we show that any algorithm that uses only
the vector unit of MMV-RAM requires
$\Omega\left(\frac{\log_2(n)}{\log_2\log_2(n)}\right)$ steps. We further apply
these techniques to obtain similar theoretical speedups for element-wise vector
multiplication and matrix multiplication. Beyond the worst-case complexity
analysis, we propose algorithms for segmented operations that could lead to
highly efficient and pragmatic implementations. For example, we observe that
segmented sum is a combination of three elementary parallel primitives: scan,
compress, and vector differentiation. As a case study, we implement...

### 10. [Translating between the representations of an acyclic convex geometry of bounded degree](http://arxiv.org/pdf/2506.24052v1)

Authors: Oscar Defrain, Arthur Ohana, Simon Vilmin

We consider the problem of enumerating the irreducible closed sets of a
closure system given by an implicational base. In the context of Horn logic,
these correspond to Horn expressions and characteristic models, respectively.
To date, the complexity status of this problem is widely open, and it is
further known to generalize the notorious hypergraph dualization problem, even
in the context of acyclic convex geometries, i.e., closure systems admitting an
acyclic implicational base. This paper studies this later class with a focus on
the degree, which corresponds to the maximal number of implications in which an
element occurs. We show that the problem is tractable for bounded values of
this parameter, even when relaxed to the notions of premise- and
conclusion-degree. Our algorithms rely on structural properties of acyclic
convex geometries and involve various techniques from algorithmic enumeration
such as solution graph traversal, saturation techniques, and a sequential
approach leveraging from acyclicity. They are shown to perform in
incremental-polynomial time for the computation of irreducible closed sets, and
in polynomial time for the construction of an implicational base. Finally, we
argue that our running times cannot be improved to polynomial delay using the
standard framework of flashlight search.

### Emerging Technologies

### 1. [Mutli-Level Autoencoder: Deep Learning Based Channel Coding and Modulation](http://arxiv.org/pdf/2506.23511v1)

Authors: Ahmad Abdel-Qader, Anas Chaaban, Mohamed S. Shehata

In this paper, we design a deep learning-based convolutional autoencoder for
channel coding and modulation. The objective is to develop an adaptive scheme
capable of operating at various signal-to-noise ratios (SNR)s without the need
for re-training. Additionally, the proposed framework allows validation by
testing all possible codes in the codebook, as opposed to previous AI-based
encoder/decoder frameworks which relied on testing only a small subset of the
available codes. This limitation in earlier methods often led to unreliable
conclusions when generalized to larger codebooks. In contrast to previous
methods, our multi-level encoding and decoding approach splits the message into
blocks, where each encoder block processes a distinct group of $B$ bits. By
doing so, the proposed scheme can exhaustively test $2^{B}$ possible codewords
for each encoder/decoder level, constituting a layer of the overall scheme. The
proposed model was compared to classical polar codes and TurboAE-MOD schemes,
showing improved reliability with achieving comparable, or even superior
results in some settings. Notably, the architecture can adapt to different SNRs
by selectively removing one of the encoder/decoder layers without re-training,
thus demonstrating flexibility and efficiency in practical wireless
communication scenarios.

### 2. [Harnessing AI Agents to Advance Research on Refugee Child Mental Health](http://arxiv.org/pdf/2506.23992v1)

Authors: Aditya Shrivastava, Komal Gupta, Shraddha Arora

The international refugee crisis deepens, exposing millions of dis placed
children to extreme psychological trauma. This research suggests a com pact,
AI-based framework for processing unstructured refugee health data and
distilling knowledge on child mental health. We compare two Retrieval-Aug
mented Generation (RAG) pipelines, Zephyr-7B-beta and DeepSeek R1-7B, to
determine how well they process challenging humanitarian datasets while avoid
ing hallucination hazards. By combining cutting-edge AI methods with migration
research and child psychology, this study presents a scalable strategy to
assist policymakers, mental health practitioners, and humanitarian agencies to
better assist displaced children and recognize their mental wellbeing. In
total, both the models worked properly but significantly Deepseek R1 is
superior to Zephyr with an accuracy of answer relevance 0.91

### 3. [Comparative Studies: Cloud-Enabled Adaptive Learning System for Scalable Education in Sub-Saharan](http://arxiv.org/pdf/2506.23851v1)

Authors: Israel Fianyi, Soonja Yeom, Ju-Hyun Shin

The integration of cloud computing in education can revolutionise learning in
advanced (Australia & South Korea) and middle-income (Ghana & Nigeria)
countries, while offering scalable, cost-effective and equitable access to
adaptive learning systems. This paper explores how cloud computing and adaptive
learning technologies are deployed across different socio-economic and
infrastructure contexts. The study identifies enabling factors and systematic
challenges, providing insights into how cloud-based education can be tailored
to bridge the digital and educational divide globally.

### 4. [Green Metrics Tool: Measuring for fun and profit](http://arxiv.org/pdf/2506.23967v1)

Authors: Geerd-Dietger Hoffmann, Verena Majuntke

The environmental impact of software is gaining increasing attention as the
demand for computational resources continues to rise. In order to optimize
software resource consumption and reduce carbon emissions, measuring and
evaluating software is a first essential step. In this paper we discuss what
metrics are important for fact base decision making. We introduce the Green
Metrics Tool (GMT), a novel framework for accurately measuring the resource
consumption of software. The tool provides a containerized, controlled, and
reproducible life cycle-based approach, assessing the resource use of software
during key phases. Finally, we discuss GMT features like visualization,
comparability and rule- and LLM-based optimisations highlighting its potential
to guide developers and researchers in reducing the environmental impact of
their software.

### 5. [Spatial QUBO: Convolutional Formulation of Large-Scale Binary Optimization with Dense Interactions](http://arxiv.org/pdf/2506.24008v1)

Authors: Hiroshi Yamashita, Hideyuki Suzuki

The spatial photonic Ising machine (SPIM) is a promising optical hardware
solver for large-scale combinatorial optimization problems with dense
interactions. As the SPIM can represent Ising problems with rank-one coupling
matrices, multiplexed versions have been proposed to enhance the applicability
to higher-rank interactions. However, the multiplexing cost reduces the
implementation efficiency, and even without multiplexing, the SPIM is known to
represent coupling matrices beyond rank-one. In this paper, to clarify the
intrinsic representation power of the original SPIM, we propose spatial QUBO
(spQUBO), a formulation of Ising problems with spatially convolutional
structures. We prove that any spQUBO reduces to a two-dimensional spQUBO, with
the convolutional structure preserved, and that any two-dimensional spQUBO can
be efficiently implemented on the SPIM without multiplexing. We further
demonstrate its practical applicability to distance-based combinatorial
optimization, such as placement problems and clustering problems. These results
advance our understanding of the class of optimization problems where SPIMs
exhibit superior efficiency and scalability. Furthermore, spQUBO's efficiency
is not limited to the SPIM architecture; we show that its convolutional
structure allows efficient computation using Fast Fourier Transforms (FFT).

### 6. [Towards the "Digital Me": A vision of authentic Conversational Agents powered by personal Human Digital Twins](http://arxiv.org/pdf/2506.23826v1)

Authors: Lluís C. Coll, Martin W. Lauer-Schmaltz, Philip Cash, John P. Hansen, Anja Maier

Human Digital Twins (HDTs) have traditionally been conceptualized as
data-driven models designed to support decision-making across various domains.
However, recent advancements in conversational AI open new possibilities for
HDTs to function as authentic, interactive digital counterparts of individuals.
This paper introduces a novel HDT system architecture that integrates large
language models with dynamically updated personal data, enabling it to mirror
an individual's conversational style, memories, and behaviors. To achieve this,
our approach implements context-aware memory retrieval, neural
plasticity-inspired consolidation, and adaptive learning mechanisms, creating a
more natural and evolving digital persona. The resulting system does not only
replicate an individual's unique conversational style depending on who they are
speaking with, but also enriches responses with dynamically captured personal
experiences, opinions, and memories. While this marks a significant step toward
developing authentic virtual counterparts, it also raises critical ethical
concerns regarding privacy, accountability, and the long-term implications of
persistent digital identities. This study contributes to the field of HDTs by
describing our novel system architecture, demonstrating its capabilities, and
discussing future directions and emerging challenges to ensure the responsible
and ethical development of HDTs.

### Formal Languages and Automata Theory

### 1. [Reachability in symmetric VASS](http://arxiv.org/pdf/2506.23578v1)

Authors: Łukasz Kamiński, Sławomir Lasota

We investigate the reachability problem in symmetric vector addition systems
with states (VASS), where transitions are invariant under a group of
permutations of coordinates. One extremal case, the trivial groups, yields
general VASS. In another extremal case, the symmetric groups, we show that the
reachability problem can be solved in PSPACE, regardless of the dimension of
input VASS (to be contrasted with Ackermannian complexity in general VASS). We
also consider other groups, in particular alternating and cyclic ones.
Furthermore, motivated by the open status of the reachability problem in data
VASS, we estimate the gain in complexity when the group arises as a combination
of the trivial and symmetric groups.

### Graphics

### 1. [Synthetically Expressive: Evaluating gesture and voice for emotion and empathy in VR and 2D scenarios](http://arxiv.org/pdf/2506.23777v1)

Authors: Haoyang Du, Kiran Chhatre, Christopher Peters, Brian Keegan, Rachel McDonnell, Cathy Ennis

The creation of virtual humans increasingly leverages automated synthesis of
speech and gestures, enabling expressive, adaptable agents that effectively
engage users. However, the independent development of voice and gesture
generation technologies, alongside the growing popularity of virtual reality
(VR), presents significant questions about the integration of these signals and
their ability to convey emotional detail in immersive environments. In this
paper, we evaluate the influence of real and synthetic gestures and speech,
alongside varying levels of immersion (VR vs. 2D displays) and emotional
contexts (positive, neutral, negative) on user perceptions. We investigate how
immersion affects the perceived match between gestures and speech and the
impact on key aspects of user experience, including emotional and empathetic
responses and the sense of co-presence. Our findings indicate that while VR
enhances the perception of natural gesture-voice pairings, it does not
similarly improve synthetic ones - amplifying the perceptual gap between them.
These results highlight the need to reassess gesture appropriateness and refine
AI-driven synthesis for immersive environments. See video:
https://youtu.be/WMfjIB1X-dc

### 2. [HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity](http://arxiv.org/pdf/2506.23854v1)

Authors: Yida Wang, Xueyang Zhang, Kun Zhan, Peng Jia, Xianpeng Lang

Neural surface reconstruction faces persistent challenges in reconciling
geometric fidelity with photometric consistency under complex scene conditions.
We present HiNeuS, a unified framework that holistically addresses three core
limitations in existing approaches: multi-view radiance inconsistency, missing
keypoints in textureless regions, and structural degradation from over-enforced
Eikonal constraints during joint optimization. To resolve these issues through
a unified pipeline, we introduce: 1) Differential visibility verification
through SDF-guided ray tracing, resolving reflection ambiguities via continuous
occlusion modeling; 2) Planar-conformal regularization via ray-aligned geometry
patches that enforce local surface coherence while preserving sharp edges
through adaptive appearance weighting; and 3) Physically-grounded Eikonal
relaxation that dynamically modulates geometric constraints based on local
radiance gradients, enabling detail preservation without sacrificing global
regularity. Unlike prior methods that handle these aspects through sequential
optimizations or isolated modules, our approach achieves cohesive integration
where appearance-geometry constraints evolve synergistically throughout
training. Comprehensive evaluations across synthetic and real-world datasets
demonstrate state-of-the-art performance, including a 21.4% reduction in
Chamfer distance over reflection-aware baselines and 2.32 dB PSNR improvement
against neural rendering counterparts. Qualitative analyses reveal superior
capability in recovering specular instruments, urban layouts with
centimeter-scale infrastructure, and low-textured surfaces without local patch
collapse. The method's generalizability is further validated through successful
application to inverse rendering tasks, including material decomposition and
view-consistent relighting.

### 3. [GaVS: 3D-Grounded Video Stabilization via Temporally-Consistent Local Reconstruction and Rendering](http://arxiv.org/pdf/2506.23957v1)

Authors: Zinuo You, Stamatios Georgoulis, Anpei Chen, Siyu Tang, Dengxin Dai

Video stabilization is pivotal for video processing, as it removes unwanted
shakiness while preserving the original user motion intent. Existing
approaches, depending on the domain they operate, suffer from several issues
(e.g. geometric distortions, excessive cropping, poor generalization) that
degrade the user experience. To address these issues, we introduce
\textbf{GaVS}, a novel 3D-grounded approach that reformulates video
stabilization as a temporally-consistent `local reconstruction and rendering'
paradigm. Given 3D camera poses, we augment a reconstruction model to predict
Gaussian Splatting primitives, and finetune it at test-time, with multi-view
dynamics-aware photometric supervision and cross-frame regularization, to
produce temporally-consistent local reconstructions. The model are then used to
render each stabilized frame. We utilize a scene extrapolation module to avoid
frame cropping. Our method is evaluated on a repurposed dataset, instilled with
3D-grounded information, covering samples with diverse camera motions and scene
dynamics. Quantitatively, our method is competitive with or superior to
state-of-the-art 2D and 2.5D approaches in terms of conventional task metrics
and new geometry consistency. Qualitatively, our method produces noticeably
better results compared to alternatives, validated by the user study.

### 4. [Navigating with Annealing Guidance Scale in Diffusion Space](http://arxiv.org/pdf/2506.24108v1)

Authors: Shai Yehezkel, Omer Dahary, Andrey Voynov, Daniel Cohen-Or

Denoising diffusion models excel at generating high-quality images
conditioned on text prompts, yet their effectiveness heavily relies on careful
guidance during the sampling process. Classifier-Free Guidance (CFG) provides a
widely used mechanism for steering generation by setting the guidance scale,
which balances image quality and prompt alignment. However, the choice of the
guidance scale has a critical impact on the convergence toward a visually
appealing and prompt-adherent image. In this work, we propose an annealing
guidance scheduler which dynamically adjusts the guidance scale over time based
on the conditional noisy signal. By learning a scheduling policy, our method
addresses the temperamental behavior of CFG. Empirical results demonstrate that
our guidance scheduler significantly enhances image quality and alignment with
the text prompt, advancing the performance of text-to-image generation.
Notably, our novel scheduler requires no additional activations or memory
consumption, and can seamlessly replace the common classifier-free guidance,
offering an improved trade-off between prompt alignment and quality.

### Computer Science and Game Theory

### 1. [Interdependent Bilateral Trade: Information vs Approximation](http://arxiv.org/pdf/2506.23896v1)

Authors: Shahar Dobzinski, Alon Eden, Kira Goldner, Ariel Shaulker, Thodoris Tsilivis

Welfare maximization in bilateral trade has been extensively studied in
recent years. Previous literature obtained incentive-compatible approximation
mechanisms only for the private values case. In this paper, we study welfare
maximization in bilateral trade with interdependent values. Designing
mechanisms for interdependent settings is much more challenging because the
values of the players depend on the private information of the others,
requiring complex belief updates and strategic inference. We propose to
classify information structures by quantifying the influence that a player's
private signal has on their own valuation. We then paint a picture of where
approximations are possible and impossible based on these information
structures. Finally, we also study the possible approximation ratios for a
natural family of information structures.

### 2. [Quickest Detection of Adversarial Attacks Against Correlated Equilibria](http://arxiv.org/pdf/2506.24040v1)

Authors: Kiarash Kazari, Aris Kanellopoulos, György Dán

We consider correlated equilibria in strategic games in an adversarial
environment, where an adversary can compromise the public signal used by the
players for choosing their strategies, while players aim at detecting a
potential attack as soon as possible to avoid loss of utility. We model the
interaction between the adversary and the players as a zero-sum game and we
derive the maxmin strategies for both the defender and the attacker using the
framework of quickest change detection. We define a class of adversarial
strategies that achieve the optimal trade-off between attack impact and attack
detectability and show that a generalized CUSUM scheme is asymptotically
optimal for the detection of the attacks. Our numerical results on the
Sioux-Falls benchmark traffic routing game show that the proposed detection
scheme can effectively limit the utility loss by a potential adversary.

### 3. [Marker Gene Method : Identifying Stable Solutions in a Dynamic Environment](http://arxiv.org/pdf/2506.23734v1)

Authors: Hao Shi, Xi Li, Fangfang Xie

Competitive Co-evolutionary Algorithms (CCEAs) are often hampered by complex
dynamics like intransitivity and the Red Queen effect, leading to unstable
convergence. To counter these challenges, this paper introduces the Marker Gene
Method (MGM), a framework that establishes stability by using a 'marker gene'
as a dynamic benchmark and an adaptive weighting mechanism to balance
exploration and exploitation. We provide rigorous mathematical proofs
demonstrating that MGM creates strong attractors near Nash Equilibria within
the Strictly Competitive Game framework. Empirically, MGM demonstrates its
efficacy across a spectrum of challenges: it stabilizes the canonical
Rock-Paper-Scissors game, significantly improves the performance of C-RMOEA/D
on ZDT benchmarks, and, when augmented with a Memory Pool (MP) extension, it
successfully tames the notoriously pathological Shapley Biased Game. This work
presents a theoretically sound and empirically validated framework that
substantially enhances the stability and robustness of CCEAs in complex
competitive environments.

### Human-Computer Interaction

### 1. [Accessible Data Access and Analysis by People who are Blind or Have Low Vision](http://arxiv.org/pdf/2506.23443v1)

Authors: Samuel Reinders, Munazza Zaib, Matthew Butler, Bongshin Lee, Ingrid Zukerman, Lizhen Qu, Kim Marriott

Our work aims to develop new assistive technologies that enable blind or low
vision (BLV) people to explore and analyze data readily. At present, barriers
exist for BLV people to explore and analyze data, restricting access to
government, health and personal data, and limiting employment opportunities.
This work explores the co-design and development of an innovative system to
support data access, with a focus on the use of refreshable tactile displays
(RTDs) and conversational agents. The envisaged system will use a combination
of tactile graphics and speech to communicate with BLV users, and proactively
assist with data analysis tasks. As well as addressing significant equity gaps,
our work expects to produce innovations in assistive technology, multimodal
interfaces, dialogue systems, and natural language understanding and
generation.

### 2. [Reducing Motion Sickness in Passengers of Autonomous Personal Mobility Vehicles by Presenting a Driving Path](http://arxiv.org/pdf/2506.23457v1)

Authors: Yuya Ide, Hailong Liu, Takahiro Wada

Autonomous personal mobility vehicles (APMVs) are small mobility devices
designed for individual automated transportation in shared spaces. In such
environments, frequent pedestrian avoidance maneuvers may cause rapid steering
adjustments and passive postural responses from passengers, thereby increasing
the risk of motion sickness. This study investigated the effects of providing
path information on 16 passengers' head movement behavior and motion sickness
while riding an APMV. Through a controlled experiment comparing manual driving
(MD), autonomous driving without path information (AD w/o path), and autonomous
driving with path information (AD w/ path), we found that providing path cues
significantly reduced MISC scores and delayed the onset of motion sickness
symptoms. In addition, participants were more likely to proactively align their
head movements with the direction of vehicle rotation in both MD and AD w/ path
conditions. Although a small correlation was observed between the delay in yaw
rotation of the passenger's head relative to the vehicle and the occurrence of
motion sickness, the underlying physiological mechanism remains to be
elucidated.

### 3. [If You Had to Pitch Your Ideal Software -- Evaluating Large Language Models to Support User Scenario Writing for User Experience Experts and Laypersons](http://arxiv.org/pdf/2506.23694v1)

Authors: Patrick Stadler, Christopher Lazik, Christopher Katins, Thomas Kosch

The process of requirements analysis requires an understanding of the end
users of a system. Thus, expert stakeholders, such as User Experience (UX)
designers, usually create various descriptions containing information about the
users and their possible needs. In our paper, we investigate to what extent UX
novices are able to write such descriptions into user scenarios. We conducted a
user study with 60 participants consisting of 30 UX experts and 30 novices who
were asked to write a user scenario with or without the help of an
LLM-supported writing assistant. Our findings show that LLMs empower laypersons
to write reasonable user scenarios and provide first-hand insights for
requirements analysis that are comparable to UX experts in terms of structure
and clarity, while especially excelling at audience-orientation. We present our
qualitative and quantitative findings, including user scenario anatomies,
potential influences, and differences in the way participants approached the
task.

### 4. [Email as the Interface to Generative AI Models: Seamless Administrative Automation](http://arxiv.org/pdf/2506.23850v1)

Authors: Andres Navarro, Carlos de Quinto, José Alberto Hernández

This paper introduces a novel architectural framework that integrates Large
Language Models (LLMs) with email interfaces to automate administrative tasks,
specifically targeting accessibility barriers in enterprise environments. The
system connects email communication channels with Optical Character Recognition
(OCR) and intelligent automation, enabling non-technical administrative staff
to delegate complex form-filling and document processing tasks using familiar
email interfaces. By treating the email body as a natural language prompt and
attachments as contextual information, the workflow bridges the gap between
advanced AI capabilities and practical usability. Empirical evaluation shows
that the system can complete complex administrative forms in under 8 seconds of
automated processing, with human supervision reducing total staff time by a
factor of three to four compared to manual workflows. The top-performing LLM
accurately filled 16 out of 29 form fields and reduced the total cost per
processed form by 64% relative to manual completion. These findings demonstrate
that email-based LLM integration is a viable and cost-effective approach for
democratizing advanced automation in organizational settings, supporting
widespread adoption without requiring specialized technical knowledge or major
workflow changes. This aligns with broader trends in leveraging LLMs to enhance
accessibility and automate complex tasks for non-technical users, making
technology more inclusive and efficient.

### 5. [Access InContext: Futuring Accessible Prototyping Tools and Methods](http://arxiv.org/pdf/2506.24057v1)

Authors: Patricia Piedade, Peter A Hayton, Cynthia Bennett, Anna R L Carter, Clara Crivellaro, Alan Dix, Jess McGowan, Katta Spiel, Miriam Sturdee, Garreth W. Tigwell, Hugo Nicolau

The popularity of accessibility research has grown recently, improving
digital inclusion for people with disabilities. However, researchers, including
those who have disabilities, have attempted to include people with disabilities
in all aspects of design, and they have identified a myriad of practical
accessibility barriers posed by tools and methods leveraged by human-computer
interaction (HCI) researchers during prototyping. To build a more inclusive
technological landscape, we must question the effectiveness of existing
prototyping tools and methods, repurpose/retrofit existing resources, and build
new tools and methods to support the participation of both researchers and
people with disabilities within the prototyping design process of novel
technologies. This full-day workshop at CHI 2025 will provide a platform for
HCI researchers, designers, and practitioners to discuss barriers and
opportunities for creating accessible prototyping and promote hands-on ideation
and fabrication exercises aimed at futuring accessible prototyping.

### 6. [Bridging Service Design, Visualizations, and Visual Analytics in Healthcare Digital Twins: Challenges, Gaps, and Research Opportunities](http://arxiv.org/pdf/2506.24104v1)

Authors: Mariia Ershova, Graziano Blasilli

Digital twins (DT) are increasingly used in healthcare to model patients,
processes, and physiological systems. While recent solutions leverage
visualization, visual analytics, and user interaction, these systems rarely
incorporate structured service design methodologies. Bridging service design
with visual analytics and visualization can be valuable for the healthcare DT
community. This paper aims to introduce the service design discipline to
visualization researchers by framing this integration gap and suggesting
research directions to enhance the real-world applicability of DT solutions.

### 7. [Neuro-Informed Joint Learning Enhances Cognitive Workload Decoding in Portable BCIs](http://arxiv.org/pdf/2506.23458v1)

Authors: Xiaoxiao Yang, Chan Feng, Jiancheng Chen

Portable and wearable consumer-grade electroencephalography (EEG) devices,
like Muse headbands, offer unprecedented mobility for daily brain-computer
interface (BCI) applications, including cognitive load detection. However, the
exacerbated non-stationarity in portable EEG signals constrains data fidelity
and decoding accuracy, creating a fundamental trade-off between portability and
performance. To mitigate such limitation, we propose MuseCogNet (Muse-based
Cognitive Network), a unified joint learning framework integrating
self-supervised and supervised training paradigms. In particular, we introduce
an EEG-grounded self-supervised reconstruction loss based on average pooling to
capture robust neurophysiological patterns, while cross-entropy loss refines
task-specific cognitive discriminants. This joint learning framework resembles
the bottom-up and top-down attention in humans, enabling MuseCogNet to
significantly outperform state-of-the-art methods on a publicly available Muse
dataset and establish an implementable pathway for neurocognitive monitoring in
ecological settings.

### 8. [Immersive Technologies in Training and Healthcare: From Space Missions to Psychophysiological Research](http://arxiv.org/pdf/2506.23545v1)

Authors: Barbara Karpowicz, Maciej Grzeszczuk, Adam Kuzdraliński, Monika Kornacka, Aliaksandr Marozau, Wiktor Stawski, Pavlo Zinevych, Grzegorz Marcin Wójcik, Tomasz Kowalewski, Grzegorz Pochwatko, Wiesław Kopeć

Virtual, Augmented, and eXtended Reality (VR/AR/XR) technologies are
increasingly recognized for their applications in training, diagnostics, and
psychological research, particularly in high-risk and highly regulated
environments. In this panel we discuss how immersive systems enhance human
performance across multiple domains, including clinical psychology, space
exploration, and medical education. In psychological research and training, XR
can offer a controlled yet ecologically valid setting for measuring cognitive
and affective processes. In space exploration, we discuss the development of
VR-based astronaut training and diagnostic systems, allowing astronauts to
perform real-time health assessments. In medical education and rehabilitation,
we cover procedural training and patient engagement. From virtual surgical
simulations to gamified rehabilitation exercises, immersive environments
enhance both learning outcomes and treatment adherence.

### 9. [Interactive Reasoning: Visualizing and Controlling Chain-of-Thought Reasoning in Large Language Models](http://arxiv.org/pdf/2506.23678v1)

Authors: Rock Yuren Pang, K. J. Kevin Feng, Shangbin Feng, Chu Li, Weijia Shi, Yulia Tsvetkov, Jeffrey Heer, Katharina Reinecke

The output quality of large language models (LLMs) can be improved via
"reasoning": generating segments of chain-of-thought (CoT) content to further
condition the model prior to producing user-facing output. While these chains
contain valuable information, they are verbose and lack explicit organization,
making them tedious to review. Moreover, they lack opportunities for user
feedback, such as to remove unwanted considerations, add desired ones, or
clarify unclear assumptions. We introduce Interactive Reasoning, an interaction
design that visualizes chain-of-thought outputs as a hierarchy of topics and
enables user review and modification. We implement interactive reasoning in
Hippo, a prototype for AI-assisted decision making in the face of uncertain
trade-offs. In a user study with 16 participants, we find that interactive
reasoning in Hippo allows users to quickly identify and interrupt erroneous
generations, efficiently steer the model towards customized responses, and
better understand both model reasoning and model outputs. Our work contributes
to a new paradigm that incorporates user oversight into LLM reasoning
processes.

### 10. [Leveraging a Multi-Agent LLM-Based System to Educate Teachers in Hate Incidents Management](http://arxiv.org/pdf/2506.23774v1)

Authors: Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jarosław A. Chudziak

Computer-aided teacher training is a state-of-the-art method designed to
enhance teachers' professional skills effectively while minimising concerns
related to costs, time constraints, and geographical limitations. We
investigate the potential of large language models (LLMs) in teacher education,
using a case of teaching hate incidents management in schools. To this end, we
create a multi-agent LLM-based system that mimics realistic situations of hate,
using a combination of retrieval-augmented prompting and persona modelling. It
is designed to identify and analyse hate speech patterns, predict potential
escalation, and propose effective intervention strategies. By integrating
persona modelling with agentic LLMs, we create contextually diverse simulations
of hate incidents, mimicking real-life situations. The system allows teachers
to analyse and understand the dynamics of hate incidents in a safe and
controlled environment, providing valuable insights and practical knowledge to
manage such situations confidently in real life. Our pilot evaluation
demonstrates teachers' enhanced understanding of the nature of annotator
disagreements and the role of context in hate speech interpretation, leading to
the development of more informed and effective strategies for addressing hate
in classroom settings.

### Information Retrieval

### 1. [Act-With-Think: Chunk Auto-Regressive Modeling for Generative Recommendation](http://arxiv.org/pdf/2506.23643v1)

Authors: Yifan Wang, Weinan Gan, Longtao Xiao, Jieming Zhu, Heng Chang, Haozhao Wang, Rui Zhang, Zhenhua Dong, Ruiming Tang, Ruixuan Li

Generative recommendation (GR) typically encodes behavioral or semantic
aspects of item information into discrete tokens, leveraging the standard
autoregressive (AR) generation paradigm to make predictions. However, existing
methods tend to overlook their intrinsic relationship, that is, the semantic
usually provides some reasonable explainability "$\textbf{why}$" for the
behavior "$\textbf{what}$", which may constrain the full potential of GR. To
this end, we present Chunk AutoRegressive Modeling (CAR), a new generation
paradigm following the decision pattern that users usually think semantic
aspects of items (e.g. brand) and then take actions on target items (e.g.
purchase). Our CAR, for the $\textit{first time}$, incorporates semantics
(SIDs) and behavior (UID) into a single autoregressive transformer from an
``act-with-think'' dual perspective via chunk-level autoregression.
Specifically, CAR packs SIDs and UID into a conceptual chunk for item unified
representation, allowing each decoding step to make a holistic prediction.
Experiments show that our CAR significantly outperforms existing methods based
on traditional AR, improving Recall@5 by 7.93% to 22.30%. Furthermore, we
verify the scaling effect between model performance and SIDs bit number,
demonstrating that CAR preliminary emulates a kind of slow-thinking style
mechanism akin to the reasoning processes observed in large language models
(LLMs).

### 2. [KiseKloset: Comprehensive System For Outfit Retrieval, Recommendation, And Try-On](http://arxiv.org/pdf/2506.23471v1)

Authors: Thanh-Tung Phan-Nguyen, Khoi-Nguyen Nguyen-Ngoc, Tam V. Nguyen, Minh-Triet Tran, Trung-Nghia Le

The global fashion e-commerce industry has become integral to people's daily
lives, leveraging technological advancements to offer personalized shopping
experiences, primarily through recommendation systems that enhance customer
engagement through personalized suggestions. To improve customers' experience
in online shopping, we propose a novel comprehensive KiseKloset system for
outfit retrieval, recommendation, and try-on. We explore two approaches for
outfit retrieval: similar item retrieval and text feedback-guided item
retrieval. Notably, we introduce a novel transformer architecture designed to
recommend complementary items from diverse categories. Furthermore, we enhance
the overall performance of the search pipeline by integrating approximate
algorithms to optimize the search process. Additionally, addressing the crucial
needs of online shoppers, we employ a lightweight yet efficient virtual try-on
framework capable of real-time operation, memory efficiency, and maintaining
realistic outputs compared to its predecessors. This virtual try-on module
empowers users to visualize specific garments on themselves, enhancing the
customers' experience and reducing costs associated with damaged items for
retailers. We deployed our end-to-end system for online users to test and
provide feedback, enabling us to measure their satisfaction levels. The results
of our user study revealed that 84% of participants found our comprehensive
system highly useful, significantly improving their online shopping experience.

### 3. [Zero-Shot Contextual Embeddings via Offline Synthetic Corpus Generation](http://arxiv.org/pdf/2506.23662v1)

Authors: Philip Lippmann, Jie Yang

Context-aware embedding methods boost retrieval accuracy by conditioning on
corpus statistics (e.g., term co-occurrence and topical patterns) extracted
from neighboring documents. However, this context-aware approach requires
access to the target corpus or requires domain-specific finetuning, posing
practical barriers in privacy-sensitive or resource-constrained settings. We
present ZEST, a zero-shot contextual adaptation framework that replaces real
corpus access with a one-time offline synthesis of a compact proxy. Given only
a handful exemplar documents representative of the general target domain, we
use a multi-step hierarchical procedure to generate a synthetic context corpus
of several hundred documents that aims to emulate key domain-specific
distributions. At inference, the frozen context-aware encoder uses this proxy
corpus -- without any finetuning or target corpus access -- to produce
domain-adapted embeddings. Across the MTEB benchmark, ZEST's zero-shot
synthetic context adaptation using only five example documents performs within
0.5% of models leveraging full target corpus access -- demonstrating remarkable
efficacy without any retraining. ZEST thus provides a practical method for
deploying high-performance, adaptable embeddings in constrained environments.

### 4. [Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent](http://arxiv.org/pdf/2506.23485v1)

Authors: Haocheng Yu, Yaxiong Wu, Hao Wang, Wei Guo, Yong Liu, Yawen Li, Yuyang Ye, Junping Du, Enhong Chen

Interactive recommendation is a typical information-seeking task that allows
users to interactively express their needs through natural language and obtain
personalized recommendations. Large language model-powered (LLM-powered) agents
have become a new paradigm in interactive recommendations, effectively
capturing users' real-time needs and enhancing personalized experiences.
However, due to limited planning and generalization capabilities, existing
formulations of LLM-powered interactive recommender agents struggle to
effectively address diverse and complex user intents, such as intuitive,
unrefined, or occasionally ambiguous requests. To tackle this challenge, we
propose a novel thought-augmented interactive recommender agent system (TAIRA)
that addresses complex user intents through distilled thought patterns.
Specifically, TAIRA is designed as an LLM-powered multi-agent system featuring
a manager agent that orchestrates recommendation tasks by decomposing user
needs and planning subtasks, with its planning capacity strengthened through
Thought Pattern Distillation (TPD), a thought-augmentation method that extracts
high-level thoughts from the agent's and human experts' experiences. Moreover,
we designed a set of user simulation schemes to generate personalized queries
of different difficulties and evaluate the recommendations based on specific
datasets. Through comprehensive experiments conducted across multiple datasets,
TAIRA exhibits significantly enhanced performance compared to existing methods.
Notably, TAIRA shows a greater advantage on more challenging tasks while
generalizing effectively on novel tasks, further validating its superiority in
managing complex user intents within interactive recommendation systems. The
code is publicly available at:https://github.com/Alcein/TAIRA.

### 5. [Emergent musical properties of a transformer under contrastive self-supervised learning](http://arxiv.org/pdf/2506.23873v1)

Authors: Yuexuan Kong, Gabriel Meseguer-Brocal, Vincent Lostanlen, Mathieu Lagrange, Romain Hennequin

In music information retrieval (MIR), contrastive self-supervised learning
for general-purpose representation models is effective for global tasks such as
automatic tagging. However, for local tasks such as chord estimation, it is
widely assumed that contrastively trained general-purpose self-supervised
models are inadequate and that more sophisticated SSL is necessary; e.g.,
masked modeling. Our paper challenges this assumption by revealing the
potential of contrastive SSL paired with a transformer in local MIR tasks. We
consider a lightweight vision transformer with one-dimensional patches in the
time--frequency domain (ViT-1D) and train it with simple contrastive SSL
through normalized temperature-scaled cross-entropy loss (NT-Xent). Although
NT-Xent operates only over the class token, we observe that, potentially thanks
to weight sharing, informative musical properties emerge in ViT-1D's sequence
tokens. On global tasks, the temporal average of class and sequence tokens
offers a performance increase compared to the class token alone, showing useful
properties in the sequence tokens. On local tasks, sequence tokens perform
unexpectedly well, despite not being specifically trained for. Furthermore,
high-level musical features such as onsets emerge from layer-wise attention
maps and self-similarity matrices show different layers capture different
musical dimensions. Our paper does not focus on improving performance but
advances the musical interpretation of transformers and sheds light on some
overlooked abilities of contrastive SSL paired with transformers for sequence
modeling in MIR.

### 6. [Towards the "Digital Me": A vision of authentic Conversational Agents powered by personal Human Digital Twins](http://arxiv.org/pdf/2506.23826v1)

Authors: Lluís C. Coll, Martin W. Lauer-Schmaltz, Philip Cash, John P. Hansen, Anja Maier

Human Digital Twins (HDTs) have traditionally been conceptualized as
data-driven models designed to support decision-making across various domains.
However, recent advancements in conversational AI open new possibilities for
HDTs to function as authentic, interactive digital counterparts of individuals.
This paper introduces a novel HDT system architecture that integrates large
language models with dynamically updated personal data, enabling it to mirror
an individual's conversational style, memories, and behaviors. To achieve this,
our approach implements context-aware memory retrieval, neural
plasticity-inspired consolidation, and adaptive learning mechanisms, creating a
more natural and evolving digital persona. The resulting system does not only
replicate an individual's unique conversational style depending on who they are
speaking with, but also enriches responses with dynamically captured personal
experiences, opinions, and memories. While this marks a significant step toward
developing authentic virtual counterparts, it also raises critical ethical
concerns regarding privacy, accountability, and the long-term implications of
persistent digital identities. This study contributes to the field of HDTs by
describing our novel system architecture, demonstrating its capabilities, and
discussing future directions and emerging challenges to ensure the responsible
and ethical development of HDTs.

### Machine Learning

### 1. [Enhancing Insider Threat Detection Using User-Based Sequencing and Transformer Encoders](http://arxiv.org/pdf/2506.23446v1)

Authors: Mohamed Elbasheer, Adewale Akinfaderin

Insider threat detection presents unique challenges due to the authorized
status of malicious actors and the subtlety of anomalous behaviors. Existing
machine learning methods often treat user activity as isolated events, thereby
failing to leverage sequential dependencies in user behavior. In this study, we
propose a User-Based Sequencing (UBS) methodology, transforming the CERT
insider threat dataset into structured temporal sequences suitable for deep
sequential modeling. We deploy a Transformer Encoder architecture to model
benign user activity and employ its reconstruction errors as anomaly scores.
These scores are subsequently evaluated using three unsupervised outlier
detection algorithms: One-Class SVM (OCSVM), Local Outlier Factor (LOF), and
Isolation Forest (iForest). Across four rigorously designed test sets,
including combinations of multiple CERT dataset releases, our UBS-Transformer
pipeline consistently achieves state-of-the-art performance - notably 96.61%
accuracy, 99.43% recall, 96.38% F1-score, 95.00% AUROC, and exceptionally low
false negative (0.0057) and false positive (0.0571) rates. Comparative analyses
demonstrate that our approach substantially outperforms tabular and
conventional autoencoder baselines, underscoring the efficacy of sequential
user modeling and advanced anomaly detection in the insider threat domain.

### 2. [A unified framework on the universal approximation of transformer-type architectures](http://arxiv.org/pdf/2506.23551v1)

Authors: Jingpu Cheng, Qianxiao Li, Ting Lin, Zuowei Shen

We investigate the universal approximation property (UAP) of transformer-type
architectures, providing a unified theoretical framework that extends prior
results on residual networks to models incorporating attention mechanisms. Our
work identifies token distinguishability as a fundamental requirement for UAP
and introduces a general sufficient condition that applies to a broad class of
architectures. Leveraging an analyticity assumption on the attention layer, we
can significantly simplify the verification of this condition, providing a
non-constructive approach in establishing UAP for such architectures. We
demonstrate the applicability of our framework by proving UAP for transformers
with various attention mechanisms, including kernel-based and sparse attention
mechanisms. The corollaries of our results either generalize prior works or
establish UAP for architectures not previously covered. Furthermore, our
framework offers a principled foundation for designing novel transformer
architectures with inherent UAP guarantees, including those with specific
functional symmetries. We propose examples to illustrate these insights.

### 3. [Model-driven Stochastic Trace Clustering](http://arxiv.org/pdf/2506.23776v1)

Authors: Jari Peeperkorn, Johannes De Smedt, Jochen De Weerdt

Process discovery algorithms automatically extract process models from event
logs, but high variability often results in complex and hard-to-understand
models. To mitigate this issue, trace clustering techniques group process
executions into clusters, each represented by a simpler and more understandable
process model. Model-driven trace clustering improves on this by assigning
traces to clusters based on their conformity to cluster-specific process
models. However, most existing clustering techniques rely on either no process
model discovery, or non-stochastic models, neglecting the frequency or
probability of activities and transitions, thereby limiting their capability to
capture real-world execution dynamics. We propose a novel model-driven trace
clustering method that optimizes stochastic process models within each cluster.
Our approach uses entropic relevance, a stochastic conformance metric based on
directly-follows probabilities, to guide trace assignment. This allows
clustering decisions to consider both structural alignment with a cluster's
process model and the likelihood that a trace originates from a given
stochastic process model. The method is computationally efficient, scales
linearly with input size, and improves model interpretability by producing
clusters with clearer control-flow patterns. Extensive experiments on public
real-life datasets show that our method outperforms existing alternatives in
representing process behavior and reveals how clustering performance rankings
can shift when stochasticity is considered.

### 4. [KAIROS: Scalable Model-Agnostic Data Valuation](http://arxiv.org/pdf/2506.23799v1)

Authors: Jiongli Zhu, Parjanya Prajakta Prashant, Alex Cloninger, Babak Salimi

Training data increasingly shapes not only model accuracy but also regulatory
compliance and market valuation of AI assets. Yet existing valuation methods
remain inadequate: model-based techniques depend on a single fitted model and
inherit its biases, while algorithm-based approaches such as Data Shapley
require costly retrainings at web scale. Recent Wasserstein-based
model-agnostic methods rely on approximations that misrank examples relative to
their true leave-one-out (LOO) utility. We introduce KAIROS, a scalable,
model-agnostic valuation framework that assigns each example a distributional
influence score: its contribution to the Maximum Mean Discrepancy (MMD) between
the empirical training distribution and a clean reference set. Unlike
Wasserstein surrogates, our MMD-based influence admits a closed-form solution
that faithfully approximates the exact LOO ranking within $O(1/N^2)$ error,
requires no retraining, and naturally extends to conditional kernels for
unified label- and feature-error detection. Moreover, KAIROS supports efficient
online updates: when a new batch of size m arrives, all scores can be updated
in $O(mN)$ time, delivering up to 50x speedup without compromising ranking
quality. Empirical evaluations on noise, mislabeling, and poisoning benchmarks
show that KAIROS consistently outperforms state-of-the-art model-, Shapley-,
and Wasserstein-based baselines in both accuracy and runtime. We provide
rigorous theoretical guarantees, including symmetry for reproducible rankings
and density-separation for interpretable thresholds.

### 5. [Towards the Training of Deeper Predictive Coding Neural Networks](http://arxiv.org/pdf/2506.23800v1)

Authors: Chang Qi, Matteo Forasassi, Thomas Lukasiewicz, Tommaso Salvatori

Predictive coding networks trained with equilibrium propagation are neural
models that perform inference through an iterative energy minimization process.
Previous studies have demonstrated their effectiveness in shallow
architectures, but show significant performance degradation when depth exceeds
five to seven layers. In this work, we show that the reason behind this
degradation is due to exponentially imbalanced errors between layers during
weight updates, and predictions from the previous layer not being effective in
guiding updates in deeper layers. We address the first issue by introducing two
novel methods to optimize the latent variables that use precision-weighting to
re-balance the distribution of energy among layers during the `relaxation
phase', and the second issue by proposing a novel weight update mechanism that
reduces error accumulation in deeper layers. Empirically, we test our methods
on a large number of image classification tasks, resulting in large
improvements in test accuracy across networks with more than seven layers, with
performances comparable to those of backprop on similar models. These findings
suggest that a better understanding of the relaxation phase is important to
train models using equilibrium propagation at scale, and open new possibilities
for their application in complex tasks.

### 6. [Adaptive Out-of-Control Point Pattern Detection in Sequential Random Finite Set Observations](http://arxiv.org/pdf/2506.23802v1)

Authors: Konstantinos Bourazas, Savvas Papaioannou, Panayiotis Kolios

In this work we introduce a novel adaptive anomaly detection framework
specifically designed for monitoring sequential random finite set (RFS)
observations. Our approach effectively distinguishes between In-Control data
(normal) and Out-Of-Control data (anomalies) by detecting deviations from the
expected statistical behavior of the process. The primary contributions of this
study include the development of an innovative RFS-based framework that not
only learns the normal behavior of the data-generating process online but also
dynamically adapts to behavioral shifts to accurately identify abnormal point
patterns. To achieve this, we introduce a new class of RFS-based posterior
distributions, named Power Discounting Posteriors (PD), which facilitate
adaptation to systematic changes in data while enabling anomaly detection of
point pattern data through a novel predictive posterior density function. The
effectiveness of the proposed approach is demonstrated by extensive qualitative
and quantitative simulation experiments.

### 7. [EFPI: Elastic Formation and Position Identification in Football (Soccer) using Template Matching and Linear Assignment](http://arxiv.org/pdf/2506.23843v1)

Authors: Joris Bekkers

Understanding team formations and player positioning is crucial for tactical
analysis in football (soccer). This paper presents a flexible method for
formation recognition and player position assignment in football using
predefined static formation templates and cost minimization from spatiotemporal
tracking data, called EFPI. Our approach employs linear sum assignment to
optimally match players to positions within a set of template formations by
minimizing the total distance between actual player locations and template
positions, subsequently selecting the formation with the lowest assignment
cost. To improve accuracy, we scale actual player positions to match the
dimensions of these formation templates in both width and length. While the
method functions effectively on individual frames, it extends naturally to
larger game segments such as complete periods, possession sequences or specific
intervals (e.g. 10 second intervals, 5 minute intervals etc.). Additionally, we
incorporate an optional stability parameter that prevents unnecessary formation
changes when assignment costs differ only marginally between time segments.
EFPI is available as open-source code through the unravelsports Python package.

### 8. [When Plants Respond: Electrophysiology and Machine Learning for Green Monitoring Systems](http://arxiv.org/pdf/2506.23872v1)

Authors: Eduard Buss, Till Aust, Heiko Hamann

Living plants, while contributing to ecological balance and climate
regulation, also function as natural sensors capable of transmitting
information about their internal physiological states and surrounding
conditions. This rich source of data provides potential for applications in
environmental monitoring and precision agriculture. With integration into
biohybrid systems, we establish novel channels of physiological signal flow
between living plants and artificial devices. We equipped *Hedera helix* with a
plant-wearable device called PhytoNode to continuously record the plant's
electrophysiological activity. We deployed plants in an uncontrolled outdoor
environment to map electrophysiological patterns to environmental conditions.
Over five months, we collected data that we analyzed using state-of-the-art and
automated machine learning (AutoML). Our classification models achieve high
performance, reaching macro F1 scores of up to 95 percent in binary tasks.
AutoML approaches outperformed manual tuning, and selecting subsets of
statistical features further improved accuracy. Our biohybrid living system
monitors the electrophysiology of plants in harsh, real-world conditions. This
work advances scalable, self-sustaining, and plant-integrated living biohybrid
systems for sustainable environmental monitoring.

### 9. [Bridging the Gap with Retrieval-Augmented Generation: Making Prosthetic Device User Manuals Available in Marginalised Languages](http://arxiv.org/pdf/2506.23958v1)

Authors: Ikechukwu Ogbonna, Lesley Davidson, Soumya Banerjee, Abhishek Dasgupta, Laurence Kenney, Vikranth Harthikote Nagaraja

Millions of people in African countries face barriers to accessing healthcare
due to language and literacy gaps. This research tackles this challenge by
transforming complex medical documents -- in this case, prosthetic device user
manuals -- into accessible formats for underserved populations. This case study
in cross-cultural translation is particularly pertinent/relevant for
communities that receive donated prosthetic devices but may not receive the
accompanying user documentation. Or, if available online, may only be available
in formats (e.g., language and readability) that are inaccessible to local
populations (e.g., English-language, high resource settings/cultural context).
The approach is demonstrated using the widely spoken Pidgin dialect, but our
open-source framework has been designed to enable rapid and easy extension to
other languages/dialects. This work presents an AI-powered framework designed
to process and translate complex medical documents, e.g., user manuals for
prosthetic devices, into marginalised languages. The system enables users --
such as healthcare workers or patients -- to upload English-language medical
equipment manuals, pose questions in their native language, and receive
accurate, localised answers in real time. Technically, the system integrates a
Retrieval-Augmented Generation (RAG) pipeline for processing and semantic
understanding of the uploaded manuals. It then employs advanced Natural
Language Processing (NLP) models for generative question-answering and
multilingual translation. Beyond simple translation, it ensures accessibility
to device instructions, treatment protocols, and safety information, empowering
patients and clinicians to make informed healthcare decisions.

### 10. [UMA: A Family of Universal Models for Atoms](http://arxiv.org/pdf/2506.23971v1)

Authors: Brandon M. Wood, Misko Dzamba, Xiang Fu, Meng Gao, Muhammed Shuaibi, Luis Barroso-Luque, Kareem Abdelmaqsoud, Vahe Gharakhanyan, John R. Kitchin, Daniel S. Levine, Kyle Michel, Anuroop Sriram, Taco Cohen, Abhishek Das, Ammar Rizvi, Sushree Jagriti Sahoo, Zachary W. Ulissi, C. Lawrence Zitnick

The ability to quickly and accurately compute properties from atomic
simulations is critical for advancing a large number of applications in
chemistry and materials science including drug discovery, energy storage, and
semiconductor manufacturing. To address this need, Meta FAIR presents a family
of Universal Models for Atoms (UMA), designed to push the frontier of speed,
accuracy, and generalization. UMA models are trained on half a billion unique
3D atomic structures (the largest training runs to date) by compiling data
across multiple chemical domains, e.g. molecules, materials, and catalysts. We
develop empirical scaling laws to help understand how to increase model
capacity alongside dataset size to achieve the best accuracy. The UMA small and
medium models utilize a novel architectural design we refer to as mixture of
linear experts that enables increasing model capacity without sacrificing
speed. For example, UMA-medium has 1.4B parameters but only ~50M active
parameters per atomic structure. We evaluate UMA models on a diverse set of
applications across multiple domains and find that, remarkably, a single model
without any fine-tuning can perform similarly or better than specialized
models. We are releasing the UMA code, weights, and associated data to
accelerate computational workflows and enable the community to continue to
build increasingly capable AI models.

### Neural and Evolutionary Computing

### 1. [More Efficient Real-Valued Gray-Box Optimization through Incremental Distribution Estimation in RV-GOMEA](http://arxiv.org/pdf/2506.23738v1)

Authors: Renzo J. Scholman, Tanja Alderliesten, Peter A. N. Bosman

The Gene-pool Optimal Mixing EA (GOMEA) family of EAs offers a specific means
to exploit problem-specific knowledge through linkage learning, i.e.,
inter-variable dependency detection, expressed using subsets of variables, that
should undergo joint variation. Such knowledge can be exploited if faster
fitness evaluations are possible when only a few variables are changed in a
solution, enabling large speed-ups. The recent-most version of Real-Valued
GOMEA (RV-GOMEA) can learn a conditional linkage model during optimization
using fitness-based linkage learning, enabling fine-grained dependency
exploitation in learning and sampling a Gaussian distribution. However, while
the most efficient Gaussian-based EAs, like NES and CMA-ES, employ incremental
learning of the Gaussian distribution rather than performing full re-estimation
every generation, the recent-most RV-GOMEA version does not employ such
incremental learning. In this paper, we therefore study whether incremental
distribution estimation can lead to efficiency enhancements of RV-GOMEA. We
consider various benchmark problems with varying degrees of overlapping
dependencies. We find that, compared to RV-GOMEA and VKD-CMA-ES, the required
number of evaluations to reach high-quality solutions can be reduced by a
factor of up to 1.5 if population sizes are tuned problem-specifically, while a
reduction by a factor of 2-3 can be achieved with generic population-sizing
guidelines.

### 2. [Unsupervised Sparse Coding-based Spiking Neural Network for Real-time Spike Sorting](http://arxiv.org/pdf/2506.24041v1)

Authors: Alexis Melot, Sean U. N. Wood, Yannick Coffinier, Pierre Yger, Fabien Alibart

Spike sorting is a crucial step in decoding multichannel extracellular neural
signals, enabling the identification of individual neuronal activity. A key
challenge in brain-machine interfaces (BMIs) is achieving real-time, low-power
spike sorting at the edge while keeping high neural decoding performance. This
study introduces the Neuromorphic Sparse Sorter (NSS), a compact two-layer
spiking neural network optimized for efficient spike sorting. NSS leverages the
Locally Competitive Algorithm (LCA) for sparse coding to extract relevant
features from noisy events with reduced computational demands. NSS learns to
sort detected spike waveforms in an online fashion and operates entirely
unsupervised. To exploit multi-bit spike coding capabilities of neuromorphic
platforms like Intel's Loihi 2, a custom neuron model was implemented, enabling
flexible power-performance trade-offs via adjustable spike bit-widths.
Evaluations on simulated and real-world tetrode signals with biological drift
showed NSS outperformed established pipelines such as WaveClus3 and PCA+KMeans.
With 2-bit graded spikes, NSS on Loihi 2 outperformed NSS implemented with
leaky integrate-and-fire neuron and achieved an F1-score of 77% (+10%
improvement) while consuming 8.6mW (+1.65mW) when tested on a drifting
recording, with a computational processing time of 0.25ms (+60 us) per
inference.

### 3. [Neural Langevin Machine: a local asymmetric learning rule can be creative](http://arxiv.org/pdf/2506.23546v1)

Authors: Zhendong Yu, Weizhong Huang, Haiping Huang

Fixed points of recurrent neural networks can be leveraged to store and
generate information. These fixed points can be captured by the Boltzmann-Gibbs
measure, which leads to neural Langevin dynamics that can be used for sampling
and learning a real dataset. We call this type of generative model neural
Langevin machine, which is interpretable due to its analytic form of
distribution and is simple to train. Moreover, the learning process is derived
as a local asymmetric plasticity rule, bearing biological relevance. Therefore,
one can realize a continuous sampling of creative dynamics in a neural network,
mimicking an imagination process in brain circuits. This neural Langevin
machine may be another promising generative model, at least in its strength in
circuit-based sampling and biologically plausible learning rule.

### 4. [Towards Efficient and Accurate Spiking Neural Networks via Adaptive Bit Allocation](http://arxiv.org/pdf/2506.23717v1)

Authors: Xingting Yao, Qinghao Hu, Fei Zhou, Tielong Liu, Gang Li, Peisong Wang, Jian Cheng

Multi-bit spiking neural networks (SNNs) have recently become a heated
research spot, pursuing energy-efficient and high-accurate AI. However, with
more bits involved, the associated memory and computation demands escalate to
the point where the performance improvements become disproportionate. Based on
the insight that different layers demonstrate different importance and extra
bits could be wasted and interfering, this paper presents an adaptive bit
allocation strategy for direct-trained SNNs, achieving fine-grained layer-wise
allocation of memory and computation resources. Thus, SNN's efficiency and
accuracy can be improved. Specifically, we parametrize the temporal lengths and
the bit widths of weights and spikes, and make them learnable and controllable
through gradients. To address the challenges caused by changeable bit widths
and temporal lengths, we propose the refined spiking neuron, which can handle
different temporal lengths, enable the derivation of gradients for temporal
lengths, and suit spike quantization better. In addition, we theoretically
formulate the step-size mismatch problem of learnable bit widths, which may
incur severe quantization errors to SNN, and accordingly propose the step-size
renewal mechanism to alleviate this issue. Experiments on various datasets,
including the static CIFAR and ImageNet and the dynamic CIFAR-DVS and
DVS-GESTURE, demonstrate that our methods can reduce the overall memory and
computation cost while achieving higher accuracy. Particularly, our
SEWResNet-34 can achieve a 2.69\% accuracy gain and 4.16$\times$ lower bit
budgets over the advanced baseline work on ImageNet. This work will be fully
open-sourced.

### 5. [Marker Gene Method : Identifying Stable Solutions in a Dynamic Environment](http://arxiv.org/pdf/2506.23734v1)

Authors: Hao Shi, Xi Li, Fangfang Xie

Competitive Co-evolutionary Algorithms (CCEAs) are often hampered by complex
dynamics like intransitivity and the Red Queen effect, leading to unstable
convergence. To counter these challenges, this paper introduces the Marker Gene
Method (MGM), a framework that establishes stability by using a 'marker gene'
as a dynamic benchmark and an adaptive weighting mechanism to balance
exploration and exploitation. We provide rigorous mathematical proofs
demonstrating that MGM creates strong attractors near Nash Equilibria within
the Strictly Competitive Game framework. Empirically, MGM demonstrates its
efficacy across a spectrum of challenges: it stabilizes the canonical
Rock-Paper-Scissors game, significantly improves the performance of C-RMOEA/D
on ZDT benchmarks, and, when augmented with a Memory Pool (MP) extension, it
successfully tames the notoriously pathological Shapley Biased Game. This work
presents a theoretically sound and empirically validated framework that
substantially enhances the stability and robustness of CCEAs in complex
competitive environments.

### Networking and Internet Architecture

### 1. [Generative AI-enhanced Low-Altitude UAV-Mounted Stacked Intelligent Metasurfaces](http://arxiv.org/pdf/2506.23488v1)

Authors: Geng Sun, Mingzhe Fan, Lei Zhang, Hongyang Pan, Jiahui Li, Chuang Zhang, Linyao Li, Changyuan Zhao, Chau Yuen

Wireless communication systems face significant challenges in meeting the
increasing demands for higher data rates and more reliable connectivity in
complex environments. Stacked intelligent metasurfaces (SIMs) have emerged as a
promising technology for realizing wave-domain signal processing, with mobile
SIMs offering superior communication performance compared to their fixed
counterparts. In this paper, we investigate a novel unmanned aerial vehicle
(UAV)-mounted SIMs (UAV-SIMs) assisted communication system within the
low-altitude economy (LAE) networks paradigm, where UAVs function as both base
stations that cache SIM-processed data and mobile platforms that flexibly
deploy SIMs to enhance uplink communications from ground users. To maximize
network capacity, we formulate a UAV-SIM-based joint optimization problem
(USBJOP) that comprehensively addresses three critical aspects: the association
between UAV-SIMs and users, the three-dimensional positioning of UAV-SIMs, and
the phase shifts across multiple SIM layers. Due to the inherent non-convexity
and NP-hardness of USBJOP, we decompose it into three sub-optimization
problems, \textit{i.e.}, association between UAV-SIMs and users optimization
problem (AUUOP), UAV location optimization problem (ULOP), and UAV-SIM phase
shifts optimization problem (USPSOP), and solve them using an alternating
optimization strategy. Specifically, we transform AUUOP and ULOP into convex
forms solvable by the CVX tool, while addressing USPSOP through a generative
artificial intelligence (GAI)-based hybrid optimization algorithm. Simulations
demonstrate that our proposed approach significantly outperforms benchmark
schemes, achieving approximately 1.5 times higher network capacity compared to
suboptimal alternatives. Additionally, our proposed GAI method reduces the
algorithm runtime by 10\% while maintaining solution quality.

### 2. [Campus5G: A Campus Scale Private 5G Open RAN Testbed](http://arxiv.org/pdf/2506.23740v1)

Authors: Andrew E. Ferguson, Ujjwal Pawar, Tianxin Wang, Mahesh K. Marina

Mobile networks are embracing disaggregation, reflected by the industry trend
towards Open RAN. Private 5G networks are viewed as particularly suitable
contenders as early adopters of Open RAN, owing to their setting, high degree
of control, and opportunity for innovation they present. Motivated by this, we
have recently deployed Campus5G, the first of its kind campus-wide,
O-RAN-compliant private 5G testbed across the central campus of the University
of Edinburgh. We present in detail our process developing the testbed, from
planning, to architecting, to deployment, and measuring the testbed
performance. We then discuss the lessons learned from building the testbed, and
highlight some research opportunities that emerged from our deployment
experience.

### 3. [All Proof of Work But No Proof of Play](http://arxiv.org/pdf/2506.23435v1)

Authors: Hayder Tirmazi

Speedrunning is a competition that emerged from communities of early video
games such as Doom (1993). Speedrunners try to finish a game in minimal time.
Provably verifying the authenticity of submitted speedruns is an open problem.
Traditionally, best-effort speedrun verification is conducted by on-site human
observers, forensic audio analysis, or a rigorous mathematical analysis of the
game mechanics. Such methods are tedious, fallible, and, perhaps worst of all,
not cryptographic. Motivated by naivety and the Dunning-Kruger effect, we
attempt to build a system that cryptographically proves the authenticity of
speedruns. This paper describes our attempted solutions and ways to circumvent
them. Through a narration of our failures, we attempt to demonstrate the
difficulty of authenticating live and interactive human input in untrusted
environments, as well as the limits of signature schemes, game integrity, and
provable play.

### 4. [Securing the Sky: Integrated Satellite-UAV Physical Layer Security for Low-Altitude Wireless Networks](http://arxiv.org/pdf/2506.23493v1)

Authors: Jiahui Li, Geng Sun, Xiaoyu Sun, Fang Mei, Jingjing Wang, Xiangwang Hou, Daxin Tian, Victor C. M. Leung

Low-altitude wireless networks (LAWNs) have garnered significant attention in
the forthcoming 6G networks. In LAWNs, satellites with wide coverage and
unmanned aerial vehicles (UAVs) with flexible mobility can complement each
other to form integrated satellite-UAV networks, providing ubiquitous and
high-speed connectivity for low-altitude operations. However, the higher
line-of-sight probability in low-altitude airspace increases transmission
security concerns. In this work, we present a collaborative beamforming-based
physical layer security scheme for LAWNs. We introduce the fundamental aspects
of integrated satellite-UAV networks, physical layer security, UAV swarms, and
collaborative beamforming for LAWN applications. Following this, we highlight
several opportunities for collaborative UAV swarm secure applications enabled
by satellite networks, including achieving physical layer security in scenarios
involving data dissemination, data relay, eavesdropper collusion, and imperfect
eavesdropper information. Next, we detail two case studies: a secure relay
system and a two-way aerial secure communication framework specifically
designed for LAWN environments. Simulation results demonstrate that these
physical layer security schemes are effective and beneficial for secure
low-altitude wireless communications. A short practicality analysis shows that
the proposed method is applicable to LAWN scenarios. Finally, we discuss
current challenges and future research directions for enhancing security in
LAWNs.

### 5. [The Kubernetes Network Driver Model: A Composable Architecture for High-Performance Networking](http://arxiv.org/pdf/2506.23628v1)

Authors: Antonio Ojea

Traditional Kubernetes networking struggles to meet the escalating demands of
AI/ML and evolving Telco infrastructure. This paper introduces Kubernetes
Network Drivers (KNDs), a transformative, modular, and declarative architecture
designed to overcome current imperative provisioning and API limitations. KNDs
integrate network resource management into Kubernetes' core by utilizing
Dynamic Resource Allocation (DRA), Node Resource Interface (NRI) improvements,
and upcoming OCI Runtime Specification changes. Our DraNet implementation
demonstrates declarative attachment of network interfaces, including Remote
Direct Memory Access (RDMA) devices, significantly boosting high-performance
AI/ML workloads. This capability enables sophisticated cloud-native
applications and lays crucial groundwork for future Telco solutions, fostering
a "galaxy" of specialized KNDs for enhanced application delivery and reduced
operational complexity.

### 6. [Geminet: Learning the Duality-based Iterative Process for Lightweight Traffic Engineering in Changing Topologies](http://arxiv.org/pdf/2506.23640v1)

Authors: Ximeng Liu, Shizhen Zhao, Xinbing Wang

Recently, researchers have explored ML-based Traffic Engineering (TE),
leveraging neural networks to solve TE problems traditionally addressed by
optimization. However, existing ML-based TE schemes remain impractical: they
either fail to handle topology changes or suffer from poor scalability due to
excessive computational and memory overhead. To overcome these limitations, we
propose Geminet, a lightweight and scalable ML-based TE framework that can
handle changing topologies. Geminet is built upon two key insights: (i) a
methodology that decouples neural networks from topology by learning an
iterative gradient-descent-based adjustment process, as the update rule of
gradient descent is topology-agnostic, relying only on a few gradient-related
quantities; (ii) shifting optimization from path-level routing weights to
edge-level dual variables, reducing memory consumption by leveraging the fact
that edges are far fewer than paths. Evaluations on WAN and data center
datasets show that Geminet significantly improves scalability. Its neural
network size is only 0.04% to 7% of existing schemes, while handling topology
variations as effectively as HARP, a state-of-the-art ML-based TE approach,
without performance degradation. When trained on large-scale topologies,
Geminet consumes under 10 GiB of memory, more than eight times less than the
80-plus GiB required by HARP, while achieving 5.45 times faster convergence
speed, demonstrating its potential for large-scale deployment.

### 7. [How Long Can I Transmit? A Mobility Aware mmWave-based UAV Communication Framework](http://arxiv.org/pdf/2506.23755v1)

Authors: Shawon Mitra, Subhojit Sarkar, Sasthi C. Ghosh

One primary focus of next generation wireless communication networks is the
millimeterwave (mmWave) spectrum, typically considered in the 30 GHz to 300 GHz
frequency range. Despite their promise of high data rates, mmWaves suffer from
severe attenuation while passing through obstacles. Unmanned aerial vehicles
(UAVs) have been proposed to offset this limitation on account of their
additional degrees of freedom, which can be leveraged to provide line of sight
(LoS) transmission paths. While some prior works have proposed analytical
frameworks to compute the LoS probability for static ground users and a UAV,
the same is lacking for mobile users on the ground. In this paper, we consider
the popular Manhattan point line process (MPLP) to model an urban environment,
within which a ground user moves with a known velocity for a small time
interval along the roads. We derive an expression for the expected duration of
LoS between a static UAV in the air and a mobile ground user, and validate the
same through simulations. To demonstrate the efficacy of the proposed analysis,
we propose a simple user association algorithm that greedily assigns the UAVs
to users with the highest expected LoS time, and show that it outperforms the
existing benchmark schemes that assign the users to the nearest UAVs with LoS
without considering the user mobility.

### 8. [E-WAN: Efficient Communication in Energy Harvesting Low-Power Networks](http://arxiv.org/pdf/2506.23788v1)

Authors: Naomi Stricker, David Blaser, Andres Gomez, Lothar Thiele

The ever-increasing number of distributed embedded systems in the context of
the Internet of Things (IoT), Wireless Sensor Networks (WSN), and
Cyber-Physical Systems (CPS) rely on wireless communication to collect and
exchange data. Nodes can employ single-hop communication which, despite its
ease, may necessitate energy-intensive long-range communication to cover long
distances. Conversely, multi-hop communication allows for more energy-efficient
short-range communication since nodes can rely on other nodes to forward their
data. Yet, this approach requires relay nodes to be available and continuous
maintenance of a dynamically changing distributed state. At the same time,
energy harvesting has the potential to outperform traditional battery-based
systems by improving their lifetime, scalability with lower maintenance costs,
and environmental impact. However, the limited and temporally and spatially
variable harvested energy poses significant challenges for networking in energy
harvesting networks, particularly considering the energy demands and
characteristics of both multi-hop and single-hop communication. We propose
E-WAN, a protocol for energy harvesting wide-area low-power networks that
builds on the concept of \emph{virtual sub-networks} to enable
resource-efficient multi-hop communication when possible and reliable however
energy-intensive point-to-point communication otherwise. Nodes autonomously and
dynamically move between the two and adjust to changing network states and
resources based only on easily obtainable network state information. We
illustrate E-WAN's advantages both in terms of efficiency and adaptability in
various communication and harvesting scenarios. Furthermore, we demonstrate
E-WAN operating in a realistic setting by deploying an energy harvesting
network in a real-world indoor environment.

### 9. [Learning Constraints Directly from Network Data](http://arxiv.org/pdf/2506.23964v1)

Authors: Hongyu Hè, Minhao Jin, Maria Apostolaki

Network data conforms to a wide range of rules that arise from protocols,
design principles, and deployment decisions (e.g., a packet's queuing delay
must be less than its end-to-end delay). Formalizing such rules as logic
constraints can (i) improve the quality of synthetic data, (ii) reduce the
brittleness of machine learning (ML) models, and (iii) improve semantic
understanding of network measurements. However, these benefits remain out of
reach if rule extraction is manual or solely reliant on ML, as both approaches
yield incomplete, unreliable, and/or inaccurate rules.
  This paper formulates rule extraction as a constraint modeling problem and
introduces NetNomos that learns propositional logic constraints directly from
raw network measurements. Constraint modeling in this domain is uniquely
challenging due to the scale of the data, the inherent learning complexity and
passive environment, and the lack of ground truth supervision. NetNomos
addresses these challenges via a lattice-based search structured by constraint
specificity and succinctness. Our approach reduces learning complexity from
superquadratic to logarithmic and enables efficient traversal in combinatorial
search space.
  Our evaluations on diverse network datasets show that NetNomos learns all
benchmark rules, including those associated with as little as 0.01% of data
points, in under three hours. In contrast, baseline methods discover less than
25% of the rules and require several days to run. Through three case studies,
we show that: NetNomos (i) finds rule violations in the outputs of all seven
synthetic traffic generators, hence can be used to assess and guide their
generation process; (ii) detects semantic differences in traffic, hence can be
used for anomaly detection; and (iii) automatically finds rules used for
telemetry imputation, hence can support monitoring through inference.

### Robotics

### 1. [Risk-Based Filtering of Valuable Driving Situations in the Waymo Open Motion Dataset](http://arxiv.org/pdf/2506.23433v1)

Authors: Tim Puphal, Vipul Ramtekkar, Kenji Nishimiya

Improving automated vehicle software requires driving data rich in valuable
road user interactions. In this paper, we propose a risk-based filtering
approach that helps identify such valuable driving situations from large
datasets. Specifically, we use a probabilistic risk model to detect high-risk
situations. Our method stands out by considering a) first-order situations
(where one vehicle directly influences another and induces risk) and b)
second-order situations (where influence propagates through an intermediary
vehicle). In experiments, we show that our approach effectively selects
valuable driving situations in the Waymo Open Motion Dataset. Compared to the
two baseline interaction metrics of Kalman difficulty and Tracks-To-Predict
(TTP), our filtering approach identifies complex and complementary situations,
enriching the quality in automated vehicle testing. The risk data is made
open-source: https://github.com/HRI-EU/RiskBasedFiltering.

### 2. [A comprehensive control architecture for semi-autonomous dual-arm robots in agriculture settings](http://arxiv.org/pdf/2506.23723v1)

Authors: Jozsef Palmieri, Paolo Di Lillo, Stefano Chiaverini, Alessandro Marino

The adoption of mobile robotic platforms in complex environments, such as
agricultural settings, requires these systems to exhibit a flexible yet
effective architecture that integrates perception and control. In such
scenarios, several tasks need to be accomplished simultaneously, ranging from
managing robot limits to performing operational tasks and handling human
inputs. The purpose of this paper is to present a comprehensive control
architecture for achieving complex tasks such as robotized harvesting in
vineyards within the framework of the European project CANOPIES. In detail, a
16-DOF dual-arm mobile robot is employed, controlled via a Hierarchical
Quadratic Programming (HQP) approach capable of handling both equality and
inequality constraints at various priorities to harvest grape bunches selected
by the perception system developed within the project. Furthermore, given the
complexity of the scenario and the uncertainty in the perception system, which
could potentially lead to collisions with the environment, the handling of
interaction forces is necessary. Remarkably, this was achieved using the same
HQP framework. This feature is further leveraged to enable semi-autonomous
operations, allowing a human operator to assist the robotic counterpart in
completing harvesting tasks. Finally, the obtained results are validated
through extensive testing conducted first in a laboratory environment to prove
individual functionalities, then in a real vineyard, encompassing both
autonomous and semi-autonomous grape harvesting operations.

### 3. [Motion Tracking with Muscles: Predictive Control of a Parametric Musculoskeletal Canine Model](http://arxiv.org/pdf/2506.23768v1)

Authors: Vittorio La Barbera, Steven Bohez, Leonard Hasenclever, Yuval Tassa, John R. Hutchinson

We introduce a novel musculoskeletal model of a dog, procedurally generated
from accurate 3D muscle meshes. Accompanying this model is a motion
capture-based locomotion task compatible with a variety of control algorithms,
as well as an improved muscle dynamics model designed to enhance convergence in
differentiable control frameworks. We validate our approach by comparing
simulated muscle activation patterns with experimentally obtained
electromyography (EMG) data from previous canine locomotion studies. This work
aims to bridge gaps between biomechanics, robotics, and computational
neuroscience, offering a robust platform for researchers investigating muscle
actuation and neuromuscular control.We plan to release the full model along
with the retargeted motion capture clips to facilitate further research and
development.

### 4. [World4Omni: A Zero-Shot Framework from Image Generation World Model to Robotic Manipulation](http://arxiv.org/pdf/2506.23919v1)

Authors: Haonan Chen, Bangjun Wang, Jingxiang Guo, Tianrui Zhang, Yiwen Hou, Xuchuan Huang, Chenrui Tie, Lin Shao

Improving data efficiency and generalization in robotic manipulation remains
a core challenge. We propose a novel framework that leverages a pre-trained
multimodal image-generation model as a world model to guide policy learning. By
exploiting its rich visual-semantic representations and strong generalization
across diverse scenes, the model generates open-ended future state predictions
that inform downstream manipulation. Coupled with zero-shot low-level control
modules, our approach enables general-purpose robotic manipulation without
task-specific training. Experiments in both simulation and real-world
environments demonstrate that our method achieves effective performance across
a wide range of manipulation tasks with no additional data collection or
fine-tuning. Supplementary materials are available on our website:
https://world4omni.github.io/.

### 5. [Predictive Risk Analysis and Safe Trajectory Planning for Intelligent and Connected Vehicles](http://arxiv.org/pdf/2506.23999v1)

Authors: Zeyu Han, Mengchi Cai, Chaoyi Chen, Qingwen Meng, Guangwei Wang, Ying Liu, Qing Xu, Jianqiang Wang, Keqiang Li

The safe trajectory planning of intelligent and connected vehicles is a key
component in autonomous driving technology. Modeling the environment risk
information by field is a promising and effective approach for safe trajectory
planning. However, existing risk assessment theories only analyze the risk by
current information, ignoring future prediction. This paper proposes a
predictive risk analysis and safe trajectory planning framework for intelligent
and connected vehicles. This framework first predicts future trajectories of
objects by a local risk-aware algorithm, following with a
spatiotemporal-discretised predictive risk analysis using the prediction
results. Then the safe trajectory is generated based on the predictive risk
analysis. Finally, simulation and vehicle experiments confirm the efficacy and
real-time practicability of our approach.

### 6. [Towards foundational LiDAR world models with efficient latent flow matching](http://arxiv.org/pdf/2506.23434v1)

Authors: Tianran Liu, Shengwen Zhao, Nicholas Rhinehart

LiDAR-based world models offer more structured and geometry-aware
representations than their image-based counterparts. However, existing LiDAR
world models are narrowly trained; each model excels only in the domain for
which it was built. Can we develop LiDAR world models that exhibit strong
transferability across multiple domains? We conduct the first systematic domain
transfer study across three demanding scenarios: (i) outdoor to indoor
generalization, (ii) sparse-beam \& dense-beam adaptation, and (iii)
non-semantic to semantic transfer. Given different amounts of fine-tuning data,
our experiments show that a single pre-trained model can achieve up to 11%
absolute improvement (83\% relative) over training from scratch and outperforms
training from scratch in 30/36 of our comparisons. This transferability of
dynamic learning significantly reduces the reliance on manually annotated data
for semantic occupancy forecasting: our method exceed the previous semantic
occupancy forecasting models with only 5% of the labeled training data required
by prior models. We also observed inefficiencies of current LiDAR world models,
mainly through their under-compression of LiDAR data and inefficient training
objectives. To address this, we propose a latent conditional flow matching
(CFM)-based frameworks that achieves state-of-the-art reconstruction accuracy
using only half the training data and a compression ratio 6 times higher than
that of prior methods. Our model achieves SOTA performance on
future-trajectory-conditioned semantic occupancy forecasting while being 23x
more computationally efficient (a 28x FPS speedup); and achieves SOTA
performance on semantic occupancy forecasting while being 2x more
computationally efficient (a 1.1x FPS speedup).

### 7. [Passage-traversing optimal path planning with sampling-based algorithms](http://arxiv.org/pdf/2506.23614v1)

Authors: Jing Huang, Hao Su, Kwok Wai Samuel Au

This paper introduces a new paradigm of optimal path planning, i.e.,
passage-traversing optimal path planning (PTOPP), that optimizes paths'
traversed passages for specified optimization objectives. In particular, PTOPP
is utilized to find the path with optimal accessible free space along its
entire length, which represents a basic requirement for paths in robotics. As
passages are places where free space shrinks and becomes constrained, the core
idea is to leverage the path's passage traversal status to characterize its
accessible free space comprehensively. To this end, a novel passage detection
and free space decomposition method using proximity graphs is proposed,
enabling fast detection of sparse but informative passages and environment
decompositions. Based on this preprocessing, optimal path planning with
accessible free space objectives or constraints is formulated as PTOPP problems
compatible with sampling-based optimal planners. Then, sampling-based
algorithms for PTOPP, including their dependent primitive procedures, are
developed leveraging partitioned environments for fast passage traversal check.
All these methods are implemented and thoroughly tested for effectiveness and
efficiency validation. Compared to existing approaches, such as clearance-based
methods, PTOPP demonstrates significant advantages in configurability, solution
optimality, and efficiency, addressing prior limitations and incapabilities. It
is believed to provide an efficient and versatile solution to accessible free
space optimization over conventional avenues and more generally, to a broad
class of path planning problems that can be formulated as PTOPP.

### 8. [PAC Bench: Do Foundation Models Understand Prerequisites for Executing Manipulation Policies?](http://arxiv.org/pdf/2506.23725v1)

Authors: Atharva Gundawar, Som Sagar, Ransalu Senanayake

Vision-Language Models (VLMs) are increasingly pivotal for generalist robot
manipulation, enabling tasks such as physical reasoning, policy generation, and
failure detection. However, their proficiency in these high-level applications
often assumes a deep understanding of low-level physical prerequisites, a
capability that remains largely unverified. For robots to perform actions
reliably, they must comprehend intrinsic object properties (e.g., material,
weight), action affordances (e.g., graspable, stackable), and physical
constraints (e.g., stability, reachability, or an object's state, such as being
closed). Despite the widespread use of VLMs in manipulation tasks, we argue
that off-the-shelf models may lack this granular, physically grounded
understanding, as such prerequisites are often overlooked during training.
  To address this critical gap, we introduce PAC Bench, a comprehensive
benchmark designed to systematically evaluate VLMs on their understanding of
core Properties, Affordances, and Constraints (PAC) from a task executability
perspective. PAC Bench features a diverse dataset with over 30,000 annotations,
comprising 673 real-world images (115 object classes, 15 property types, and 1
to 3 affordances defined per class), 100 real-world humanoid-view scenarios,
and 120 unique simulated constraint scenarios across four tasks.
  Our evaluations reveal significant gaps in the ability of current VLMs to
grasp fundamental physical concepts, highlighting limitations in their
suitability for reliable robot manipulation and pointing to key areas for
targeted research. PAC Bench also serves as a standardized benchmark for
rigorously evaluating physical reasoning in VLMs and guiding the development of
more robust, physically grounded models for robotic applications.
  Project Page: https://pacbench.github.io/

### 9. [Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving](http://arxiv.org/pdf/2506.23771v1)

Authors: Guizhe Jin, Zhuoren Li, Bo Leng, Ran Yu, Lu Xiong

Reinforcement Learning (RL) is increasingly used in autonomous driving (AD)
and shows clear advantages. However, most RL-based AD methods overlook policy
structure design. An RL policy that only outputs short-timescale vehicle
control commands results in fluctuating driving behavior due to fluctuations in
network outputs, while one that only outputs long-timescale driving goals
cannot achieve unified optimality of driving behavior and control. Therefore,
we propose a multi-timescale hierarchical reinforcement learning approach. Our
approach adopts a hierarchical policy structure, where high- and low-level RL
policies are unified-trained to produce long-timescale motion guidance and
short-timescale control commands, respectively. Therein, motion guidance is
explicitly represented by hybrid actions to capture multimodal driving
behaviors on structured road and support incremental low-level extend-state
updates. Additionally, a hierarchical safety mechanism is designed to ensure
multi-timescale safety. Evaluation in simulator-based and HighD dataset-based
highway multi-lane scenarios demonstrates that our approach significantly
improves AD performance, effectively increasing driving efficiency, action
consistency and safety.

### 10. [Adapt Your Body: Mitigating Proprioception Shifts in Imitation Learning](http://arxiv.org/pdf/2506.23944v1)

Authors: Fuhang Kuang, Jiacheng You, Yingdong Hu, Tong Zhang, Chuan Wen, Yang Gao

Imitation learning models for robotic tasks typically rely on multi-modal
inputs, such as RGB images, language, and proprioceptive states. While
proprioception is intuitively important for decision-making and obstacle
avoidance, simply incorporating all proprioceptive states leads to a surprising
degradation in imitation learning performance. In this work, we identify the
underlying issue as the proprioception shift problem, where the distributions
of proprioceptive states diverge significantly between training and deployment.
To address this challenge, we propose a domain adaptation framework that
bridges the gap by utilizing rollout data collected during deployment. Using
Wasserstein distance, we quantify the discrepancy between expert and rollout
proprioceptive states and minimize this gap by adding noise to both sets of
states, proportional to the Wasserstein distance. This strategy enhances
robustness against proprioception shifts by aligning the training and
deployment distributions. Experiments on robotic manipulation tasks demonstrate
the efficacy of our method, enabling the imitation policy to leverage
proprioception while mitigating its adverse effects. Our approach outperforms
the naive solution which discards proprioception, and other baselines designed
to address distributional shifts.

### Software Engineering

### 1. [Improving vulnerability type prediction and line-level detection via adversarial training-based data augmentation and multi-task learning](http://arxiv.org/pdf/2506.23534v1)

Authors: Siyu Chen, Jiongyi Yang, Xiang Chen, Menglin Zheng, Minnan Wei, Xiaolin Ju

Context: Software vulnerabilities pose a significant threat to modern
software systems, as evidenced by the growing number of reported
vulnerabilities and cyberattacks. These escalating trends underscore the urgent
need for effective approaches that can automatically detect and understand
software vulnerabilities. Objective: However, the scarcity of labeled samples
and the class imbalance issue in vulnerability datasets present significant
challenges for both Vulnerability Type Prediction (VTP) and Line-level
Vulnerability Detection (LVD), especially for rare yet critical vulnerability
types. Moreover, most existing studies treat VTP and LVD as independent tasks,
overlooking their inherent correlation, which limits the potential to leverage
shared semantic patterns across tasks. Methods: To address these limitations,
we propose a unified approach that integrates Embedding-Layer Driven
Adversarial Training (EDAT) with Multi-task Learning (MTL). Specifically, EDAT
enhances model robustness by introducing adversarial perturbations to
identifier embeddings, guided by semantic importance. Meanwhile, MTL improves
overall performance by leveraging shared representations and inter-task
correlations between VTP and LVD. Results: Extensive experiments demonstrate
that our proposed approach outperforms state-of-the-art baselines on both VTP
and LVD tasks. For VTP, it yields notable improvements in accuracy, precision,
recall, and F1-score, particularly in identifying rare vulnerability types.
Similarly, for LVD, our approach enhances line-level detection accuracy while
significantly reducing false positives. Conclusion: Our study demonstrates that
combining EDAT with MTL provides a unified solution that improves performance
on both tasks and warrants further investigation.

### 2. [Comparative Analysis of the Code Generated by Popular Large Language Models (LLMs) for MISRA C++ Compliance](http://arxiv.org/pdf/2506.23535v1)

Authors: Malik Muhammad Umer

Safety-critical systems are engineered systems whose failure or malfunction
could result in catastrophic consequences. The software development for
safety-critical systems necessitates rigorous engineering practices and
adherence to certification standards like DO-178C for avionics. DO-178C is a
guidance document which requires compliance to well-defined software coding
standards like MISRA C++ to enforce coding guidelines that prevent the use of
ambiguous, unsafe, or undefined constructs. Large Language Models (LLMs) have
demonstrated significant capabilities in automatic code generation across a
wide range of programming languages, including C++. Despite their impressive
performance, code generated by LLMs in safety-critical domains must be
carefully analyzed for conformance to MISRA C++ coding standards. In this
paper, I have conducted a comparative analysis of the C++ code generated by
popular LLMs including: OpenAI ChatGPT, Google Gemini, DeepSeek, Meta AI, and
Microsoft Copilot for compliance with MISRA C++.

### 3. [Towards a Science of Developer eXperience (DevX)](http://arxiv.org/pdf/2506.23715v1)

Authors: Benoit Combemale

As software continues to permeate nearly every facet of modern life, the
complexity and ubiquity of digital services underscore the need for
sustainable, effective, and inclusive software development practices. Although
software engineering has made significant progress in technical challenges
since its inception, the human experience of those involved in software
creation, broadly defined as developers, remains underexplored. This column
advocates for the formal recognition of Developer eXperience (DevX) as a
distinct research field. We argue that DevX profoundly influences critical
development activities and overall productivity, especially as development
becomes increasingly collaborative and diverse in terms of application domains.
Building on existing efforts to measure and enhance DevX, we identify key
rationales, scientific enablers, and interdisciplinary intersections that
support this emerging discipline. We also outline the core scientific
challenges ahead, aiming to call for actions from the research community and to
promote more human-centered approaches to software engineering.

### 4. [A Survey of LLM-based Automated Program Repair: Taxonomies, Design Paradigms, and Applications](http://arxiv.org/pdf/2506.23749v1)

Authors: Boyang Yang, Zijian Cai, Fengling Liu, Bach Le, Lingming Zhang, Tegawendé F. Bissyandé, Yang Liu, Haoye Tian

Large language models (LLMs) are reshaping automated program repair (APR). We
categorize the recent 63 LLM-based APR systems published from January 2022 to
June 2025 into four paradigms, and show how retrieval- or analysis-augmented
contexts strengthen any of them. This taxonomy clarifies key trade-offs:
fine-tuning delivers strong task alignment at high training cost; prompting
enables rapid deployment but is limited by prompt design and context windows;
procedural pipelines offer reproducible control with moderate overhead; agentic
frameworks tackle multi-hunk or cross-file bugs at the price of increased
latency and complexity. Persistent challenges include verifying semantic
correctness beyond test suites, repairing repository-scale defects, and
lowering the costs of LLMs. We outline research directions that combine
lightweight human feedback, repository-aware retrieval, code analysis, and
cost-aware planning to advance reliable and efficient LLM-based APR.

### 5. [Requirements for Active Assistance of Natural Questions in Software Architecture](http://arxiv.org/pdf/2506.23898v1)

Authors: Diogo Lemos, Ademar Aguiar, Neil B. Harrison

Natural questions are crucial to shaping key architectural decisions and
preserving architectural knowledge. They arise organically during the
architectural design process, often resulting from the existing architectural
experience of the designer and the distinctive characteristics of the system
being designed. However, natural questions are often mismanaged or ignored,
which can lead to architectural drift, knowledge loss, inefficient resource
use, or poor understandability of the system's architecture. We aim to better
understand the lifecycle of natural questions, its key requirements, challenges
and difficulties, and then to envision an assisted environment to properly
support it. The environment should be adaptable and responsive to real-world
constraints and uncertainties by seamlessly integrating knowledge management
tools and artificial intelligence techniques into software development
workflows. Based on existing literature, a requirements workshop, and three
design iterations, we proposed a lifecycle for natural questions and elicited
essential functional and non-functional requirements for such an environment.
At last, the results of a survey conducted with experts helped to analyze and
validate the elicited requirements and proposed features for the environment to
enhance collaboration, decision-making, and the preservation of architectural
knowledge more effectively than conventional methods.

### 6. [Bug Fixing with Broader Context: Enhancing LLM-Based Program Repair via Layered Knowledge Injection](http://arxiv.org/pdf/2506.24015v1)

Authors: Ramtin Ehsani, Esteban Parra, Sonia Haiduc, Preetha Chatterjee

Prompting LLMs with bug-related context (e.g., error messages, stack traces)
improves automated program repair, but many bugs still remain unresolved. In
real-world projects, developers often rely on broader repository and
project-level context beyond the local code to resolve such bugs. In this
paper, we investigate how automatically extracting and providing such knowledge
can improve LLM-based program repair. We propose a layered knowledge injection
framework that incrementally augments LLMs with structured context. It starts
with the Bug Knowledge Layer, which includes information such as the buggy
function and failing tests; expands to the Repository Knowledge Layer, which
adds structural dependencies, related files, and commit history; and finally
injects the Project Knowledge Layer, which incorporates relevant details from
documentation and previously fixed bugs. We evaluate this framework on a
dataset of 314 bugs from BugsInPy using two LLMs (Llama 3.3 and GPT-4o-mini),
and analyze fix rates across six bug types. By progressively injecting
knowledge across layers, our approach achieves a fix rate of 79% (250/314)
using Llama 3.3, a significant improvement of 23% over previous work. All bug
types show improvement with the addition of repository-level context, while
only a subset benefit further from project-level knowledge, highlighting that
different bug types require different levels of contextual information for
effective repair. We also analyze the remaining unresolved bugs and find that
more complex and structurally isolated bugs, such as Program Anomaly and GUI
bugs, remain difficult even after injecting all available information. Our
results show that layered context injection improves program repair and suggest
the need for interactive and adaptive APR systems.

### 7. [What Challenges Do Developers Face When Using Verification-Aware Programming Languages?](http://arxiv.org/pdf/2506.23696v1)

Authors: Francisco Oliveira, Alexandra Mendes, Carolina Carreira

Software reliability is critical in ensuring that the digital systems we
depend on function correctly. In software development, increasing software
reliability often involves testing. However, for complex and critical systems,
developers can use Design by Contract (DbC) methods to define precise
specifications that software components must satisfy. Verification-Aware (VA)
programming languages support DbC and formal verification at compile-time or
run-time, offering stronger correctness guarantees than traditional testing.
However, despite the strong guarantees provided by VA languages, their adoption
remains limited. In this study, we investigate the barriers to adopting VA
languages by analyzing developer discussions on public forums using topic
modeling techniques. We complement this analysis with a developer survey to
better understand the practical challenges associated with VA languages. Our
findings reveal key obstacles to adoption, including steep learning curves and
usability issues. Based on these insights, we identify actionable
recommendations to improve the usability and accessibility of VA languages. Our
findings suggest that simplifying tool interfaces, providing better educational
materials, and improving integration with everyday development environments
could improve the usability and adoption of these languages. Our work provides
actionable insights for improving the usability of VA languages and making
verification tools more accessible.

### 8. [Software Engineering for Large Language Models: Research Status, Challenges and the Road Ahead](http://arxiv.org/pdf/2506.23762v1)

Authors: Hongzhou Rao, Yanjie Zhao, Xinyi Hou, Shenao Wang, Haoyu Wang

The rapid advancement of large language models (LLMs) has redefined
artificial intelligence (AI), pushing the boundaries of AI research and
enabling unbounded possibilities for both academia and the industry. However,
LLM development faces increasingly complex challenges throughout its lifecycle,
yet no existing research systematically explores these challenges and solutions
from the perspective of software engineering (SE) approaches. To fill the gap,
we systematically analyze research status throughout the LLM development
lifecycle, divided into six phases: requirements engineering, dataset
construction, model development and enhancement, testing and evaluation,
deployment and operations, and maintenance and evolution. We then conclude by
identifying the key challenges for each phase and presenting potential research
directions to address these challenges. In general, we provide valuable
insights from an SE perspective to facilitate future advances in LLM
development.

### 9. [An ontological lens on attack trees: Toward adequacy and interoperability](http://arxiv.org/pdf/2506.23841v1)

Authors: Ítalo Oliveira, Stefano M. Nicoletti, Gal Engelberg, Mattia Fumagalli, Dan Klein, Giancarlo Guizzardi

Attack Trees (AT) are a popular formalism for security analysis. They are
meant to display an attacker's goal decomposed into attack steps needed to
achieve it and compute certain security metrics (e.g., attack cost,
probability, and damage). ATs offer three important services: (a) conceptual
modeling capabilities for representing security risk management scenarios, (b)
a qualitative assessment to find root causes and minimal conditions of
successful attacks, and (c) quantitative analyses via security metrics
computation under formal semantics, such as minimal time and cost among all
attacks. Still, the AT language presents limitations due to its lack of
ontological foundations, thus compromising associated services. Via an
ontological analysis grounded in the Common Ontology of Value and Risk (COVER)
-- a reference core ontology based on the Unified Foundational Ontology (UFO)
-- we investigate the ontological adequacy of AT and reveal four significant
shortcomings: (1) ambiguous syntactical terms that can be interpreted in
various ways; (2) ontological deficit concerning crucial domain-specific
concepts; (3) lacking modeling guidance to construct ATs decomposing a goal;
(4) lack of semantic interoperability, resulting in ad hoc stand-alone tools.
We also discuss existing incremental solutions and how our analysis paves the
way for overcoming those issues through a broader approach to risk management
modeling.

### 10. [QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration](http://arxiv.org/pdf/2506.23644v1)

Authors: Junze Hu, Xiangyu Jin, Yizhe Zeng, Yuling Liu, Yunpeng Li, Dan Du, Kaiyu Xie, Hongsong Zhu

We introduce QLPro, a vulnerability detection framework that systematically
integrates LLMs and static analysis tools to enable comprehensive vulnerability
detection across entire open-source projects.We constructed a new dataset,
JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed
vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only
24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro
discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed
as 0-days.

### Social and Information Networks

### 1. [Reconciling Attribute and Structural Anomalies for Improved Graph Anomaly Detection](http://arxiv.org/pdf/2506.23469v1)

Authors: Chunjing Xiao, Jiahui Lu, Xovee Xu, Fan Zhou, Tianshu Xie, Wei Lu, Lifeng Xu

Graph anomaly detection is critical in domains such as healthcare and
economics, where identifying deviations can prevent substantial losses.
Existing unsupervised approaches strive to learn a single model capable of
detecting both attribute and structural anomalies. However, they confront the
tug-of-war problem between two distinct types of anomalies, resulting in
suboptimal performance. This work presents TripleAD, a mutual
distillation-based triple-channel graph anomaly detection framework. It
includes three estimation modules to identify the attribute, structural, and
mixed anomalies while mitigating the interference between different types of
anomalies. In the first channel, we design a multiscale attribute estimation
module to capture extensive node interactions and ameliorate the over-smoothing
issue. To better identify structural anomalies, we introduce a link-enhanced
structure estimation module in the second channel that facilitates information
flow to topologically isolated nodes. The third channel is powered by an
attribute-mixed curvature, a new indicator that encapsulates both attribute and
structural information for discriminating mixed anomalies. Moreover, a mutual
distillation strategy is introduced to encourage communication and
collaboration between the three channels. Extensive experiments demonstrate the
effectiveness of the proposed TripleAD model against strong baselines.

### 2. [Scaling Human Judgment in Community Notes with LLMs](http://arxiv.org/pdf/2506.24118v1)

Authors: Haiwen Li, Soham De, Manon Revel, Andreas Haupt, Brad Miller, Keith Coleman, Jay Baxter, Martin Saveski, Michiel A. Bakker

This paper argues for a new paradigm for Community Notes in the LLM era: an
open ecosystem where both humans and LLMs can write notes, and the decision of
which notes are helpful enough to show remains in the hands of humans. This
approach can accelerate the delivery of notes, while maintaining trust and
legitimacy through Community Notes' foundational principle: A community of
diverse human raters collectively serve as the ultimate evaluator and arbiter
of what is helpful. Further, the feedback from this diverse community can be
used to improve LLMs' ability to produce accurate, unbiased, broadly helpful
notes--what we term Reinforcement Learning from Community Feedback (RLCF). This
becomes a two-way street: LLMs serve as an asset to humans--helping deliver
context quickly and with minimal effort--while human feedback, in turn,
enhances the performance of LLMs. This paper describes how such a system can
work, its benefits, key new risks and challenges it introduces, and a research
agenda to solve those challenges and realize the potential of this approach.

### 3. [Breadth, Depth, and Flux of Course-Prerequisite Networks](http://arxiv.org/pdf/2506.23510v1)

Authors: Konstantin Zuev, Pavlos Stavrinides

Course-prerequisite networks (CPNs) are directed acyclic graphs that model
complex academic curricula by representing courses as nodes and dependencies
between them as directed links. These networks are indispensable tools for
visualizing, studying, and understanding curricula. For example, CPNs can be
used to detect important courses, improve advising, guide curriculum design,
analyze graduation time distributions, and quantify the strength of knowledge
flow between different university departments. However, most CPN analyses to
date have focused only on micro- and meso-scale properties. To fill this gap,
we define and study three new global CPN measures: breadth, depth, and flux.
All three measures are invariant under transitive reduction and are based on
the concept of topological stratification, which generalizes topological
ordering in directed acyclic graphs. These measures can be used for macro-scale
comparison of different CPNs. We illustrate the new measures numerically by
applying them to three real and synthetic CPNs from three universities: the
Cyprus University of Technology, the California Institute of Technology, and
Johns Hopkins University. The CPN data analyzed in this paper are publicly
available in a GitHub repository.

### 4. [LLM Agents Are the Antidote to Walled Gardens](http://arxiv.org/pdf/2506.23978v1)

Authors: Samuele Marro, Philip Torr

While the Internet's core infrastructure was designed to be open and
universal, today's application layer is dominated by closed, proprietary
platforms. Open and interoperable APIs require significant investment, and
market leaders have little incentive to enable data exchange that could erode
their user lock-in. We argue that LLM-based agents fundamentally disrupt this
status quo. Agents can automatically translate between data formats and
interact with interfaces designed for humans: this makes interoperability
dramatically cheaper and effectively unavoidable. We name this shift universal
interoperability: the ability for any two digital services to exchange data
seamlessly using AI-mediated adapters. Universal interoperability undermines
monopolistic behaviours and promotes data portability. However, it can also
lead to new security risks and technical debt. Our position is that the ML
community should embrace this development while building the appropriate
frameworks to mitigate the downsides. By acting now, we can harness AI to
restore user freedom and competitive markets without sacrificing security.

### Systems and Control

### 1. [Power-Gas Infrastructure Planning under Weather-induced Supply and Demand Uncertainties](http://arxiv.org/pdf/2506.23509v1)

Authors: Rahman Khorramfar, Dharik Mallapragada, Saurabh Amin

Implementing economy-wide decarbonization strategies based on decarbonizing
the power grid via variable renewable energy (VRE) expansion and
electrification of end-uses requires new approaches for energy infrastructure
planning that consider, among other factors, weather-induced uncertainty in
demand and VRE supply. An energy planning model that fails to account for these
uncertainties can hinder the intended transition efforts to a low-carbon grid
and increase the risk of supply shortage especially during extreme weather
conditions. Here, we consider the generation and transmission expansion problem
of joint power-gas infrastructure and operations planning under the uncertainty
of both demand and renewable supply. We propose two distributionally robust
optimization approaches based on moment (MDRO) and Wasserstein distance (WDRO)
ambiguity sets to endogenize these uncertainties and account for the change in
the underlying distribution of these parameters that is caused by the climate
change, among other factors. Furthermore, our model considers the risk-aversion
of the energy planners in the modeling framework via the conditional
value-at-risk (CVaR) metric. An equivalent mixed-integer linear programming
(MILP) reformulation of both modeling frameworks is presented, and a
computationally efficient approximation scheme to obtain near-optimal solutions
is proposed. We demonstrate the resulting DRO planning models and solution
strategy via a New England case study under different levels of end-use
electrification and decarbonization targets. Our experiments systematically
explore different modeling aspects and compare the DRO models with stochastic
programming (SP) results.

### 2. [A Bidirectional Power Router for Traceable Multi-energy Management](http://arxiv.org/pdf/2506.23554v1)

Authors: Shiu Mochiyama, Ryo Takahashi, Yoshihiko Susuki

To address challenges in improving self-consumption of renewables and
resilience in local residential power systems, the earlier work of the authors
introduced a novel multi-energy management concept, integrating bidirectional
power routing and electricity-hydrogen conversion. This paper focuses on an
experimental verification of the bidirectional power router based on
line-switching, the essential hardware to realize the concept. The primary
contribution is the validation of the router's capability to handle dynamic
change of bidirectional power flow. Furthermore, to achieve bidirectional power
routing without affecting the smooth and stable operation of the power system,
a novel algorithm for router's switching is designed based on power flow
monitoring. The effectiveness of the proposed method is demonstrated through an
experiment using a setup with a commercially available stationary battery.

### 3. [Reliability Assessment of Power System Based on the Dichotomy Method](http://arxiv.org/pdf/2506.23649v1)

Authors: Wenjie Wan, Han Hu, Feiyu Chen, Xiaoyu Liu, Kequan Zhao

With a sustainable increase in the scale of power system, the number of
states in the state space grows exponentially, and the reliability assessment
of the power system faces enormous challenges. Traditional state-by-state
assessment methods, such as state enumeration (SE) and Monte Carlo simulation
(MCS) methods, have encountered performance bottlenecks in terms of efficiency
and accuracy. In this paper, the Boolean lattice representation theory of the
state space was studied, and a dichotomy method was proposed to efficiently
partition the state space into some disjoint sub-lattices with a relatively
small number of optimal power flow (OPF) operations. Based on lattice
partition, the reliability indices of the entire space can be calculated
lattice-by-lattice. In addition, alone with the partitioning procedure, the
calculated loss of load probability (LOLP) monotonically increases and rapidly
tends to the analytic value with the designated error bound. Moreover, we
designed a customized Monte Carlo sampling method in lattices of interest to
compute expected energy not supply (EENS). The experiments are conducted on the
RBTS and RTS-79 systems. The results show that the proposed method achieves the
analytic LOLP of the RBTS system after five hundreds of OPF operations, which
is about hundreds of times faster than traditional methods, and the designed
Monte Carlo sampling method converged after thousands of OPF operations on test
systems.

### 4. [A Data-Ensemble-Based Approach for Sample-Efficient LQ Control of Linear Time-Varying Systems](http://arxiv.org/pdf/2506.23716v1)

Authors: Sahel Vahedi Noori, Maryam Babazadeh

This paper presents a sample-efficient, data-driven control framework for
finite-horizon linear quadratic (LQ) control of linear time-varying (LTV)
systems. In contrast to the time-invariant case, the time-varying LQ problem
involves a differential Riccati equation (DRE) with time-dependent parameters
and terminal boundary constraints. We formulate the LQ problem as a nonconvex
optimization problem and conduct a rigorous analysis of its dual structure. By
exploiting the inherent convexity of the dual problem and analyzing the KKT
conditions, we derive an explicit relationship between the optimal dual
solution and the parameters of the associated Q-function in time-varying case.
This theoretical insight supports the development of a novel, sample-efficient,
non-iterative semidefinite programming (SDP) algorithm that directly computes
the optimal sequence of feedback gains from an ensemble of input-state data
sequences without model identification. The resulting convex, data-dependent
framework provides global optimality guarantees for completely unknown LTV
systems. As a special case, the method also applies to finite-horizon LQ
control of linear time-invariant (LTI) systems. In this setting, a single
input-state trajectory suffices to identify the optimal LQ feedback policy,
improving significantly over existing Q-learning approaches for finite horizon
LTI systems that typically require data from multiple episodes. The approach
provides a new optimization-based perspective on Q-learning in time-varying
settings and contributes to the broader understanding of data-driven control in
non-stationary environments. Simulation results show that, compared to recent
methods, the proposed approach achieves superior optimality and sample
efficiency on LTV systems, and indicates potential for stabilizing and optimal
control of nonlinear systems.

### 5. [A Digital Twinning Approach to Decarbonisation: Research Challenges](http://arxiv.org/pdf/2506.23733v1)

Authors: Blair Archibald, Paul Harvey, Michele Sevegnani

Transportation accounts for around 27% of green house gas emissions in the
UK. While an obvious priority area for decarbonisation, and aligned to the UK
government goal of reducing emissions by 68% for 2030, the free-market nature
of the transportation sector combined with its fundamentally implicit and
pervasive connections to all aspects of society and national infrastructure
mean that all decarbonisation efforts to date have been siloed within a single
transport sector, e.g. only considering greener aviation fuels. Truly
decarbonising transport requires radical changes to the entire transport
infrastructure, and since that transport does not happen in isolation, a single
user often using multiple modes, we need a view over the whole transport
system. The first step to solving a problem is to understand it. As a result of
the fragmented nature of the transportation sector, there is currently no
system level view. Without the ability to monitor even adjacent transport
domains, the ability for people or organisations to (dynamically) adapt their
operations for decarbonisation outcomes is unrealistic. As transportation is a
complex social-techno-economic system, information and knowledge sharing is a
must to be able to understand and explore potential solutions to the
decarbonisation challenge. We believe a Federated Digital Twinning Approach has
the potential to tackle transport decarbonisation problems, and, in this
extended abstract, we give an overview of the research required to tackle the
fundamental challenges around digital twin design, generation, validation and
verification.

### 6. [On sample-based functional observability of linear systems](http://arxiv.org/pdf/2506.23744v1)

Authors: Isabelle Krauss, Victor G. Lopez, Matthias A. Müller

Sample-based observability characterizes the ability to reconstruct the
internal state of a dynamical system by using limited output information, i.e.,
when measurements are only infrequently and/or irregularly available. In this
work, we investigate the concept of functional observability, which refers to
the ability to infer a function of the system state from the outputs, within a
samplebased framework. Here, we give necessary and sufficient conditions for a
system to be sample-based functionally observable, and formulate conditions on
the sampling schemes such that these are satisfied. Furthermore, we provide a
numerical example, where we demonstrate the applicability of the obtained
results.

### 7. [Active Estimation of Multiplicative Faults in Dynamical Systems](http://arxiv.org/pdf/2506.23769v1)

Authors: Gabriel de Albuquerque Gleizer, Peyman Mohajerin Esfahani, Tamas Keviczky

This paper addresses the problem of estimating multiplicative fault signals
in linear time-invariant systems by processing its input and output variables,
as well as designing an input signal to maximize the accuracy of such
estimates. The proposed real-time fault estimator is based on a residual
generator used for fault detection and a multiple-output regressor generator,
which feed a moving-horizon linear regression that estimates the parameter
changes. Asymptotic performance guarantees are provided in the presence of
noise. Motivated by the performance bounds, an optimal input design problem is
formulated, for which we provide efficient algorithms and optimality bounds.
Numerical examples demonstrate the efficacy of our approach and the importance
of the optimal input design for accurate fault estimation.

### 8. [Statistical Modeling for Accurate Characterization of Doppler Effect in LEO-Terrestrial Networks](http://arxiv.org/pdf/2506.23817v1)

Authors: Islam M. Tanash, Risto Wichman, Nuria Gonzalez-Prelcic

Low Earth Orbit (LEO) satellite communication is a promising solution for
global wireless coverage, especially in underserved and remote areas. However,
the high relative velocity of LEO satellites induces significant Doppler shifts
that disrupt subcarrier orthogonality and degrade multicarrier system
performance. While the common time-varying Doppler shift can be compensated
relative to a reference point, the residual differential Doppler across users
within the coverage cell remains a significant challenge, causing severe
intercarrier interference. This paper presents a generalized analytical
framework for characterizing both the Doppler shift magnitude and the
differential Doppler in LEO systems. Unlike prior works limited by flat-Earth
assumptions or specific orbital configurations, our model incorporates Earth's
curvature and supports arbitrary elevation angles. Using spherical geometry, we
derive closed-form expressions for Doppler shift based on the central angle
between the satellite and ground users. We further provide a statistical
characterization of both the Doppler shift magnitude and the differential
Doppler in terms of their cumulative distribution function (CDF) and
probability density function (PDF) for uniformly distributed users within a
spherical cap cell. Additionally, we derive a tight upper bound for the Doppler
shift CDF and an exact expression for the maximum differential Doppler
experienced across the coverage region. To mitigate intra-cell Doppler
variation, we implement a user clustering technique that partitions the
coverage area based on a Doppler disparity threshold into spherical sub-cells,
ensuring compliance with 3GPP tolerances. Extensive simulations over realistic
satellite constellations validate our analysis and reveal the impact of
altitude, beamwidth, and satellite-user geometry on Doppler behavior.

### 9. [Orchestrated Couplings: A Time-Varying Edge Weight Framework for Efficient Event-Triggered Multiagent Networks](http://arxiv.org/pdf/2506.24017v1)

Authors: Emre Yildirim, Tansel Yucelen, Arman Sargolzaei

In this paper, we focus on reducing node-to-node information exchange in
distributed control of multiagent networks while improving the overall network
performance. Specifically, we consider a multiagent network that is composed of
leader and follower nodes over a time-varying, connected, and undirected graph.
In contrast to existing works on the event-triggered distributed control
literature, we propose a time-varying edge weight event-triggered control
framework. In this framework, each node dynamically adjusts its edge weights by
increasing them during the transient (active) phase and decreasing them during
the steady-state (idle) phase of the multiagent network. This not only reduces
the number of events in the network but also improves the performance (i.e.,
convergence speed and control effort) of the overall multiagent network.
System-theoretically, we first prove the closed-loop stability of the proposed
event-triggered distributed control framework, where we then show that this
framework does not exhibit a Zeno behavior. Finally, illustrative numerical
examples are provided to demonstrate the efficacy of this framework.

### 10. [Time Shift Governor-Guided MPC with Collision Cone CBFs for Safe Adaptive Cruise Control in Dynamic Environments](http://arxiv.org/pdf/2506.24083v1)

Authors: Robin Inho Kee, Taehyeun Kim, Anouck Girard, Ilya Kolmanovsky

This paper introduces a Time Shift Governor (TSG)-guided Model Predictive
Controller with Control Barrier Functions (CBFs)-based constraints for adaptive
cruise control (ACC). This MPC-CBF approach is defined for obstacle-free curved
road tracking, while following distance and obstacle avoidance constraints are
handled using standard CBFs and relaxed Collision Cone CBFs. In order to
address scenarios involving rapidly moving obstacles or rapidly changing
leading vehicle's behavior, the TSG augmentation is employed which alters the
target reference to enforce constraints. Simulation results demonstrate the
effectiveness of the TSG-guided MPC-CBF approach.

### Machine Learning (Statistics Category)

### 1. [Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift](http://arxiv.org/pdf/2506.23453v1)

Authors: Zhen Zhang, Xin Liu, Shaoli Wang, Jiaye Teng

Covariate shift occurs when the distribution of input features differs
between the training and testing phases. In covariate shift, estimating an
unknown function's moment is a classical problem that remains under-explored,
despite its common occurrence in real-world scenarios. In this paper, we
investigate the minimax lower bound of the problem when the source and target
distributions are known. To achieve the minimax optimal bound (up to a
logarithmic factor), we propose a two-stage algorithm. Specifically, it first
trains an optimal estimator for the function under the source distribution, and
then uses a likelihood ratio reweighting procedure to calibrate the moment
estimator. In practice, the source and target distributions are typically
unknown, and estimating the likelihood ratio may be unstable. To solve this
problem, we propose a truncated version of the estimator that ensures double
robustness and provide the corresponding upper bound. Extensive numerical
studies on synthetic examples confirm our theoretical findings and further
illustrate the effectiveness of our proposed method.

### 2. [Test of partial effects for Frechet regression on Bures-Wasserstein manifolds](http://arxiv.org/pdf/2506.23487v1)

Authors: Haoshu Xu, Hongzhe Li

We propose a novel test for assessing partial effects in Frechet regression
on Bures Wasserstein manifolds. Our approach employs a sample splitting
strategy: the first subsample is used to fit the Frechet regression model,
yielding estimates of the covariance matrices and their associated optimal
transport maps, while the second subsample is used to construct the test
statistic. We prove that this statistic converges in distribution to a weighted
mixture of chi squared components, where the weights correspond to the
eigenvalues of an integral operator defined by an appropriate RKHS kernel. We
establish that our procedure achieves the nominal asymptotic size and
demonstrate that its worst-case power converges uniformly to one. Through
extensive simulations and a real data application, we illustrate the test's
finite-sample accuracy and practical utility.

### 3. [Overparametrized models with posterior drift](http://arxiv.org/pdf/2506.23619v1)

Authors: Guillaume Coqueret, Martial Laguerre

This paper investigates the impact of posterior drift on out-of-sample
forecasting accuracy in overparametrized machine learning models. We document
the loss in performance when the loadings of the data generating process change
between the training and testing samples. This matters crucially in settings in
which regime changes are likely to occur, for instance, in financial markets.
Applied to equity premium forecasting, our results underline the sensitivity of
a market timing strategy to sub-periods and to the bandwidth parameters that
control the complexity of the model. For the average investor, we find that
focusing on holding periods of 15 years can generate very heterogeneous
returns, especially for small bandwidths. Large bandwidths yield much more
consistent outcomes, but are far less appealing from a risk-adjusted return
standpoint. All in all, our findings tend to recommend cautiousness when
resorting to large linear models for stock market predictions.

### 4. [Training of Spiking Neural Networks with Expectation-Propagation](http://arxiv.org/pdf/2506.23757v1)

Authors: Dan Yao, Steve McLaughlin, Yoann Altmann

In this paper, we propose a unifying message-passing framework for training
spiking neural networks (SNNs) using Expectation-Propagation. Our gradient-free
method is capable of learning the marginal distributions of network parameters
and simultaneously marginalizes nuisance parameters, such as the outputs of
hidden layers. This framework allows for the first time, training of discrete
and continuous weights, for deterministic and stochastic spiking networks,
using batches of training samples. Although its convergence is not ensured, the
algorithm converges in practice faster than gradient-based methods, without
requiring a large number of passes through the training data. The
classification and regression results presented pave the way for new efficient
training methods for deep Bayesian networks.

### 5. [The Trilemma of Truth in Large Language Models](http://arxiv.org/pdf/2506.23921v1)

Authors: Germans Savcisens, Tina Eliassi-Rad

We often attribute human characteristics to large language models (LLMs) and
claim that they "know" certain things. LLMs have an internal probabilistic
knowledge that represents information retained during training. How can we
assess the veracity of this knowledge? We examine two common methods for
probing the veracity of LLMs and discover several assumptions that are flawed.
To address these flawed assumptions, we introduce sAwMIL (short for Sparse
Aware Multiple-Instance Learning), a probing method that utilizes the internal
activations of LLMs to separate statements into true, false, and neither.
sAwMIL is based on multiple-instance learning and conformal prediction. We
evaluate sAwMIL on 5 validity criteria across 16 open-source LLMs, including
both default and chat-based variants, as well as on 3 new datasets. Among the
insights we provide are: (1) the veracity signal is often concentrated in the
third quarter of an LLM's depth; (2) truth and falsehood signals are not always
symmetric; (3) linear probes perform better on chat models than on default
models; (4) nonlinear probes may be required to capture veracity signals for
some LLMs with reinforcement learning from human feedback or knowledge
distillation; and (5) LLMs capture a third type of signal that is distinct from
true and false and is neither true nor false. These findings provide a reliable
method for verifying what LLMs "know" and how certain they are of their
probabilistic internal knowledge.

### 6. [Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime](http://arxiv.org/pdf/2506.24120v1)

Authors: Yuqing Wang, Shangding Gu

Data selection plays a crucial role in data-driven decision-making, including
in large language models (LLMs), and is typically task-dependent. Properties
such as data quality and diversity have been extensively studied and are known
to enhance model performance. However, it remains unclear whether there exist
other quantitative and general principles of data selection that can
consistently improve performance, especially for complex tasks with limited
prior knowledge. In this paper, we demonstrate that selecting more uniformly
distributed data can improve training efficiency while enhancing performance.
Specifically, we establish that more uniform (less biased) distribution leads
to a larger minimum pairwise distance between data points, denoted by
$h_{\min}$, and prove that a smaller $h_{\min}$ can slow down the training
dynamics of gradient descent (GD). Moreover, we theoretically show that the
approximation error of neural networks decreases as $h_{\min}$ increases. Our
analysis introduces a convergence framework for GD beyond the Neural Tangent
Kernel (NTK) regime, applicable to a broad class of architectures, including
transformers, without requiring Lipschitz smoothness. This framework further
provides theoretical justification for the use of residual connections and
function compositions in deep neural architectures. In the end, we conduct
comprehensive experiments for supervised fine-tuning across various settings,
including different optimization strategies, model sizes, and training
datasets. The results consistently demonstrate that selecting data by
maximizing pairwise distance significantly accelerates training and achieves
comparable or better performance in LLMs across diverse datasets. Code and
Datasets are available at the link:
https://github.com/SafeRL-Lab/data-uniformity.

### 7. [Sampling and Identity-Testing Without Approximate Tensorization of Entropy](http://arxiv.org/pdf/2506.23456v1)

Authors: William Gay, William He, Nicholas Kocurek, Ryan O'Donnell

Certain tasks in high-dimensional statistics become easier when the
underlying distribution satisfies a local-to-global property called approximate
tensorization of entropy (ATE). For example, the Glauber dynamics Markov chain
of an ATE distribution mixes fast and can produce approximate samples in a
small amount of time, since such a distribution satisfies a modified
log-Sobolev inequality. Moreover, identity-testing for an ATE distribution
requires few samples if the tester is given coordinate conditional access to
the unknown distribution, as shown by Blanca, Chen, \v{S}tefankovi\v{c}, and
Vigoda (COLT 2023).
  A natural class of distributions that do not satisfy ATE consists of mixtures
of (few) distributions that do satisfy ATE. We study the complexity of
identity-testing and sampling for these distributions. Our main results are the
following:
  1. We show fast mixing of Glauber dynamics from a data-based initialization,
with optimal sample complexity, for mixtures of distributions satisfying
modified log-Sobolev inequalities. This extends work of Huang, Koehler, Lee,
Mohanty, Rajaraman, Vuong, and Wu (STOC 2025, COLT 2025) for mixtures of
distributions satisfying Poincar\'e inequalities.
  2. Answering an open question posed by Blanca et al., we give efficient
identity-testers for mixtures of ATE distributions in the
coordinate-conditional sampling access model. We also give some simplifications
and improvements to the original algorithm of Blanca et al.

### 8. [Minimax and Bayes Optimal Best-arm Identification: Adaptive Experimental Design for Treatment Choice](http://arxiv.org/pdf/2506.24007v1)

Authors: Masahiro Kato

This study investigates adaptive experimental design for treatment choice,
also known as fixed-budget best-arm identification. We consider an adaptive
procedure consisting of a treatment-allocation phase followed by a
treatment-choice phase, and we design an adaptive experiment for this setup to
efficiently identify the best treatment arm, defined as the one with the
highest expected outcome. In our designed experiment, the treatment-allocation
phase consists of two stages. The first stage is a pilot phase, where we
allocate each treatment arm uniformly with equal proportions to eliminate
clearly suboptimal arms and estimate outcome variances. In the second stage, we
allocate treatment arms in proportion to the variances estimated in the first
stage. After the treatment-allocation phase, the procedure enters the
treatment-choice phase, where we choose the treatment arm with the highest
sample mean as our estimate of the best treatment arm. We prove that this
single design is simultaneously asymptotically minimax and Bayes optimal for
the simple regret, with upper bounds that match our lower bounds up to exact
constants. Therefore, our designed experiment achieves the sharp efficiency
limits without requiring separate tuning for minimax and Bayesian objectives.

### 9. [Faster Diffusion Models via Higher-Order Approximation](http://arxiv.org/pdf/2506.24042v1)

Authors: Gen Li, Yuchen Zhou, Yuting Wei, Yuxin Chen

In this paper, we explore provable acceleration of diffusion models without
any additional retraining. Focusing on the task of approximating a target data
distribution in $\mathbb{R}^d$ to within $\varepsilon$ total-variation
distance, we propose a principled, training-free sampling algorithm that
requires only the order of
  $$ d^{1+2/K} \varepsilon^{-1/K} $$
  score function evaluations (up to log factor) in the presence of accurate
scores, where $K$ is an arbitrarily large fixed integer. This result applies to
a broad class of target data distributions, without the need for assumptions
such as smoothness or log-concavity. Our theory is robust vis-a-vis inexact
score estimation, degrading gracefully as the score estimation error increases
-- without demanding higher-order smoothness on the score estimates as assumed
in previous work. The proposed algorithm draws insight from high-order ODE
solvers, leveraging high-order Lagrange interpolation and successive refinement
to approximate the integral derived from the probability flow ODE.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-01 PST.

### 1. [A multi-agent system based on HNC for domain-specific machine translation](https://www.nature.com/articles/s41598-025-03414-9)

Authors: Ming Li et al.

### 2. [How AI is used in FDA-authorized medical devices: a taxonomy across 1,016 authorizations](https://www.nature.com/articles/s41746-025-01800-1)

Authors: Rohan Singh et al.

### 3. [A study on classification based concurrent API calls and optimal model combination for tool augmented LLMs for AI agent](https://www.nature.com/articles/s41598-025-06469-w)

Authors: HeounMo Go et al.

### 4. [Dataset for Single Character Detection in Dongba Manuscripts](https://www.nature.com/articles/s41597-025-05434-6)

Authors: Yuqi Ma et al.

### 5. [Predict the degree of secondary structures of the encoding sequences in DNA storage by deep learning model](https://www.nature.com/articles/s41598-025-05717-3)

Authors: Wanmin Lin et al.

### 6. [Why we need mandatory safeguards for emotionally responsive AI](https://www.nature.com/articles/d41586-025-02031-w)

Authors: Ziv Ben-Zion

### 7. [Artificial intelligence-augmented smart grid architecture for cyber intrusion detection and mitigation in electric vehicle charging infrastructure](https://www.nature.com/articles/s41598-025-04984-4)

Authors: Ankita Sharma et al.

### 8. [A customized image editing framework for diverse prohibited and restricted products in illegal online transactions](https://www.nature.com/articles/s41598-025-07043-0)

Authors: Wenjin Liu et al.

### 9. [Towards decoding motor imagery from EEG signal using optimized back propagation neural network with honey badger algorithm](https://www.nature.com/articles/s41598-025-05423-0)

Authors: Zainab Hadi-Saleh et al.

### 10. [Multiclass semantic segmentation for prime disease detection with severity level identification in Citrus plant leaves](https://www.nature.com/articles/s41598-025-04758-y)

Authors: P. Dinesh et al.

### 11. [Interpretable longitudinal glaucoma visual field estimation deep learning system from fundus images and clinical narratives](https://www.nature.com/articles/s41746-025-01750-8)

Authors: Xiaoling Huang et al.

### 12. [A novel feature extractor based on constrained cross network for detecting sleep state](https://www.nature.com/articles/s41598-025-08627-6)

Authors: Chenlei Tian et al.

### 13. [Self-adaptive evolutionary neural networks for high-precision short-term electric load forecasting](https://www.nature.com/articles/s41598-025-05918-w)

Authors: Muhammad Abbas et al.

