# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-23 17:00:26.049368 PST.

### Artificial Intelligence

### 1. [TaxCalcBench: Evaluating Frontier Models on the Tax Calculation Task](http://arxiv.org/pdf/2507.16126v1)

Authors: Michael R. Bock, Kara Molisee, Zachary Ozer, Sumit Shah

Can AI file your taxes? Not yet. Calculating US personal income taxes is a
task that requires building an understanding of vast amounts of English text
and using that knowledge to carefully compute results. We propose TaxCalcBench,
a benchmark for determining models' abilities to calculate personal income tax
returns given all of the necessary information. Our experiment shows that
state-of-the-art models succeed in calculating less than a third of federal
income tax returns even on this simplified sample set. Our analysis concludes
that models consistently misuse tax tables, make errors in tax calculation, and
incorrectly determine eligibility. Our findings point to the need for
additional infrastructure to apply LLMs to the personal income tax calculation
task.

### 2. [ResearcherBench: Evaluating Deep AI Research Systems on the Frontiers of Scientific Inquiry](http://arxiv.org/pdf/2507.16280v1)

Authors: Tianze Xu, Pengrui Lu, Lyumanshan Ye, Xiangkun Hu, Pengfei Liu

The emergence of deep research systems presents significant capabilities in
problem-solving, extending from basic queries to sophisticated research tasks.
However, existing benchmarks primarily evaluate these systems as agents for web
retrieval and report generation, overlooking their potential to discover novel
insights on the frontiers of scientific research. To address this gap, we
introduce ResearcherBench, the first benchmark focused on evaluating the
capabilities of these advanced, agentic systems - which we refer to as Deep AI
Research Systems (DARS) - on frontier AI scientific questions. We compiled a
dataset of 65 research questions expertly selected from real-world scientific
scenarios such as laboratory discussions and interviews, spanning 35 different
AI subjects and categorized into three types: technical details, literature
review, and open consulting. Our dual evaluation framework combines rubric
assessment, which uses expert-designed criteria to evaluate insight quality,
with factual assessment, which measures citation accuracy (faithfulness) and
coverage (groundedness). We evaluated several leading commercial DARS and
baseline systems. Results show that OpenAI Deep Research and Gemini Deep
Research significantly outperform other systems, with particular strength in
open-ended consulting questions. Such capabilities represent a meaningful step
toward AI self-improvement, aligning with the vision of ASI for AI. We
open-source ResearcherBench to provide a standardized platform for promoting
the development of next-generation AI research assistants, hoping to foster a
new perspective in AI research evaluation for a novel pattern of scientific
collaboration: https://github.com/GAIR-NLP/ResearcherBench.

### 3. [Cross-Modal Distillation For Widely Differing Modalities](http://arxiv.org/pdf/2507.16296v1)

Authors: Cairong Zhao, Yufeng Jin, Zifan Song, Haonan Chen, Duoqian Miao, Guosheng Hu

Deep learning achieved great progress recently, however, it is not easy or
efficient to further improve its performance by increasing the size of the
model. Multi-modal learning can mitigate this challenge by introducing richer
and more discriminative information as input. To solve the problem of limited
access to multi-modal data at the time of use, we conduct multi-modal learning
by introducing a teacher model to transfer discriminative knowledge to a
student model during training. However, this knowledge transfer via
distillation is not trivial because the big domain gap between the widely
differing modalities can easily lead to overfitting. In this work, we introduce
a cross-modal distillation framework. Specifically, we find hard constrained
loss, e.g. l2 loss forcing the student being exact the same as the teacher, can
easily lead to overfitting in cross-modality distillation. To address this, we
propose two soft constrained knowledge distillation strategies at the feature
level and classifier level respectively. In addition, we propose a
quality-based adaptive weights module to weigh input samples via quantified
data quality, leading to robust model training. We conducted experiments on
speaker recognition and image classification tasks, and the results show that
our approach is able to effectively achieve knowledge transfer between the
commonly used and widely differing modalities of image, text, and speech.

### 4. [Mind the Gap: Evaluating the Representativeness of Quantitative Medical Language Reasoning LLM Benchmarks for African Disease Burdens](http://arxiv.org/pdf/2507.16322v1)

Authors: Fred Mutisya, Shikoh Gitau, Christine Syovata, Diana Oigara, Ibrahim Matende, Muna Aden, Munira Ali, Ryan Nyotu, Diana Marion, Job Nyangena, Nasubo Ongoma, Keith Mbae, Elizabeth Wamicha, Eric Mibuari, Jean Philbert Nsengemana, Talkmore Chidede

Introduction: Existing medical LLM benchmarks largely reflect examination
syllabi and disease profiles from high income settings, raising questions about
their validity for African deployment where malaria, HIV, TB, sickle cell
disease and other neglected tropical diseases (NTDs) dominate burden and
national guidelines drive care. Methodology: We systematically reviewed 31
quantitative LLM evaluation papers (Jan 2019 May 2025) identifying 19 English
medical QA benchmarks. Alama Health QA was developed using a retrieval
augmented generation framework anchored on the Kenyan Clinical Practice
Guidelines. Six widely used sets (AfriMedQA, MMLUMedical, PubMedQA, MedMCQA,
MedQAUSMLE, and guideline grounded Alama Health QA) underwent harmonized
semantic profiling (NTD proportion, recency, readability, lexical diversity
metrics) and blinded expert rating across five dimensions: clinical relevance,
guideline alignment, clarity, distractor plausibility, and language/cultural
fit. Results: Alama Health QA captured >40% of all NTD mentions across corpora
and the highest within set frequencies for malaria (7.7%), HIV (4.1%), and TB
(5.2%); AfriMedQA ranked second but lacked formal guideline linkage. Global
benchmarks showed minimal representation (e.g., sickle cell disease absent in
three sets) despite large scale. Qualitatively, Alama scored highest for
relevance and guideline alignment; PubMedQA lowest for clinical utility.
Discussion: Quantitative medical LLM benchmarks widely used in the literature
underrepresent African disease burdens and regulatory contexts, risking
misleading performance claims. Guideline anchored, regionally curated resources
such as Alama Health QA and expanded disease specific derivatives are essential
for safe, equitable model evaluation and deployment across African health
systems.

### 5. [Learning to Call: A Field Trial of a Collaborative Bandit Algorithm for Improved Message Delivery in Mobile Maternal Health](http://arxiv.org/pdf/2507.16356v1)

Authors: Arpan Dasgupta, Mizhaan Maniyar, Awadhesh Srivastava, Sanat Kumar, Amrita Mahale, Aparna Hedge, Arun Suggala, Karthikeyan Shanmugam, Aparna Taneja, Milind Tambe

Mobile health (mHealth) programs utilize automated voice messages to deliver
health information, particularly targeting underserved communities,
demonstrating the effectiveness of using mobile technology to disseminate
crucial health information to these populations, improving health outcomes
through increased awareness and behavioral change. India's Kilkari program
delivers vital maternal health information via weekly voice calls to millions
of mothers. However, the current random call scheduling often results in missed
calls and reduced message delivery. This study presents a field trial of a
collaborative bandit algorithm designed to optimize call timing by learning
individual mothers' preferred call times. We deployed the algorithm with around
$6500$ Kilkari participants as a pilot study, comparing its performance to the
baseline random calling approach. Our results demonstrate a statistically
significant improvement in call pick-up rates with the bandit algorithm,
indicating its potential to enhance message delivery and impact millions of
mothers across India. This research highlights the efficacy of personalized
scheduling in mobile health interventions and underscores the potential of
machine learning to improve maternal health outreach at scale.

### 6. [Identifying Pre-training Data in LLMs: A Neuron Activation-Based Detection Framework](http://arxiv.org/pdf/2507.16414v1)

Authors: Hongyi Tang, Zhihao Zhu, Yi Yang

The performance of large language models (LLMs) is closely tied to their
training data, which can include copyrighted material or private information,
raising legal and ethical concerns. Additionally, LLMs face criticism for
dataset contamination and internalizing biases. To address these issues, the
Pre-Training Data Detection (PDD) task was proposed to identify if specific
data was included in an LLM's pre-training corpus. However, existing PDD
methods often rely on superficial features like prediction confidence and loss,
resulting in mediocre performance. To improve this, we introduce NA-PDD, a
novel algorithm analyzing differential neuron activation patterns between
training and non-training data in LLMs. This is based on the observation that
these data types activate different neurons during LLM inference. We also
introduce CCNewsPDD, a temporally unbiased benchmark employing rigorous data
transformations to ensure consistent time distributions between training and
non-training data. Our experiments demonstrate that NA-PDD significantly
outperforms existing methods across three benchmarks and multiple LLMs.

### 7. [Learning Temporal Abstractions via Variational Homomorphisms in Option-Induced Abstract MDPs](http://arxiv.org/pdf/2507.16473v1)

Authors: Chang Li, Yaren Zhang, Haoran Lv, Qiong Cao, Chao Xue, Xiaodong He

Large Language Models (LLMs) have shown remarkable reasoning ability through
explicit Chain-of-Thought (CoT) prompting, but generating these step-by-step
textual explanations is computationally expensive and slow. To overcome this,
we aim to develop a framework for efficient, implicit reasoning, where the
model "thinks" in a latent space without generating explicit text for every
step. We propose that these latent thoughts can be modeled as
temporally-extended abstract actions, or options, within a hierarchical
reinforcement learning framework. To effectively learn a diverse library of
options as latent embeddings, we first introduce the Variational Markovian
Option Critic (VMOC), an off-policy algorithm that uses variational inference
within the HiT-MDP framework. To provide a rigorous foundation for using these
options as an abstract reasoning space, we extend the theory of continuous MDP
homomorphisms. This proves that learning a policy in the simplified, abstract
latent space, for which VMOC is suited, preserves the optimality of the
solution to the original, complex problem. Finally, we propose a cold-start
procedure that leverages supervised fine-tuning (SFT) data to distill human
reasoning demonstrations into this latent option space, providing a rich
initialization for the model's reasoning capabilities. Extensive experiments
demonstrate that our approach achieves strong performance on complex logical
reasoning benchmarks and challenging locomotion tasks, validating our framework
as a principled method for learning abstract skills for both language and
control.

### 8. [Novel Multi-Agent Action Masked Deep Reinforcement Learning for General Industrial Assembly Lines Balancing Problems](http://arxiv.org/pdf/2507.16635v1)

Authors: Ali Mohamed Ali, Luca Tirel, Hashim A. Hashim

Efficient planning of activities is essential for modern industrial assembly
lines to uphold manufacturing standards, prevent project constraint violations,
and achieve cost-effective operations. While exact solutions to such challenges
can be obtained through Integer Programming (IP), the dependence of the search
space on input parameters often makes IP computationally infeasible for
large-scale scenarios. Heuristic methods, such as Genetic Algorithms, can also
be applied, but they frequently produce suboptimal solutions in extensive
cases. This paper introduces a novel mathematical model of a generic industrial
assembly line formulated as a Markov Decision Process (MDP), without imposing
assumptions on the type of assembly line a notable distinction from most
existing models. The proposed model is employed to create a virtual environment
for training Deep Reinforcement Learning (DRL) agents to optimize task and
resource scheduling. To enhance the efficiency of agent training, the paper
proposes two innovative tools. The first is an action-masking technique, which
ensures the agent selects only feasible actions, thereby reducing training
time. The second is a multi-agent approach, where each workstation is managed
by an individual agent, as a result, the state and action spaces were reduced.
A centralized training framework with decentralized execution is adopted,
offering a scalable learning architecture for optimizing industrial assembly
lines. This framework allows the agents to learn offline and subsequently
provide real-time solutions during operations by leveraging a neural network
that maps the current factory state to the optimal action. The effectiveness of
the proposed scheme is validated through numerical simulations, demonstrating
significantly faster convergence to the optimal solution compared to a
comparable model-based approach.

### 9. [Adaptive Inventory Strategies using Deep Reinforcement Learning for Dynamic Agri-Food Supply Chains](http://arxiv.org/pdf/2507.16670v1)

Authors: Amandeep Kaur, Gyan Prakash

Agricultural products are often subject to seasonal fluctuations in
production and demand. Predicting and managing inventory levels in response to
these variations can be challenging, leading to either excess inventory or
stockouts. Additionally, the coordination among stakeholders at various level
of food supply chain is not considered in the existing body of literature. To
bridge these research gaps, this study focuses on inventory management of
agri-food products under demand and lead time uncertainties. By implementing
effective inventory replenishment policy results in maximize the overall profit
throughout the supply chain. However, the complexity of the problem increases
due to these uncertainties and shelf-life of the product, that makes
challenging to implement traditional approaches to generate optimal set of
solutions. Thus, the current study propose a novel Deep Reinforcement Learning
(DRL) algorithm that combines the benefits of both value- and policy-based DRL
approaches for inventory optimization under uncertainties. The proposed
algorithm can incentivize collaboration among stakeholders by aligning their
interests and objectives through shared optimization goal of maximizing
profitability along the agri-food supply chain while considering perishability,
and uncertainty simultaneously. By selecting optimal order quantities with
continuous action space, the proposed algorithm effectively addresses the
inventory optimization challenges. To rigorously evaluate this algorithm, the
empirical data from fresh agricultural products supply chain inventory is
considered. Experimental results corroborate the improved performance of the
proposed inventory replenishment policy under stochastic demand patterns and
lead time scenarios. The research findings hold managerial implications for
policymakers to manage the inventory of agricultural products more effectively
under uncertainty.

### 10. [Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints](http://arxiv.org/pdf/2507.16727v1)

Authors: Zhenyun Yin, Shujie Wang, Xuhong Wang, Xingjun Ma, Yinchun Wang

Improving the reliability of large language models (LLMs) is critical for
deploying them in real-world scenarios. In this paper, we propose
\textbf{Deliberative Searcher}, the first framework to integrate certainty
calibration with retrieval-based search for open-domain question answering. The
agent performs multi-step reflection and verification over Wikipedia data and
is trained with a reinforcement learning algorithm that optimizes for accuracy
under a soft reliability constraint. Empirical results show that proposed
method improves alignment between model confidence and correctness, leading to
more trustworthy outputs. This paper will be continuously updated.

### Hardware Architecture

### 1. [A Sparsity-Aware Autonomous Path Planning Accelerator with HW/SW Co-Design and Multi-Level Dataflow Optimization](http://arxiv.org/pdf/2507.16177v1)

Authors: Yifan Zhang, Xiaoyu Niu, Hongzheng Tian, Yanjun Zhang, Bo Yu, Shaoshan Liu, Sitao Huang

Path planning is critical for autonomous driving, generating smooth,
collision-free, feasible paths based on perception and localization inputs.
However, its computationally intensive nature poses significant challenges for
resource-constrained autonomous driving hardware. This paper presents an
end-to-end FPGA-based acceleration framework targeting the quadratic
programming (QP), core of optimization-based path planning. We employ a
hardware-friendly alternating direction method of multipliers (ADMM) for QP
solving and a parallelizable preconditioned conjugate gradient (PCG) method for
linear systems. By analyzing sparse matrix patterns, we propose customized
storage schemes and efficient sparse matrix multiplication units, significantly
reducing resource usage and accelerating matrix operations. Our multi-level
dataflow optimization strategy incorporates intra-operator parallelization and
pipelining, inter-operator fine-grained pipelining, and CPU-FPGA system-level
task mapping. Implemented on the AMD ZCU102 platform, our framework achieves
state-of-the-art latency and energy efficiency, including 1.48x faster
performance than the best FPGA-based design, 2.89x over an Intel i7-11800H CPU,
5.62x over an ARM Cortex-A57 embedded CPU, and 1.56x over a state-of-the-art
GPU solution, along with a 2.05x throughput improvement over existing
FPGA-based designs.

### 2. [Hourglass Sorting: A novel parallel sorting algorithm and its implementation](http://arxiv.org/pdf/2507.16326v1)

Authors: Daniel Bascones, Borja Morcillo

Sorting is one of the fundamental problems in computer science. Playing a
role in many processes, it has a lower complexity bound imposed by
$\mathcal{O}(n\log{n})$ when executing on a sequential machine. This limit can
be brought down to sub-linear times thanks to parallelization techniques that
increase the number of comparisons done in parallel. This, however, increases
the cost of implementation, which limits the application of such techniques.
Moreover, as the size of the arrays increases, a bottleneck arises in moving
the vast quantities of data required at the input, and generated at the output
of such sorter. This might impose time requirements much stricter than those of
the sorting itself. In this paper, a novel parallel sorter is proposed for the
specific case where the input is parallel, but the output is serial. The design
is then implemented and verified on an FPGA within the context of a quantum
LDPC decoder. A latency of $\log{n}$ is achieved for the output of the first
element, after which the rest stream out for a total sorting time of
$n+\log{n}$. Contrary to other parallel sorting methods, clock speed does not
degrade with $n$, and resources scale linearly with input size.

### 3. [ApproxGNN: A Pretrained GNN for Parameter Prediction in Design Space Exploration for Approximate Computing](http://arxiv.org/pdf/2507.16379v1)

Authors: Ondrej Vlcek, Vojtech Mrazek

Approximate computing offers promising energy efficiency benefits for
error-tolerant applications, but discovering optimal approximations requires
extensive design space exploration (DSE). Predicting the accuracy of circuits
composed of approximate components without performing complete synthesis
remains a challenging problem. Current machine learning approaches used to
automate this task require retraining for each new circuit configuration,
making them computationally expensive and time-consuming. This paper presents
ApproxGNN, a construction methodology for a pre-trained graph neural network
model predicting QoR and HW cost of approximate accelerators employing
approximate adders from a library. This approach is applicable in DSE for
assignment of approximate components to operations in accelerator. Our approach
introduces novel component feature extraction based on learned embeddings
rather than traditional error metrics, enabling improved transferability to
unseen circuits. ApproxGNN models can be trained with a small number of
approximate components, supports transfer to multiple prediction tasks,
utilizes precomputed embeddings for efficiency, and significantly improves
accuracy of the prediction of approximation error. On a set of image
convolutional filters, our experimental results demonstrate that the proposed
embeddings improve prediction accuracy (mean square error) by 50% compared to
conventional methods. Furthermore, the overall prediction accuracy is 30%
better than statistical machine learning approaches without fine-tuning and 54%
better with fast finetuning.

### 4. [Ironman: Accelerating Oblivious Transfer Extension for Privacy-Preserving AI with Near-Memory Processing](http://arxiv.org/pdf/2507.16391v1)

Authors: Chenqi Lin, Kang Yang, Tianshi Xu, Ling Liang, Yufei Wang, Zhaohui Chen, Runsheng Wang, Mingyu Gao, Meng Li

With the wide application of machine learning (ML), privacy concerns arise
with user data as they may contain sensitive information. Privacy-preserving ML
(PPML) based on cryptographic primitives has emerged as a promising solution in
which an ML model is directly computed on the encrypted data to provide a
formal privacy guarantee. However, PPML frameworks heavily rely on the
oblivious transfer (OT) primitive to compute nonlinear functions. OT mainly
involves the computation of single-point correlated OT (SPCOT) and learning
parity with noise (LPN) operations. As OT is still computed extensively on
general-purpose CPUs, it becomes the latency bottleneck of modern PPML
frameworks.
  In this paper, we propose a novel OT accelerator, dubbed Ironman, to
significantly increase the efficiency of OT and the overall PPML framework. We
observe that SPCOT is computation-bounded, and thus propose a hardware-friendly
SPCOT algorithm with a customized accelerator to improve SPCOT computation
throughput. In contrast, LPN is memory-bandwidth-bounded due to irregular
memory access patterns. Hence, we further leverage the near-memory processing
(NMP) architecture equipped with memory-side cache and index sorting to improve
effective memory bandwidth. With extensive experiments, we demonstrate Ironman
achieves a 39.2-237.4 times improvement in OT throughput across different NMP
configurations compared to the full-thread CPU implementation. For different
PPML frameworks, Ironman demonstrates a 2.1-3.4 times reduction in end-to-end
latency for both CNN and Transformer models.

### 5. [Augmenting Von Neumann's Architecture for an Intelligent Future](http://arxiv.org/pdf/2507.16628v1)

Authors: Rajpreet Singh, Vidhi Kothari

This work presents a novel computer architecture that extends the Von Neumann
model with a dedicated Reasoning Unit (RU) to enable native artificial general
intelligence capabilities. The RU functions as a specialized co-processor that
executes symbolic inference, multi-agent coordination, and hybrid
symbolic-neural computation as fundamental architectural primitives. This
hardware-embedded approach allows autonomous agents to perform goal-directed
planning, dynamic knowledge manipulation, and introspective reasoning directly
within the computational substrate at system scale. The architecture
incorporates a reasoning-specific instruction set architecture, parallel
symbolic processing pipelines, agent-aware kernel abstractions, and a unified
memory hierarchy that seamlessly integrates cognitive and numerical workloads.
Through systematic co-design across hardware, operating system, and agent
runtime layers, this architecture establishes a computational foundation where
reasoning, learning, and adaptation emerge as intrinsic execution properties
rather than software abstractions, potentially enabling the development of
general-purpose intelligent machines.

### 6. [MTU: The Multifunction Tree Unit in zkSpeed for Accelerating HyperPlonk](http://arxiv.org/pdf/2507.16793v1)

Authors: Jianqiao Mo, Alhad Daftardar, Joey Ah-kiow, Kaiyue Guo, Benedikt Bünz, Siddharth Garg, Brandon Reagen

Zero-Knowledge Proofs (ZKPs) are critical for privacy preservation and
verifiable computation. Many ZKPs rely on kernels such as the SumCheck protocol
and Merkle Tree commitments, which enable their security properties. These
kernels exhibit balanced binary tree computational patterns, which enable
efficient hardware acceleration. Prior work has investigated accelerating these
kernels as part of an overarching ZKP protocol; however, a focused study of how
to best exploit the underlying tree pattern for hardware efficiency remains
limited. We conduct a systematic evaluation of these tree-based workloads under
different traversal strategies, analyzing performance on multi-threaded CPUs
and a hardware accelerator, the Multifunction Tree Unit (MTU). We introduce a
hardware-friendly Hybrid Traversal for binary tree that improves parallelism
and scalability while significantly reducing memory traffic on hardware. Our
results show that MTU achieves up to 1478$\times$ speedup over CPU at DDR-level
bandwidth and that our hybrid traversal outperforms as standalone approach by
up to 3$\times$. These findings offer practical guidance for designing
efficient hardware accelerators for ZKP workloads with binary tree structures.

### 7. [RealBench: Benchmarking Verilog Generation Models with Real-World IP Designs](http://arxiv.org/pdf/2507.16200v1)

Authors: Pengwei Jin, Di Huang, Chongxiao Li, Shuyao Cheng, Yang Zhao, Xinyao Zheng, Jiaguo Zhu, Shuyi Xing, Bohan Dou, Rui Zhang, Zidong Du, Qi Guo, Xing Hu

The automatic generation of Verilog code using Large Language Models (LLMs)
has garnered significant interest in hardware design automation. However,
existing benchmarks for evaluating LLMs in Verilog generation fall short in
replicating real-world design workflows due to their designs' simplicity,
inadequate design specifications, and less rigorous verification environments.
To address these limitations, we present RealBench, the first benchmark aiming
at real-world IP-level Verilog generation tasks. RealBench features complex,
structured, real-world open-source IP designs, multi-modal and formatted design
specifications, and rigorous verification environments, including 100% line
coverage testbenches and a formal checker. It supports both module-level and
system-level tasks, enabling comprehensive assessments of LLM capabilities.
Evaluations on various LLMs and agents reveal that even one of the
best-performing LLMs, o1-preview, achieves only a 13.3% pass@1 on module-level
tasks and 0% on system-level tasks, highlighting the need for stronger Verilog
generation models in the future. The benchmark is open-sourced at
https://github.com/IPRC-DIP/RealBench.

### 8. [Custom Algorithm-based Fault Tolerance for Attention Layers in Transformers](http://arxiv.org/pdf/2507.16676v1)

Authors: Vasileios Titopoulos, Kosmas Alexandridis, Giorgos Dimitrakopoulos

Transformers and large language models (LLMs), powered by the attention
mechanism, have transformed numerous AI applications, driving the need for
specialized hardware accelerators. A major challenge in these accelerators is
efficiently detecting errors caused by random hardware faults. Traditional
algorithm-based fault tolerance (ABFT) techniques verify individual matrix
multiplications but fall short in handling the full attention mechanism,
particularly due to intermediate softmax normalization. This work proposes
Flash-ABFT, a novel method that computes an online checksum across the entire
three-matrix product of query, key and value matrices, of an attention layer,
including the softmax operation, with a single check. This approach
significantly reduces overhead by eliminating redundant checks while
maintaining high fault-detection accuracy. Experimental results demonstrate
that Flash-ABFT incurs only 5.3% hardware area overhead and less than 1.9%
energy overhead, making it a cost-effective and robust solution for error
detection in attention accelerators.

### 9. [SVAgent: AI Agent for Hardware Security Verification Assertion](http://arxiv.org/pdf/2507.16203v1)

Authors: Rui Guo, Avinash Ayalasomayajula, Henian Li, Jingbo Zhou, Sujan Kumar Saha, Farimah Farahmandi

Verification using SystemVerilog assertions (SVA) is one of the most popular
methods for detecting circuit design vulnerabilities. However, with the
globalization of integrated circuit design and the continuous upgrading of
security requirements, the SVA development model has exposed major limitations.
It is not only inefficient in development, but also unable to effectively deal
with the increasing number of security vulnerabilities in modern complex
integrated circuits. In response to these challenges, this paper proposes an
innovative SVA automatic generation framework SVAgent. SVAgent introduces a
requirement decomposition mechanism to transform the original complex
requirements into a structured, gradually solvable fine-grained problem-solving
chain. Experiments have shown that SVAgent can effectively suppress the
influence of hallucinations and random answers, and the key evaluation
indicators such as the accuracy and consistency of the SVA are significantly
better than existing frameworks. More importantly, we successfully integrated
SVAgent into the most mainstream integrated circuit vulnerability assessment
framework and verified its practicality and reliability in a real engineering
design environment.

### 10. [Optimization of DNN-based HSI Segmentation FPGA-based SoC for ADS: A Practical Approach](http://arxiv.org/pdf/2507.16556v1)

Authors: Jon Gutiérrez-Zaballa, Koldo Basterretxea, Javier Echanobe

The use of HSI for autonomous navigation is a promising research field aimed
at improving the accuracy and robustness of detection, tracking, and scene
understanding systems based on vision sensors. Combining advanced computer
algorithms, such as DNNs, with small-size snapshot HSI cameras enhances the
reliability of these systems. HSI overcomes intrinsic limitations of greyscale
and RGB imaging in depicting physical properties of targets, particularly
regarding spectral reflectance and metamerism. Despite promising results in
HSI-based vision developments, safety-critical systems like ADS demand strict
constraints on latency, resource consumption, and security, motivating the
shift of ML workloads to edge platforms. This involves a thorough
software/hardware co-design scheme to distribute and optimize the tasks
efficiently among the limited resources of computing platforms. With respect to
inference, the over-parameterized nature of DNNs poses significant
computational challenges for real-time on-the-edge deployment. In addition, the
intensive data preprocessing required by HSI, which is frequently overlooked,
must be carefully managed in terms of memory arrangement and inter-task
communication to enable an efficient integrated pipeline design on a SoC. This
work presents a set of optimization techniques for the practical co-design of a
DNN-based HSI segmentation processor deployed on a FPGA-based SoC targeted at
ADS, including key optimizations such as functional software/hardware task
distribution, hardware-aware preprocessing, ML model compression, and a
complete pipelined deployment. Applied compression techniques significantly
reduce the complexity of the designed DNN to 24.34% of the original operations
and to 1.02% of the original number of parameters, achieving a 2.86x speed-up
in the inference task without noticeable degradation of the segmentation
accuracy.

### Computational Complexity

### 1. [Computational aspects of the trace norm contraction coefficient](http://arxiv.org/pdf/2507.16737v1)

Authors: Idris Delsol, Omar Fawzi, Jan Kochanowski, Akshay Ramachandran

We show that approximating the trace norm contraction coefficient of a
quantum channel within a constant factor is NP-hard. Equivalently, this shows
that determining the optimal success probability for encoding a bit in a
quantum system undergoing noise is NP-hard. This contrasts with the classical
analogue of this problem that can clearly by solved efficiently. Our hardness
results also hold for deciding if the contraction coefficient is equal to 1. As
a consequence, we show that deciding if a non-commutative graph has an
independence number of at least 2 is NP-hard. In addition, we establish a
converging hierarchy of semidefinite programming upper bounds on the
contraction coefficient.

### 2. [Constructing material network representations for intelligent amorphous alloys design](http://arxiv.org/pdf/2507.16336v1)

Authors: S. -Y. Zhang, J. Tian, S. -L. Liu, H. -M. Zhang, H. -Y. Bai, Y. -C. Hu, W. -H. Wang

Designing high-performance amorphous alloys is demanding for various
applications. But this process intensively relies on empirical laws and
unlimited attempts. The high-cost and low-efficiency nature of the traditional
strategies prevents effective sampling in the enormous material space. Here, we
propose material networks to accelerate the discovery of binary and ternary
amorphous alloys. The network topologies reveal hidden material candidates that
were obscured by traditional tabular data representations. By scrutinizing the
amorphous alloys synthesized in different years, we construct dynamical
material networks to track the history of the alloy discovery. We find that
some innovative materials designed in the past were encoded in the networks,
demonstrating their predictive power in guiding new alloy design. These
material networks show physical similarities with several real-world networks
in our daily lives. Our findings pave a new way for intelligent materials
design, especially for complex alloys.

### 3. [An unconditional lower bound for the active-set method in convex quadratic maximization](http://arxiv.org/pdf/2507.16648v1)

Authors: Eleon Bach, Yann Disser, Sophie Huiberts, Nils Mosis

We prove that the active-set method needs an exponential number of iterations
in the worst-case to maximize a convex quadratic function subject to linear
constraints, regardless of the pivot rule used. This substantially improves
over the best previously known lower bound [IPCO 2025], which needs objective
functions of polynomial degrees $\omega(\log d)$ in dimension $d$, to a bound
using a convex polynomial of degree 2. In particular, our result firmly
resolves the open question [IPCO 2025] of whether a constant degree suffices,
and it represents significant progress towards linear objectives, where the
active-set method coincides with the simplex method and a lower bound for all
pivot rules would constitute a major breakthrough.
  Our result is based on a novel extended formulation, recursively constructed
using deformed products. Its key feature is that it projects onto a polygonal
approximation of a parabola while preserving all of its exponentially many
vertices. We define a quadratic objective that forces the active-set method to
follow the parabolic boundary of this projection, without allowing any
shortcuts along chords corresponding to edges of its full-dimensional preimage.

### Computational Engineering

### 1. [Computational design of personalized drugs via robust optimization under uncertainty](http://arxiv.org/pdf/2507.16470v1)

Authors: Rabia Altunay, Jarkko Suuronen, Eero Immonen, Lassi Roininen, Jari Hämäläinen

Effective disease treatment often requires precise control of the release of
the active pharmaceutical ingredient (API). In this work, we present a
computational inverse design approach to determine the optimal drug composition
that yields a target release profile. We assume that the drug release is
governed by the Noyes-Whitney model, meaning that dissolution occurs at the
surface of the drug. Our inverse design method is based on topology
optimization. The method optimizes the drug composition based on the target
release profile, considering the drug material parameters and the shape of the
final drug. Our method is non-parametric and applicable to arbitrary drug
shapes. The inverse design method is complemented by robust topology
optimization, which accounts for the random drug material parameters. We use
the stochastic reduced-order method (SROM) to propagate the uncertainty in the
dissolution model. Unlike Monte Carlo methods, SROM requires fewer samples and
improves computational performance. We apply our method to designing drugs with
several target release profiles. The numerical results indicate that the
release profiles of the designed drugs closely resemble the target profiles.
The SROM-based drug designs exhibit less uncertainty in their release profiles,
suggesting that our method is a convincing approach for uncertainty-aware drug
design.

### 2. [Multi-objective Portfolio Optimization Via Gradient Descent](http://arxiv.org/pdf/2507.16717v1)

Authors: Christian Oliva, Pedro R. Ventura, Luis F. Lago-Fernández

Traditional approaches to portfolio optimization, often rooted in Modern
Portfolio Theory and solved via quadratic programming or evolutionary
algorithms, struggle with scalability or flexibility, especially in scenarios
involving complex constraints, large datasets and/or multiple conflicting
objectives. To address these challenges, we introduce a benchmark framework for
multi-objective portfolio optimization (MPO) using gradient descent with
automatic differentiation. Our method supports any optimization objective, such
as minimizing risk measures (e.g., CVaR) or maximizing Sharpe ratio, along with
realistic constraints, such as tracking error limits, UCITS regulations, or
asset group restrictions. We have evaluated our framework across six
experimental scenarios, from single-objective setups to complex multi-objective
cases, and have compared its performance against standard solvers like CVXPY
and SKFOLIO. Our results show that our method achieves competitive performance
while offering enhanced flexibility for modeling multiple objectives and
constraints. We aim to provide a practical and extensible tool for researchers
and practitioners exploring advanced portfolio optimization problems in
real-world conditions.

### Computational Geometry

### 1. [Analysis of Design Algorithms and Fabrication of a Graph-based Double-curvature Structure with Planar Hexagonal Panels](http://arxiv.org/pdf/2507.16171v1)

Authors: Mehdi Gorjian, Gregory A. Luhan, Stephen M. Caffey

This paper presents a novel algorithmic framework for the computational
design, simulation, and fabrication of a hexagonal grid-based double-curvature
structure with planar hexagonal panels. The journey begins with constructing a
robust data structure through the meticulous subdivision of an equilateral
triangle surface, forming a foundational triangular grid. This grid is the
basis for a graph that encapsulates hexagons, laying the groundwork for
simulating dynamic interactions and form-finding. The developed algorithm
ensures a well-structured hexagonal grid data representation, and the
experimental results showcase the successful implementation of the algorithm,
leading to the fabrication of planar hexagons mirroring physics-generated mesh
surfaces.

### 2. [Improved Wake-Up Time For Euclidean Freeze-Tag Problem](http://arxiv.org/pdf/2507.16269v1)

Authors: Sharareh Alipour, Arash Ahadi, Kajal Baghestani

The Freeze-Tag Problem (FTP) involves activating a set of initially asleep
robots as quickly as possible, starting from a single awake robot. Once
activated, a robot can assist in waking up other robots. Each active robot
moves at unit speed. The objective is to minimize the makespan, i.e., the time
required to activate the last robot. A key performance measure is the wake-up
ratio, defined as the maximum time needed to activate any number of robots in
any primary positions. This work focuses on the geometric (Euclidean) version
of FTP in $\mathbb{R}^d$ under the $\ell_p$ norm, where the initial distance
between each asleep robot and the single active robot is at most 1. For
$(\mathbb{R}^2, \ell_2)$, we improve the previous upper bound of 4.62 ([7],
CCCG 2024) to 4.31. Note that it is known that 3.82 is a lower bound for the
wake-up ratio. In $\mathbb{R}^3$, we propose a new strategy that achieves a
wake-up ratio of 12 for $(\mathbb{R}^3, \ell_1)$ and 12.76 for $(\mathbb{R}^3,
\ell_2)$, improving upon the previous bounds of 13 and $13\sqrt{3}$,
respectively, reported in [2].

### Computation and Language

### 1. [BIDWESH: A Bangla Regional Based Hate Speech Detection Dataset](http://arxiv.org/pdf/2507.16183v1)

Authors: Azizul Hakim Fayaz, MD. Shorif Uddin, Rayhan Uddin Bhuiyan, Zakia Sultana, Md. Samiul Islam, Bidyarthi Paul, Tashreef Muhammad, Shahriar Manzoor

Hate speech on digital platforms has become a growing concern globally,
especially in linguistically diverse countries like Bangladesh, where regional
dialects play a major role in everyday communication. Despite progress in hate
speech detection for standard Bangla, Existing datasets and systems fail to
address the informal and culturally rich expressions found in dialects such as
Barishal, Noakhali, and Chittagong. This oversight results in limited detection
capability and biased moderation, leaving large sections of harmful content
unaccounted for. To address this gap, this study introduces BIDWESH, the first
multi-dialectal Bangla hate speech dataset, constructed by translating and
annotating 9,183 instances from the BD-SHS corpus into three major regional
dialects. Each entry was manually verified and labeled for hate presence, type
(slander, gender, religion, call to violence), and target group (individual,
male, female, group), ensuring linguistic and contextual accuracy. The
resulting dataset provides a linguistically rich, balanced, and inclusive
resource for advancing hate speech detection in Bangla. BIDWESH lays the
groundwork for the development of dialect-sensitive NLP tools and contributes
significantly to equitable and context-aware content moderation in low-resource
language settings.

### 2. [Do Large Language Models Have a Planning Theory of Mind? Evidence from MindGames: a Multi-Step Persuasion Task](http://arxiv.org/pdf/2507.16196v1)

Authors: Jared Moore, Ned Cooper, Rasmus Overmark, Beba Cibralic, Nick Haber, Cameron R. Jones

Recent evidence suggests Large Language Models (LLMs) display Theory of Mind
(ToM) abilities. Most ToM experiments place participants in a spectatorial
role, wherein they predict and interpret other agents' behavior. However, human
ToM also contributes to dynamically planning action and strategically
intervening on others' mental states. We present MindGames: a novel `planning
theory of mind' (PToM) task which requires agents to infer an interlocutor's
beliefs and desires to persuade them to alter their behavior. Unlike previous
evaluations, we explicitly evaluate use cases of ToM. We find that humans
significantly outperform o1-preview (an LLM) at our PToM task (11% higher;
$p=0.006$). We hypothesize this is because humans have an implicit causal model
of other agents (e.g., they know, as our task requires, to ask about people's
preferences). In contrast, o1-preview outperforms humans in a baseline
condition which requires a similar amount of planning but minimal mental state
inferences (e.g., o1-preview is better than humans at planning when already
given someone's preferences). These results suggest a significant gap between
human-like social reasoning and LLM abilities.

### 3. [WakenLLM: A Fine-Grained Benchmark for Evaluating LLM Reasoning Potential and Reasoning Process Stability](http://arxiv.org/pdf/2507.16199v1)

Authors: Zipeng Ling, Yuehao Tang, Shuliang Liu, Junqi Yang, Shenghong Fu, Yao Wan, Kejia Huang, Zhichao Hou, Xuming Hu

Large Language Models (LLMs) frequently output the label \emph{Unknown}, yet
current evaluations focus almost exclusively on whether such answers are
\emph{honest} rather than why they arise. This blurs two distinct cases: (i) an
input that is genuinely indeterminate and (ii) a solvable problem that the
model fails to resolve. We call this phenomenon \emph{Vague Perception}. And
thus we introduce a framework that quantifies the proportion of \emph{Unknown}
responses attributable to model incapacity and tests whether guided stimulation
can convert them into either correct (\emph{Known}) or intrinsically
indeterminate outcomes. By separating these sources of uncertainty, our method
provides a clearer picture of LLM reasoning limits and their potential for
improvement. As we get a theoretical accuracy of reasoning task on different
LLMs, we apply different methods to test whether the model can reach the
accuracy given a baseline framework. Our work is meaningful in exploring the
true reasoning ability of LLMs and providing a new perspective on solving the
\emph{Vague Perception} phenomenon.

### 4. [FinResearchBench: A Logic Tree based Agent-as-a-Judge Evaluation Framework for Financial Research Agents](http://arxiv.org/pdf/2507.16248v1)

Authors: Run Sun, Zuo Bai, Wentao Zhang, Yuxiang Zhang, Li Zhao, Shan Sun, Zhengwen Qiu

Recently, AI agents are rapidly evolving in intelligence and widely used in
professional research applications, such as STEM, software development,
finance, etc. Among these AI agents, deep research agent is a key category as
it can perform long-horizon tasks and solve problems of greater complexity.
However, there are few evaluation frameworks and benchmarks that systematically
and automatically investigate the capabilities of these research agents.
Furthermore, financial research problems have distinct complexity and subtlety.
To fill in the gap, we propose FinResearchBench, which is a logic tree based
Agent-as-a-Judge and targets specifically for the financial research agents. It
provides a comprehensive and automatic assessment of the research agents across
7 key types of tasks in the financial research domain. The contributions of
this work are two-folded: (1) the first and innovative Agent-as-a-Judge system
that extracts the logic tree of the research outcome and uses it as the
intermediate information to present a comprehensive, reliable and robust
evaluation; (2) finance oriented that it covers 70 typical financial research
questions, spreading across 7 frequently encountered types of tasks in the
domain.

### 5. [iShumei-Chinchunmei at SemEval-2025 Task 4: A balanced forgetting and retention multi-task framework using effective unlearning loss](http://arxiv.org/pdf/2507.16263v1)

Authors: Yujian Sun, Tian Li

As the Large Language Model (LLM) gains widespread adoption, increasing
attention has been given to the challenge of making LLM forget non-compliant
data memorized during its pre-training. Machine Unlearning focuses on
efficiently erasing sensitive information from LLM under limited computational
resources. To advance research in this area, SemEval 2025 Task 4: "Unlearning
Sensitive Content from Large Language Models" introduces three unlearning
datasets and establishes a benchmark by evaluating both forgetting
effectiveness and the preservation of standard capabilities. In this work, we
propose a more controllable forgetting loss, Effective Unlearning Loss, and
explore its integration with various techniques to achieve more efficient and
controlled unlearning. Our system ultimately ranked 5th on the competition
leaderboard.

### 6. [Beyond Isolated Dots: Benchmarking Structured Table Construction as Deep Knowledge Extraction](http://arxiv.org/pdf/2507.16271v1)

Authors: Tianyun Zhong, Guozhao Mo, Yanjiang Liu, Yihan Chen, Lingdi Kong, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Le Sun

With the emergence of large language models (LLMs), there is an expectation
that LLMs can effectively extract explicit information from complex real-world
documents (e.g., papers, reports). However, most LLMs generate paragraph-style
answers that are chaotic, disorganized, and untraceable. To bridge this gap, we
introduce the Arranged and Organized Extraction Benchmark (AOE), a new
bilingual benchmark with data and documents of varying lengths designed to
systematically evaluate the ability of LLMs to comprehend fragmented documents
and reconstruct isolated information into one organized table. Unlike
conventional text-to-table tasks, which rely on fixed schema and narrow task
domains, AOE includes 11 carefully crafted tasks across three diverse domains,
requiring models to generate context-specific schema tailored to varied input
queries. In the experiment, we evaluated both open-source and closed-source
state-of-the-art LLMs. The results show that even the most advanced models
struggled significantly. The benchmark is available at
https://huggingface.co/datasets/tianyumyum/AOE.

### 7. [Language Detection by Means of the Minkowski Norm: Identification Through Character Bigrams and Frequency Analysis](http://arxiv.org/pdf/2507.16284v1)

Authors: Paul-Andrei Pogăcean, Sanda-Maria Avram

The debate surrounding language identification has gained renewed attention
in recent years, especially with the rapid evolution of AI-powered language
models. However, the non-AI-based approaches to language identification have
been overshadowed. This research explores a mathematical implementation of an
algorithm for language determinism by leveraging monograms and bigrams
frequency rankings derived from established linguistic research. The datasets
used comprise texts varying in length, historical period, and genre, including
short stories, fairy tales, and poems. Despite these variations, the method
achieves over 80\% accuracy on texts shorter than 150 characters and reaches
100\% accuracy for longer texts and older writings. These results demonstrate
that classical frequency-based approaches remain effective and scalable
alternatives to AI-driven models for language detection.

### 8. [SpeLLM: Character-Level Multi-Head Decoding](http://arxiv.org/pdf/2507.16323v1)

Authors: Amit Ben-Artzy, Roy Schwartz

Scaling LLM vocabulary is often used to reduce input sequence length and
alleviate attention's quadratic cost. Yet, current LLM architectures impose a
critical bottleneck to this procedure: the output projection layer scales
linearly with vocabulary size, rendering substantial expansion impractical. We
propose SpeLLM, a method that decouples input and output vocabularies by
predicting character-level strings through multiple output heads. In SpeLLM,
each of the $k$ linear heads predicts a single character simultaneously,
enabling the model to represent a much larger output space using smaller,
independent linear heads. We present a self-distillation approach for
converting a standard LLM to a SpeLLM. Our experiments with four pre-trained
LLMs show their SpeLLM variants achieve competitive performance on downstream
tasks while reducing runtime by 5.1% on average across models. Our approach
provides a potential avenue for reducing LLM costs, while increasing support
for underrepresented languages and domains.

### 9. [Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny](http://arxiv.org/pdf/2507.16331v1)

Authors: Chuanhao Yan, Fengdi Che, Xuhan Huang, Xu Xu, Xin Li, Yizhi Li, Xingwei Qu, Jingzhe Shi, Zhuangzhuang He, Chenghua Lin, Yaodong Yang, Binhang Yuan, Hang Zhao, Yu Qiao, Bowen Zhou, Jie Fu

Existing informal language-based (e.g., human language) Large Language Models
(LLMs) trained with Reinforcement Learning (RL) face a significant challenge:
their verification processes, which provide crucial training signals, are
neither reliable nor scalable. In fact, the prevalent large proprietary models
could hardly generate verifiable programs. A promising yet largely uncharted
alternative is formal language-based reasoning. Grounding LLMs in rigorous
formal systems where generative models operate in formal language spaces (e.g.,
Dafny) enables the automatic and mathematically provable verification of their
reasoning processes and outcomes. This capability is pivotal for achieving
large-scale, reliable formal software verification. It is a common practice to
employ human-annotated chain-of-thought and other human priors to induce the
reasoning and coding capabilities of LLMs. Unfortunately, it becomes
unacceptably all-consuming to provide such priors for supervising complex
programming tasks. In this work, we systematically explore ways to reduce human
priors with the formal language, Dafny, as the main environment for our pilot
study. Our pipeline mainly relies on introducing an automatic and scalable data
curation pipeline, and careful RL designs integrated with feedback from the
formal language verifier. We introduce DafnyComp, a benchmark of compositional
formal programs with auto-formalized specifications for specification
reasoning. Our supervised fine-tuning (SFT) stage enables even small models
(e.g., 0.5B) to generate syntactically valid and verifiable Dafny code,
surpassing proprietary models. RL with regularization further improves
performance, achieving stronger generalization to out-of-domain tasks and
outperforming all strong baselines on the challenging DafnyComp benchmark.

### 10. [Dutch CrowS-Pairs: Adapting a Challenge Dataset for Measuring Social Biases in Language Models for Dutch](http://arxiv.org/pdf/2507.16442v1)

Authors: Elza Strazda, Gerasimos Spanakis

Warning: This paper contains explicit statements of offensive stereotypes
which might be upsetting.
  Language models are prone to exhibiting biases, further amplifying unfair and
harmful stereotypes. Given the fast-growing popularity and wide application of
these models, it is necessary to ensure safe and fair language models. As of
recent considerable attention has been paid to measuring bias in language
models, yet the majority of studies have focused only on English language. A
Dutch version of the US-specific CrowS-Pairs dataset for measuring bias in
Dutch language models is introduced. The resulting dataset consists of 1463
sentence pairs that cover bias in 9 categories, such as Sexual orientation,
Gender and Disability. The sentence pairs are composed of contrasting
sentences, where one of the sentences concerns disadvantaged groups and the
other advantaged groups. Using the Dutch CrowS-Pairs dataset, we show that
various language models, BERTje, RobBERT, multilingual BERT, GEITje and
Mistral-7B exhibit substantial bias across the various bias categories. Using
the English and French versions of the CrowS-Pairs dataset, bias was evaluated
in English (BERT and RoBERTa) and French (FlauBERT and CamemBERT) language
models, and it was shown that English models exhibit the most bias, whereas
Dutch models the least amount of bias. Additionally, results also indicate that
assigning a persona to a language model changes the level of bias it exhibits.
These findings highlight the variability of bias across languages and contexts,
suggesting that cultural and linguistic factors play a significant role in
shaping model biases.

### Cryptography and Security

### 1. [From Contracts to Code: Automating Smart Contract Generation with Multi-Level Finite State Machines](http://arxiv.org/pdf/2507.16276v1)

Authors: Lambard Maxence, Bertelle Cyrille, Duvallet Claude

In an increasingly complex contractual landscape, the demand for
transparency, security, and efficiency has intensified. Blockchain technology,
with its decentralized and immutable nature, addresses these challenges by
reducing intermediary costs, minimizing fraud risks, and enhancing system
compatibility. Smart contracts, initially conceptualized by Nick Szabo and
later implemented on the Ethereum blockchain, automate and secure contractual
clauses, offering a robust solution for various industries. However, their
complexity and the requirement for advanced programming skills present
significant barriers to widespread adoption. This study introduces a
multi-level finite state machine model designed to represent and track the
execution of smart contracts. Our model aims to simplify smart contract
development by providing a formalized framework that abstracts underlying
technical complexities, making it accessible to professionals without deep
technical expertise. The hierarchical structure of the multi-level finite state
machine enhances contract modularity and traceability, facilitating detailed
representation and evaluation of functional properties. The paper explores the
potential of this multi-level approach, reviewing existing methodologies and
tools, and detailing the smart contract generation process with an emphasis on
reusable components and modularity. We also conduct a security analysis to
evaluate potential vulnerabilities in our model, ensuring the robustness and
reliability of the generated smart contracts.

### 2. [Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers](http://arxiv.org/pdf/2507.16291v1)

Authors: Wenhao Li, Selvakumar Manickam, Yung-wey Chong, Shankar Karuppayah

Voice phishing (vishing) remains a persistent threat in cybersecurity,
exploiting human trust through persuasive speech. While machine learning
(ML)-based classifiers have shown promise in detecting malicious call
transcripts, they remain vulnerable to adversarial manipulations that preserve
semantic content. In this study, we explore a novel attack vector where large
language models (LLMs) are leveraged to generate adversarial vishing
transcripts that evade detection while maintaining deceptive intent. We
construct a systematic attack pipeline that employs prompt engineering and
semantic obfuscation to transform real-world vishing scripts using four
commercial LLMs. The generated transcripts are evaluated against multiple ML
classifiers trained on a real-world Korean vishing dataset (KorCCViD) with
statistical testing. Our experiments reveal that LLM-generated transcripts are
both practically and statistically effective against ML-based classifiers. In
particular, transcripts crafted by GPT-4o significantly reduce classifier
accuracy (by up to 30.96%) while maintaining high semantic similarity, as
measured by BERTScore. Moreover, these attacks are both time-efficient and
cost-effective, with average generation times under 9 seconds and negligible
financial cost per query. The results underscore the pressing need for more
resilient vishing detection frameworks and highlight the imperative for LLM
providers to enforce stronger safeguards against prompt misuse in adversarial
social engineering contexts.

### 3. [Explainable Vulnerability Detection in C/C++ Using Edge-Aware Graph Attention Networks](http://arxiv.org/pdf/2507.16540v1)

Authors: Radowanul Haque, Aftab Ali, Sally McClean, Naveed Khan

Detecting security vulnerabilities in source code remains challenging,
particularly due to class imbalance in real-world datasets where vulnerable
functions are under-represented. Existing learning-based methods often optimise
for recall, leading to high false positive rates and reduced usability in
development workflows. Furthermore, many approaches lack explainability,
limiting their integration into security workflows. This paper presents
ExplainVulD, a graph-based framework for vulnerability detection in C/C++ code.
The method constructs Code Property Graphs and represents nodes using
dual-channel embeddings that capture both semantic and structural information.
These are processed by an edge-aware attention mechanism that incorporates
edge-type embeddings to distinguish among program relations. To address class
imbalance, the model is trained using class-weighted cross-entropy loss.
ExplainVulD achieves a mean accuracy of 88.25 percent and an F1 score of 48.23
percent across 30 independent runs on the ReVeal dataset. These results
represent relative improvements of 4.6 percent in accuracy and 16.9 percent in
F1 score compared to the ReVeal model, a prior learning-based method. The
framework also outperforms static analysis tools, with relative gains of 14.0
to 14.1 percent in accuracy and 132.2 to 201.2 percent in F1 score. Beyond
improved detection performance, ExplainVulD produces explainable outputs by
identifying the most influential code regions within each function, supporting
transparency and trust in security triage.

### 4. [From Text to Actionable Intelligence: Automating STIX Entity and Relationship Extraction](http://arxiv.org/pdf/2507.16576v1)

Authors: Ahmed Lekssays, Husrev Taha Sencar, Ting Yu

Sharing methods of attack and their effectiveness is a cornerstone of
building robust defensive systems. Threat analysis reports, produced by various
individuals and organizations, play a critical role in supporting security
operations and combating emerging threats. To enhance the timeliness and
automation of threat intelligence sharing, several standards have been
established, with the Structured Threat Information Expression (STIX) framework
emerging as one of the most widely adopted. However, generating STIX-compatible
data from unstructured security text remains a largely manual, expert-driven
process. To address this challenge, we introduce AZERG, a tool designed to
assist security analysts in automatically generating structured STIX
representations. To achieve this, we adapt general-purpose large language
models for the specific task of extracting STIX-formatted threat data. To
manage the complexity, the task is divided into four subtasks: entity detection
(T1), entity type identification (T2), related pair detection (T3), and
relationship type identification (T4). We apply task-specific fine-tuning to
accurately extract relevant entities and infer their relationships in
accordance with the STIX specification. To address the lack of training data,
we compiled a comprehensive dataset with 4,011 entities and 2,075 relationships
extracted from 141 full threat analysis reports, all annotated in alignment
with the STIX standard. Our models achieved F1-scores of 84.43% for T1, 88.49%
for T2, 95.47% for T3, and 84.60% for T4 in real-world scenarios. We validated
their performance against a range of open- and closed-parameter models, as well
as state-of-the-art methods, demonstrating improvements of 2-25% across tasks.

### 5. [LLMxCPG: Context-Aware Vulnerability Detection Through Code Property Graph-Guided Large Language Models](http://arxiv.org/pdf/2507.16585v1)

Authors: Ahmed Lekssays, Hamza Mouhcine, Khang Tran, Ting Yu, Issa Khalil

Software vulnerabilities present a persistent security challenge, with over
25,000 new vulnerabilities reported in the Common Vulnerabilities and Exposures
(CVE) database in 2024 alone. While deep learning based approaches show promise
for vulnerability detection, recent studies reveal critical limitations in
terms of accuracy and robustness: accuracy drops by up to 45% on rigorously
verified datasets, and performance degrades significantly under simple code
modifications. This paper presents LLMxCPG, a novel framework integrating Code
Property Graphs (CPG) with Large Language Models (LLM) for robust vulnerability
detection. Our CPG-based slice construction technique reduces code size by
67.84 to 90.93% while preserving vulnerability-relevant context. Our approach's
ability to provide a more concise and accurate representation of code snippets
enables the analysis of larger code segments, including entire projects. This
concise representation is a key factor behind the improved detection
capabilities of our method, as it can now identify vulnerabilities that span
multiple functions. Empirical evaluation demonstrates LLMxCPG's effectiveness
across verified datasets, achieving 15-40% improvements in F1-score over
state-of-the-art baselines. Moreover, LLMxCPG maintains high performance across
function-level and multi-function codebases while exhibiting robust detection
efficacy under various syntactic code modifications.

### 6. [When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs](http://arxiv.org/pdf/2507.16773v1)

Authors: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

Large Language Models (LLMs) have become integral to automated code analysis,
enabling tasks such as vulnerability detection and code comprehension. However,
their integration introduces novel attack surfaces. In this paper, we identify
and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks
(CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs.
By injecting carefully crafted triggers into external code snippets,
adversaries can induce the model to replicate malicious content during
inference. This behavior enables two classes of vulnerabilities: inference
length manipulation, where the model generates abnormally short or excessively
long reasoning traces; and inference result manipulation, where the model
produces misleading or incorrect conclusions. We formalize CGA as an
optimization problem and propose a gradient-based approach to synthesize
effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs
shows that CGA reliably induces infinite loops, premature termination, false
refusals, and semantic distortions in code analysis tasks. While highly
effective in targeted settings, we observe challenges in generalizing CGA
across diverse prompts due to computational constraints, posing an open
question for future research. Our findings expose a critical yet underexplored
vulnerability in LLM-powered development pipelines and call for urgent advances
in prompt-level defense mechanisms.

### 7. [AUTOPSY: A Framework for Tackling Privacy Challenges in the Automotive Industry](http://arxiv.org/pdf/2507.16788v1)

Authors: Sebastian Pape, Anis Bkakria, Maurice Heymann, Badreddine Chah, Abdeljalil Abbas-Turki, Sarah Syed-Winkler, Matthias Hiller, Reda Yaich

With the General Data Protection Regulation (GDPR) in place, all domains have
to ensure compliance with privacy legislation. However, compliance does not
necessarily result in a privacy-friendly system as for example getting users'
consent to process their data does not improve the privacy-friendliness of the
system. Therefore, the goal of the AUTOPSY project was to support the privacy
engineering process in the automotive domain by providing several building
blocks which technically improve the privacy-friendliness of modern, i.e.,
connected and (partially) automated vehicles. This paper presents the results
of the AUTOPSY project: a system model to identify relevant entities and
locations to apply privacy enhancing technologies (PETs); the privacy manager
aiming at more control of the data flow from the vehicle, a PET selection
approach based on GDPR principles, and an architectural framework for
automotive privacy. Furthermore, we built a demonstrator for location-based
services to evaluate the architectural framework.

### 8. [DP2Guard: A Lightweight and Byzantine-Robust Privacy-Preserving Federated Learning Scheme for Industrial IoT](http://arxiv.org/pdf/2507.16134v1)

Authors: Baofu Han, Bing Li, Yining Qi, Raja Jurdak, Kaibin Huang, Chau Yuen

Privacy-Preserving Federated Learning (PPFL) has emerged as a secure
distributed Machine Learning (ML) paradigm that aggregates locally trained
gradients without exposing raw data. To defend against model poisoning threats,
several robustness-enhanced PPFL schemes have been proposed by integrating
anomaly detection. Nevertheless, they still face two major challenges: (1) the
reliance on heavyweight encryption techniques results in substantial
communication and computation overhead; and (2) single-strategy defense
mechanisms often fail to provide sufficient robustness against adaptive
adversaries. To overcome these challenges, we propose DP2Guard, a lightweight
PPFL framework that enhances both privacy and robustness. DP2Guard leverages a
lightweight gradient masking mechanism to replace costly cryptographic
operations while ensuring the privacy of local gradients. A hybrid defense
strategy is proposed, which extracts gradient features using singular value
decomposition and cosine similarity, and applies a clustering algorithm to
effectively identify malicious gradients. Additionally, DP2Guard adopts a trust
score-based adaptive aggregation scheme that adjusts client weights according
to historical behavior, while blockchain records aggregated results and trust
scores to ensure tamper-proof and auditable training. Extensive experiments
conducted on two public datasets demonstrate that DP2Guard effectively defends
against four advanced poisoning attacks while ensuring privacy with reduced
communication and computation costs.

### 9. [Pulse-Level Simulation of Crosstalk Attacks on Superconducting Quantum Hardware](http://arxiv.org/pdf/2507.16181v1)

Authors: Syed Emad Uddin Shubha, Tasnuva Farheen

Hardware crosstalk in multi-tenant superconducting quantum computers poses a
severe security threat, allowing adversaries to induce targeted errors across
tenant boundaries by injecting carefully engineered pulses. We present a
simulation-based study of active crosstalk attacks at the pulse level,
analyzing how adversarial control of pulse timing, shape, amplitude, and
coupling can disrupt a victim's computation. Our framework models the
time-dependent dynamics of a three-qubit system in the rotating frame,
capturing both always-on couplings and injected drive pulses. We examine two
attack strategies: attacker-first (pulse before victim operation) and
victim-first (pulse after), and systematically identify the pulse and coupling
configurations that cause the largest logical errors. Protocol-level
experiments on quantum coin flip and XOR classification circuits show that some
protocols are highly vulnerable to these attacks, while others remain robust.
Based on these findings, we discuss practical methods for detection and
mitigation to improve security in quantum cloud platforms.

### 10. [LENS-DF: Deepfake Detection and Temporal Localization for Long-Form Noisy Speech](http://arxiv.org/pdf/2507.16220v1)

Authors: Xuechen Liu, Wanying Ge, Xin Wang, Junichi Yamagishi

This study introduces LENS-DF, a novel and comprehensive recipe for training
and evaluating audio deepfake detection and temporal localization under
complicated and realistic audio conditions. The generation part of the recipe
outputs audios from the input dataset with several critical characteristics,
such as longer duration, noisy conditions, and containing multiple speakers, in
a controllable fashion. The corresponding detection and localization protocol
uses models. We conduct experiments based on self-supervised learning front-end
and simple back-end. The results indicate that models trained using data
generated with LENS-DF consistently outperform those trained via conventional
recipes, demonstrating the effectiveness and usefulness of LENS-DF for robust
audio deepfake detection and localization. We also conduct ablation studies on
the variations introduced, investigating their impact on and relevance to
realistic challenges in the field.

### Computer Vision and Pattern Recognition

### 1. [PUSA V1.0: Surpassing Wan-I2V with $500 Training Cost by Vectorized Timestep Adaptation](http://arxiv.org/pdf/2507.16116v1)

Authors: Yaofang Liu, Yumeng Ren, Aitor Artola, Yuxuan Hu, Xiaodong Cun, Xiaotong Zhao, Alan Zhao, Raymond H. Chan, Suiyun Zhang, Rui Liu, Dandan Tu, Jean-Michel Morel

The rapid advancement of video diffusion models has been hindered by
fundamental limitations in temporal modeling, particularly the rigid
synchronization of frame evolution imposed by conventional scalar timestep
variables. While task-specific adaptations and autoregressive models have
sought to address these challenges, they remain constrained by computational
inefficiency, catastrophic forgetting, or narrow applicability. In this work,
we present Pusa, a groundbreaking paradigm that leverages vectorized timestep
adaptation (VTA) to enable fine-grained temporal control within a unified video
diffusion framework. Besides, VTA is a non-destructive adaptation, which means
it fully preserves the capabilities of the base model. By finetuning the SOTA
Wan2.1-T2V-14B model with VTA, we achieve unprecedented efficiency --
surpassing the performance of Wan-I2V-14B with $\leq$ 1/200 of the training
cost (\$500 vs. $\geq$ \$100,000) and $\leq$ 1/2500 of the dataset size (4K vs.
$\geq$ 10M samples). Pusa not only sets a new standard for image-to-video (I2V)
generation, achieving a VBench-I2V total score of 87.32\% (vs. 86.86\% of
Wan-I2V-14B), but also unlocks many zero-shot multi-task capabilities such as
start-end frames and video extension -- all without task-specific training.
Meanwhile, Pusa can still perform text-to-video generation. Mechanistic
analyses reveal that our approach preserves the foundation model's generative
priors while surgically injecting temporal dynamics, avoiding the combinatorial
explosion inherent to vectorized timesteps. This work establishes a scalable,
efficient, and versatile paradigm for next-generation video synthesis,
democratizing high-fidelity video generation for research and industry alike.
Code is open-sourced at https://github.com/Yaofang-Liu/Pusa-VidGen

### 2. [LongSplat: Online Generalizable 3D Gaussian Splatting from Long Sequence Images](http://arxiv.org/pdf/2507.16144v1)

Authors: Guichen Huang, Ruoyu Wang, Xiangjun Gao, Che Sun, Yuwei Wu, Shenghua Gao, Yunde Jia

3D Gaussian Splatting achieves high-fidelity novel view synthesis, but its
application to online long-sequence scenarios is still limited. Existing
methods either rely on slow per-scene optimization or fail to provide efficient
incremental updates, hindering continuous performance. In this paper, we
propose LongSplat, an online real-time 3D Gaussian reconstruction framework
designed for long-sequence image input. The core idea is a streaming update
mechanism that incrementally integrates current-view observations while
selectively compressing redundant historical Gaussians. Crucial to this
mechanism is our Gaussian-Image Representation (GIR), a representation that
encodes 3D Gaussian parameters into a structured, image-like 2D format. GIR
simultaneously enables efficient fusion of current-view and historical
Gaussians and identity-aware redundancy compression. These functions enable
online reconstruction and adapt the model to long sequences without
overwhelming memory or computational costs. Furthermore, we leverage an
existing image compression method to guide the generation of more compact and
higher-quality 3D Gaussians. Extensive evaluations demonstrate that LongSplat
achieves state-of-the-art efficiency-quality trade-offs in real-time novel view
synthesis, delivering real-time reconstruction while reducing Gaussian counts
by 44\% compared to existing per-pixel Gaussian prediction methods.

### 3. [AMMNet: An Asymmetric Multi-Modal Network for Remote Sensing Semantic Segmentation](http://arxiv.org/pdf/2507.16158v1)

Authors: Hui Ye, Haodong Chen, Zeke Zexi Hu, Xiaoming Chen, Yuk Ying Chung

Semantic segmentation in remote sensing (RS) has advanced significantly with
the incorporation of multi-modal data, particularly the integration of RGB
imagery and the Digital Surface Model (DSM), which provides complementary
contextual and structural information about the ground object. However,
integrating RGB and DSM often faces two major limitations: increased
computational complexity due to architectural redundancy, and degraded
segmentation performance caused by modality misalignment. These issues
undermine the efficiency and robustness of semantic segmentation, particularly
in complex urban environments where precise multi-modal integration is
essential. To overcome these limitations, we propose Asymmetric Multi-Modal
Network (AMMNet), a novel asymmetric architecture that achieves robust and
efficient semantic segmentation through three designs tailored for RGB-DSM
input pairs. To reduce architectural redundancy, the Asymmetric Dual Encoder
(ADE) module assigns representational capacity based on modality-specific
characteristics, employing a deeper encoder for RGB imagery to capture rich
contextual information and a lightweight encoder for DSM to extract sparse
structural features. Besides, to facilitate modality alignment, the Asymmetric
Prior Fuser (APF) integrates a modality-aware prior matrix into the fusion
process, enabling the generation of structure-aware contextual features.
Additionally, the Distribution Alignment (DA) module enhances cross-modal
compatibility by aligning feature distributions through divergence
minimization. Extensive experiments on the ISPRS Vaihingen and Potsdam datasets
demonstrate that AMMNet attains state-of-the-art segmentation accuracy among
multi-modal networks while reducing computational and memory requirements.

### 4. [AtrousMamaba: An Atrous-Window Scanning Visual State Space Model for Remote Sensing Change Detection](http://arxiv.org/pdf/2507.16172v1)

Authors: Tao Wang, Tiecheng Bai, Chao Xu, Bin Liu, Erlei Zhang, Jiyun Huang, Hongming Zhang

Recently, a novel visual state space (VSS) model, referred to as Mamba, has
demonstrated significant progress in modeling long sequences with linear
complexity, comparable to Transformer models, thereby enhancing its
adaptability for processing visual data. Although most methods aim to enhance
the global receptive field by directly modifying Mamba's scanning mechanism,
they tend to overlook the critical importance of local information in dense
prediction tasks. Additionally, whether Mamba can effectively extract local
features as convolutional neural networks (CNNs) do remains an open question
that merits further investigation. In this paper, We propose a novel model,
AtrousMamba, which effectively balances the extraction of fine-grained local
details with the integration of global contextual information. Specifically,
our method incorporates an atrous-window selective scan mechanism, enabling a
gradual expansion of the scanning range with adjustable rates. This design
shortens the distance between adjacent tokens, enabling the model to
effectively capture fine-grained local features and global context. By
leveraging the atrous window scan visual state space (AWVSS) module, we design
dedicated end-to-end Mamba-based frameworks for binary change detection (BCD)
and semantic change detection (SCD), referred to as AWMambaBCD and AWMambaSCD,
respectively. Experimental results on six benchmark datasets show that the
proposed framework outperforms existing CNN-based, Transformer-based, and
Mamba-based methods. These findings clearly demonstrate that Mamba not only
captures long-range dependencies in visual data but also effectively preserves
fine-grained local details.

### 5. [Explicit Context Reasoning with Supervision for Visual Tracking](http://arxiv.org/pdf/2507.16191v1)

Authors: Fansheng Zeng, Bineng Zhong, Haiying Xia, Yufei Tan, Xiantao Hu, Liangtao Shi, Shuxiang Song

Contextual reasoning with constraints is crucial for enhancing temporal
consistency in cross-frame modeling for visual tracking. However, mainstream
tracking algorithms typically associate context by merely stacking historical
information without explicitly supervising the association process, making it
difficult to effectively model the target's evolving dynamics. To alleviate
this problem, we propose RSTrack, which explicitly models and supervises
context reasoning via three core mechanisms. \textit{1) Context Reasoning
Mechanism}: Constructs a target state reasoning pipeline, converting
unconstrained contextual associations into a temporal reasoning process that
predicts the current representation based on historical target states, thereby
enhancing temporal consistency. \textit{2) Forward Supervision Strategy}:
Utilizes true target features as anchors to constrain the reasoning pipeline,
guiding the predicted output toward the true target distribution and
suppressing drift in the context reasoning process. \textit{3) Efficient State
Modeling}: Employs a compression-reconstruction mechanism to extract the core
features of the target, removing redundant information across frames and
preventing ineffective contextual associations. These three mechanisms
collaborate to effectively alleviate the issue of contextual association
divergence in traditional temporal modeling. Experimental results show that
RSTrack achieves state-of-the-art performance on multiple benchmark datasets
while maintaining real-time running speeds. Our code is available at
https://github.com/GXNU-ZhongLab/RSTrack.

### 6. [A Single-step Accurate Fingerprint Registration Method Based on Local Feature Matching](http://arxiv.org/pdf/2507.16201v1)

Authors: Yuwei Jia, Zhe Cui, Fei Su

Distortion of the fingerprint images leads to a decline in fingerprint
recognition performance, and fingerprint registration can mitigate this
distortion issue by accurately aligning two fingerprint images. Currently,
fingerprint registration methods often consist of two steps: an initial
registration based on minutiae, and a dense registration based on matching
points. However, when the quality of fingerprint image is low, the number of
detected minutiae is reduced, leading to frequent failures in the initial
registration, which ultimately causes the entire fingerprint registration
process to fail. In this study, we propose an end-to-end single-step
fingerprint registration algorithm that aligns two fingerprints by directly
predicting the semi-dense matching points correspondences between two
fingerprints. Thus, our method minimizes the risk of minutiae registration
failure and also leverages global-local attentions to achieve end-to-end
pixel-level alignment between the two fingerprints. Experiment results prove
that our method can achieve the state-of-the-art matching performance with only
single-step registration, and it can also be used in conjunction with dense
registration algorithms for further performance improvements.

### 7. [LDRFusion: A LiDAR-Dominant multimodal refinement framework for 3D object detection](http://arxiv.org/pdf/2507.16224v1)

Authors: Jijun Wang, Yan Wu, Yujian Mo, Junqiao Zhao, Jun Yan, Yinghao Hu

Existing LiDAR-Camera fusion methods have achieved strong results in 3D
object detection. To address the sparsity of point clouds, previous approaches
typically construct spatial pseudo point clouds via depth completion as
auxiliary input and adopts a proposal-refinement framework to generate
detection results. However, introducing pseudo points inevitably brings noise,
potentially resulting in inaccurate predictions. Considering the differing
roles and reliability levels of each modality, we propose LDRFusion, a novel
Lidar-dominant two-stage refinement framework for multi-sensor fusion. The
first stage soley relies on LiDAR to produce accurately localized proposals,
followed by a second stage where pseudo point clouds are incorporated to detect
challenging instances. The instance-level results from both stages are
subsequently merged. To further enhance the representation of local structures
in pseudo point clouds, we present a hierarchical pseudo point residual
encoding module, which encodes neighborhood sets using both feature and
positional residuals. Experiments on the KITTI dataset demonstrate that our
framework consistently achieves strong performance across multiple categories
and difficulty levels.

### 8. [MONITRS: Multimodal Observations of Natural Incidents Through Remote Sensing](http://arxiv.org/pdf/2507.16228v1)

Authors: Shreelekha Revankar, Utkarsh Mall, Cheng Perng Phoo, Kavita Bala, Bharath Hariharan

Natural disasters cause devastating damage to communities and infrastructure
every year. Effective disaster response is hampered by the difficulty of
accessing affected areas during and after events. Remote sensing has allowed us
to monitor natural disasters in a remote way. More recently there have been
advances in computer vision and deep learning that help automate satellite
imagery analysis, However, they remain limited by their narrow focus on
specific disaster types, reliance on manual expert interpretation, and lack of
datasets with sufficient temporal granularity or natural language annotations
for tracking disaster progression. We present MONITRS, a novel multimodal
dataset of more than 10,000 FEMA disaster events with temporal satellite
imagery and natural language annotations from news articles, accompanied by
geotagged locations, and question-answer pairs. We demonstrate that fine-tuning
existing MLLMs on our dataset yields significant performance improvements for
disaster monitoring tasks, establishing a new benchmark for machine
learning-assisted disaster response systems. Code can be found at:
https://github.com/ShreelekhaR/MONITRS

### 9. [Positive Style Accumulation: A Style Screening and Continuous Utilization Framework for Federated DG-ReID](http://arxiv.org/pdf/2507.16238v1)

Authors: Xin Xu, Chaoyue Ren, Wei Liu, Wenke Huang, Bin Yang, Zhixi Yu, Kui Jiang

The Federated Domain Generalization for Person re-identification (FedDG-ReID)
aims to learn a global server model that can be effectively generalized to
source and target domains through distributed source domain data. Existing
methods mainly improve the diversity of samples through style transformation,
which to some extent enhances the generalization performance of the model.
However, we discover that not all styles contribute to the generalization
performance. Therefore, we define styles that are beneficial or harmful to the
model's generalization performance as positive or negative styles. Based on
this, new issues arise: How to effectively screen and continuously utilize the
positive styles. To solve these problems, we propose a Style Screening and
Continuous Utilization (SSCU) framework. Firstly, we design a Generalization
Gain-guided Dynamic Style Memory (GGDSM) for each client model to screen and
accumulate generated positive styles. Meanwhile, we propose a style memory
recognition loss to fully leverage the positive styles memorized by Memory.
Furthermore, we propose a Collaborative Style Training (CST) strategy to make
full use of positive styles. Unlike traditional learning strategies, our
approach leverages both newly generated styles and the accumulated positive
styles stored in memory to train client models on two distinct branches. This
training strategy is designed to effectively promote the rapid acquisition of
new styles by the client models, and guarantees the continuous and thorough
utilization of positive styles, which is highly beneficial for the model's
generalization performance. Extensive experimental results demonstrate that our
method outperforms existing methods in both the source domain and the target
domain.

### 10. [Scale Your Instructions: Enhance the Instruction-Following Fidelity of Unified Image Generation Model by Self-Adaptive Attention Scaling](http://arxiv.org/pdf/2507.16240v1)

Authors: Chao Zhou, Tianyi Wei, Nenghai Yu

Recent advancements in unified image generation models, such as OmniGen, have
enabled the handling of diverse image generation and editing tasks within a
single framework, accepting multimodal, interleaved texts and images in free
form. This unified architecture eliminates the need for text encoders, greatly
reducing model complexity and standardizing various image generation and
editing tasks, making it more user-friendly. However, we found that it suffers
from text instruction neglect, especially when the text instruction contains
multiple sub-instructions. To explore this issue, we performed a perturbation
analysis on the input to identify critical steps and layers. By examining the
cross-attention maps of these key steps, we observed significant conflicts
between neglected sub-instructions and the activations of the input image. In
response, we propose Self-Adaptive Attention Scaling (SaaS), a method that
leverages the consistency of cross-attention between adjacent timesteps to
dynamically scale the attention activation for each sub-instruction. Our SaaS
enhances instruction-following fidelity without requiring additional training
or test-time optimization. Experimental results on instruction-based image
editing and visual conditional image generation validate the effectiveness of
our SaaS, showing superior instruction-following fidelity over existing
methods. The code is available https://github.com/zhouchao-ops/SaaS.

### Computers and Society

### 1. [The Impact of Pseudo-Science in Financial Loans Risk Prediction](http://arxiv.org/pdf/2507.16182v1)

Authors: Bruno Scarone, Ricardo Baeza-Yates

We study the societal impact of pseudo-scientific assumptions for predicting
the behavior of people in a straightforward application of machine learning to
risk prediction in financial lending. This use case also exemplifies the impact
of survival bias in loan return prediction. We analyze the models in terms of
their accuracy and social cost, showing that the socially optimal model may not
imply a significant accuracy loss for this downstream task. Our results are
verified for commonly used learning methods and datasets. Our findings also
show that there is a natural dynamic when training models that suffer survival
bias where accuracy slightly deteriorates, and whose recall and precision
improves with time. These results act as an illusion, leading the observer to
believe that the system is getting better, when in fact the model is suffering
from increasingly more unfairness and survival bias.

### 2. [Characterizing Online Activities Contributing to Suicide Mortality among Youth](http://arxiv.org/pdf/2507.16185v1)

Authors: Aparna Ananthasubramaniam, Elyse J. Thulin, Viktoryia Kalesnikava, Silas Falde, Jonathan Kertawidjaja, Lily Johns, Alejandro Rodríguez-Putnam, Emma Spring, Kara Zivin, Briana Mezuk

The recent rise in youth suicide highlights the urgent need to understand how
online experiences contribute to this public health issue. Our mixed-methods
approach responds to this challenge by developing a set of themes focused on
risk factors for suicide mortality in online spaces among youth ages 10-24, and
a framework to model these themes at scale. Using 29,124 open text summaries of
death investigations between 2013-2022, we conducted a thematic analysis to
identify 12 types of online activities that were considered by investigators or
next of kin to be relevant in contextualizing a given suicide death. We then
develop a zero-shot learning framework to model these 12 themes at scale, and
analyze variation in these themes by decedent characteristics and over time.
Our work uncovers several online activities related to harm to self, harm to
others, interpersonal interactions, activity levels online, and life events,
which correspond to different phases of suicide risk from two prominent suicide
theories. We find an association between these themes and decedent
characteristics like age, means of death, and interpersonal problems, and many
themes became more prevalent during the 2020 COVID-19 lockdowns. While digital
spaces have taken some steps to address expressions of suicidality online, our
work illustrates the opportunities for developing interventions related to less
explicit indicators of suicide risk by combining suicide theories with
computational research.

### 3. [Beyond Algorethics: Addressing the Ethical and Anthropological Challenges of AI Recommender Systems](http://arxiv.org/pdf/2507.16430v1)

Authors: Octavian M. Machidon

In this paper, I examine the ethical and anthropological challenges posed by
AI-driven recommender systems (RSs), which have become central to shaping
digital environments and social interactions. By curating personalized content,
RSs do not merely reflect user preferences but actively construct individual
experiences across social media, entertainment platforms, and e-commerce.
Despite their ubiquity, the ethical implications of RSs remain insufficiently
explored, even as concerns over privacy, autonomy, and mental well-being
intensify. I argue that existing ethical approaches, including algorethics, the
effort to embed ethical principles into algorithmic design, are necessary but
ultimately inadequate. RSs inherently reduce human complexity to quantifiable
dimensions, exploit user vulnerabilities, and prioritize engagement over
well-being. Addressing these concerns requires moving beyond purely technical
solutions. I propose a comprehensive framework for human-centered RS design,
integrating interdisciplinary perspectives, regulatory strategies, and
educational initiatives to ensure AI systems foster rather than undermine human
autonomy and societal flourishing.

### 4. [Disability Across Cultures: A Human-Centered Audit of Ableism in Western and Indic LLMs](http://arxiv.org/pdf/2507.16130v1)

Authors: Mahika Phutane, Aditya Vashistha

People with disabilities (PwD) experience disproportionately high levels of
discrimination and hate online, particularly in India, where entrenched stigma
and limited resources intensify these challenges. Large language models (LLMs)
are increasingly used to identify and mitigate online hate, yet most research
on online ableism focuses on Western audiences with Western AI models. Are
these models adequately equipped to recognize ableist harm in non-Western
places like India? Do localized, Indic language models perform better? To
investigate, we adopted and translated a publicly available ableist speech
dataset to Hindi, and prompted eight LLMs--four developed in the U.S. (GPT-4,
Gemini, Claude, Llama) and four in India (Krutrim, Nanda, Gajendra,
Airavata)--to score and explain ableism. In parallel, we recruited 175 PwD from
both the U.S. and India to perform the same task, revealing stark differences
between groups. Western LLMs consistently overestimated ableist harm, while
Indic LLMs underestimated it. Even more concerning, all LLMs were more tolerant
of ableism when it was expressed in Hindi and asserted Western framings of
ableist harm. In contrast, Indian PwD interpreted harm through intention,
relationality, and resilience--emphasizing a desire to inform and educate
perpetrators. This work provides groundwork for global, inclusive standards of
ableism, demonstrating the need to center local disability experiences in the
design and evaluation of AI systems.

### 5. [A Human-Centered Approach to Identifying Promises, Risks, & Challenges of Text-to-Image Generative AI in Radiology](http://arxiv.org/pdf/2507.16207v1)

Authors: Katelyn Morrison, Arpit Mathur, Aidan Bradshaw, Tom Wartmann, Steven Lundi, Afrooz Zandifar, Weichang Dai, Kayhan Batmanghelich, Motahhare Eslami, Adam Perer

As text-to-image generative models rapidly improve, AI researchers are making
significant advances in developing domain-specific models capable of generating
complex medical imagery from text prompts. Despite this, these technical
advancements have overlooked whether and how medical professionals would
benefit from and use text-to-image generative AI (GenAI) in practice. By
developing domain-specific GenAI without involving stakeholders, we risk the
potential of building models that are either not useful or even more harmful
than helpful. In this paper, we adopt a human-centered approach to responsible
model development by involving stakeholders in evaluating and reflecting on the
promises, risks, and challenges of a novel text-to-CT Scan GenAI model. Through
exploratory model prompting activities, we uncover the perspectives of medical
students, radiology trainees, and radiologists on the role that text-to-CT Scan
GenAI can play across medical education, training, and practice. This
human-centered approach additionally enabled us to surface technical challenges
and domain-specific risks of generating synthetic medical images. We conclude
by reflecting on the implications of medical text-to-image GenAI.

### 6. [Voice-based AI Agents: Filling the Economic Gaps in Digital Health Delivery](http://arxiv.org/pdf/2507.16229v1)

Authors: Bo Wen, Chen Wang, Qiwei Han, Raquel Norel, Julia Liu, Thaddeus Stappenbeck, Jeffrey L. Rogers

The integration of voice-based AI agents in healthcare presents a
transformative opportunity to bridge economic and accessibility gaps in digital
health delivery. This paper explores the role of large language model
(LLM)-powered voice assistants in enhancing preventive care and continuous
patient monitoring, particularly in underserved populations. Drawing insights
from the development and pilot study of Agent PULSE (Patient Understanding and
Liaison Support Engine) -- a collaborative initiative between IBM Research,
Cleveland Clinic Foundation, and Morehouse School of Medicine -- we present an
economic model demonstrating how AI agents can provide cost-effective
healthcare services where human intervention is economically unfeasible. Our
pilot study with 33 inflammatory bowel disease patients revealed that 70\%
expressed acceptance of AI-driven monitoring, with 37\% preferring it over
traditional modalities. Technical challenges, including real-time
conversational AI processing, integration with healthcare systems, and privacy
compliance, are analyzed alongside policy considerations surrounding
regulation, bias mitigation, and patient autonomy. Our findings suggest that
AI-driven voice agents not only enhance healthcare scalability and efficiency
but also improve patient engagement and accessibility. For healthcare
executives, our cost-utility analysis demonstrates huge potential savings for
routine monitoring tasks, while technologists can leverage our framework to
prioritize improvements yielding the highest patient impact. By addressing
current limitations and aligning AI development with ethical and regulatory
frameworks, voice-based AI agents can serve as a critical entry point for
equitable, sustainable digital healthcare solutions.

### 7. [PRAC3 (Privacy, Reputation, Accountability, Consent, Credit, Compensation): Long Tailed Risks of Voice Actors in AI Data-Economy](http://arxiv.org/pdf/2507.16247v1)

Authors: Tanusree Sharma, Yihao Zhou, Visar Berisha

Early large-scale audio datasets, such as LibriSpeech, were built with
hundreds of individual contributors whose voices were instrumental in the
development of speech technologies, including audiobooks and voice assistants.
Yet, a decade later, these same contributions have exposed voice actors to a
range of risks. While existing ethical frameworks emphasize Consent, Credit,
and Compensation (C3), they do not adequately address the emergent risks
involving vocal identities that are increasingly decoupled from context,
authorship, and control. Drawing on qualitative interviews with 20 professional
voice actors, this paper reveals how the synthetic replication of voice without
enforceable constraints exposes individuals to a range of threats. Beyond
reputational harm, such as re-purposing voice data in erotic content, offensive
political messaging, and meme culture, we document concerns about
accountability breakdowns when their voice is leveraged to clone voices that
are deployed in high-stakes scenarios such as financial fraud, misinformation
campaigns, or impersonation scams. In such cases, actors face social and legal
fallout without recourse, while very few of them have a legal representative or
union protection. To make sense of these shifting dynamics, we introduce the
PRAC3 framework, an expansion of C3 that foregrounds Privacy, Reputation,
Accountability, Consent, Credit, and Compensation as interdependent pillars of
data used in the synthetic voice economy. This framework captures how privacy
risks are amplified through non-consensual training, how reputational harm
arises from decontextualized deployment, and how accountability can be
reimagined AI Data ecosystems. We argue that voice, as both a biometric
identifier and creative labor, demands governance models that restore creator
agency, ensure traceability, and establish enforceable boundaries for ethical
reuse.

### 8. [WhatsApp Tiplines and Multilingual Claims in the 2021 Indian Assembly Elections](http://arxiv.org/pdf/2507.16298v1)

Authors: Gautam Kishore Shahi, Scot A. Hale

WhatsApp tiplines, first launched in 2019 to combat misinformation, enable
users to interact with fact-checkers to verify misleading content. This study
analyzes 580 unique claims (tips) from 451 users, covering both high-resource
languages (English, Hindi) and a low-resource language (Telugu) during the 2021
Indian assembly elections using a mixed-method approach. We categorize the
claims into three categories, election, COVID-19, and others, and observe
variations across languages. We compare content similarity through frequent
word analysis and clustering of neural sentence embeddings. We also investigate
user overlap across languages and fact-checking organizations. We measure the
average time required to debunk claims and inform tipline users. Results reveal
similarities in claims across languages, with some users submitting tips in
multiple languages to the same fact-checkers. Fact-checkers generally require a
couple of days to debunk a new claim and share the results with users. Notably,
no user submits claims to multiple fact-checking organizations, indicating that
each organization maintains a unique audience. We provide practical
recommendations for using tiplines during elections with ethical consideration
of users' information.

### 9. [GG-BBQ: German Gender Bias Benchmark for Question Answering](http://arxiv.org/pdf/2507.16410v1)

Authors: Shalaka Satheesh, Katrin Klug, Katharina Beckh, Héctor Allende-Cid, Sebastian Houben, Teena Hassan

Within the context of Natural Language Processing (NLP), fairness evaluation
is often associated with the assessment of bias and reduction of associated
harm. In this regard, the evaluation is usually carried out by using a
benchmark dataset, for a task such as Question Answering, created for the
measurement of bias in the model's predictions along various dimensions,
including gender identity. In our work, we evaluate gender bias in German Large
Language Models (LLMs) using the Bias Benchmark for Question Answering by
Parrish et al. (2022) as a reference. Specifically, the templates in the gender
identity subset of this English dataset were machine translated into German.
The errors in the machine translated templates were then manually reviewed and
corrected with the help of a language expert. We find that manual revision of
the translation is crucial when creating datasets for gender bias evaluation
because of the limitations of machine translation from English to a language
such as German with grammatical gender. Our final dataset is comprised of two
subsets: Subset-I, which consists of group terms related to gender identity,
and Subset-II, where group terms are replaced with proper names. We evaluate
several LLMs used for German NLP on this newly created dataset and report the
accuracy and bias scores. The results show that all models exhibit bias, both
along and against existing social stereotypes.

### 10. [PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](http://arxiv.org/pdf/2507.16679v1)

Authors: Han Jiang, Dongyao Zhu, Zhihua Wei, Xiaoyuan Yi, Ziang Xiao, Xing Xie

In-Context Learning has shown great potential for aligning Large Language
Models (LLMs) with human values, helping reduce harmful outputs and accommodate
diverse preferences without costly post-training, known as In-Context Alignment
(ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting
ICA's ability to address value tensions--human values are inherently
pluralistic, often imposing conflicting demands, e.g., stimulation vs.
tradition. Current ICA methods therefore face the Instruction Bottleneck
challenge, where LLMs struggle to reconcile multiple intended values within a
single prompt, leading to incomplete or biased alignment. To address this, we
propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO
optimizes a meta-instruction that navigates multiple values to better elicit
LLMs' understanding of them and improve their alignment. This is achieved by
maximizing the total correlation between specified values and LLM responses,
theoretically reinforcing value correlation while reducing distractive noise,
resulting in effective value instructions. Extensive experiments on five value
sets show that PICACO works well with both black-box and open-source LLMs,
outperforms several recent strong baselines, and achieves a better balance
across up to 8 distinct values.

### Distributed, Parallel, and Cluster Computing

### 1. [Autonomous Dominant Resource Fairness for Blockchain Ecosystems](http://arxiv.org/pdf/2507.16350v1)

Authors: Serdar Metin

Blockchain systems have been a part of mainstream academic research, and a
hot topic at that. It has spread to almost every subfield in the computer
science literature, as well as economics and finance. Especially in a world
where digital trust is much sought for, blockchains offer a rich variety of
desired properties, such as immutability, public auditing, decentralised record
keeping, among others. Not only has it been a research topic of its own, the
integration of blockchains into other systems has been proposed as solutions in
many areas, ranging from grid computing, cloud and fog computing, to internet
of things, self driving vehicles , and smart cities. In many cases the primary
function attributed to blockchains in these contexts is resource management.
Although much attention is paid to this topic, the focus is on single resource
allocation scenarios. Even the cases where multiple resource types are to be
allocated, are treated as single resource type scenarios, and problems are
formulated as allocating standardised bundles consisting of a fixed amount of
each of them, such as virtual machines. The present study addresses the problem
of allocating multiple resource types among tasks with heterogeneous resource
demands with a smart contract adaptation of Precomputed Dominant Resource
Fairness; an algorithm that approximates Dominant Resource Fairness, without
loop iterations, which makes it preferable in the blockchain context because of
the block gas limit. We present the resulting algorithm, Autonomous Dominant
Resource Fairness, along with the empirical data collected from the tests run
on the algorithm. The results show that Autonomous Dominant Resource Fairness
is a gas-cost efficient algorithm, which can be used to manage hundreds of
resource types for unlimited number of users.

### 2. [FOGNITE: Federated Learning-Enhanced Fog-Cloud Architecture](http://arxiv.org/pdf/2507.16668v1)

Authors: Somayeh Sobati-M

Modern smart grids demand fast, intelligent, and energy-aware computing at
the edge to manage real time fluctuations and ensure reliable operation. This
paper introduces FOGNITE Fog-based Grid In intelligence with Neural Integration
and Twin based Execution a next-generation fog cloud framework designed to
enhance autonomy, resilience, and efficiency in distributed energy systems.
FOGNITE combines three core components: federated learning, reinforcement
learning, and digital twin validation. Each fog node trains a local CNN LSTM
model on private energy consumption data, enabling predictive intelligence
while preserving data privacy through federated aggregation. A reinforcement
learning agent dynamically schedules tasks based on current system load and
energy conditions, optimizing for performance under uncertainty.
  To prevent unsafe or inefficient decisions, a hierarchical digital twin layer
simulates potential actions before deployment, significantly reducing execution
errors and energy waste. We evaluate FOGNITE on a real world testbed of
Raspberry Pi devices, showing up to a 93.7% improvement in load balancing
accuracy and a 63.2% reduction in energy waste compared to conventional
architectures. By shifting smart grid control from reactive correction to
proactive optimization, FOGNITE represents a step toward more intelligent,
adaptive, and sustainable energy infrastructures

### 3. [Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey of Algorithms, Execution, and Open Challenges](http://arxiv.org/pdf/2507.16731v1)

Authors: Senyao Li, Haozhao Wang, Wenchao Xu, Rui Zhang, Song Guo, Jingling Yuan, Xian Zhong, Tianwei Zhang, Ruixuan Li

As large language models (LLMs) evolve, deploying them solely in the cloud or
compressing them for edge devices has become inadequate due to concerns about
latency, privacy, cost, and personalization. This survey explores a
collaborative paradigm in which cloud-based LLMs and edge-deployed small
language models (SLMs) cooperate across both inference and training. We present
a unified taxonomy of edge-cloud collaboration strategies. For inference, we
categorize approaches into task assignment, task division, and mixture-based
collaboration at both task and token granularity, encompassing adaptive
scheduling, resource-aware offloading, speculative decoding, and modular
routing. For training, we review distributed adaptation techniques, including
parameter alignment, pruning, bidirectional distillation, and
small-model-guided optimization. We further summarize datasets, benchmarks, and
deployment cases, and highlight privacy-preserving methods and vertical
applications. This survey provides the first systematic foundation for LLM-SLM
collaboration, bridging system and algorithm co-design to enable efficient,
scalable, and trustworthy edge-cloud intelligence.

### 4. [Cooling Matters: Benchmarking Large Language Models and Vision-Language Models on Liquid-Cooled Versus Air-Cooled H100 GPU Systems](http://arxiv.org/pdf/2507.16781v1)

Authors: Imran Latif, Muhammad Ali Shafique, Hayat Ullah, Alex C. Newkirk, Xi Yu, Arslan Munir

The unprecedented growth in artificial intelligence (AI) workloads, recently
dominated by large language models (LLMs) and vision-language models (VLMs),
has intensified power and cooling demands in data centers. This study
benchmarks LLMs and VLMs on two HGX nodes, each with 8x NVIDIA H100 graphics
processing units (GPUs), using liquid and air cooling. Leveraging GPU Burn,
Weights and Biases, and IPMItool, we collect detailed thermal, power, and
computation data. Results show that the liquid-cooled systems maintain GPU
temperatures between 41-50 degrees Celsius, while the air-cooled counterparts
fluctuate between 54-72 degrees Celsius under load. This thermal stability of
liquid-cooled systems yields 17 percent higher performance (54 TFLOPs per GPU
vs. 46 TFLOPs per GPU), improved performance per watt, reduced energy overhead,
and greater system efficiency than the air-cooled counterparts. These findings
underscore the energy and sustainability benefits of liquid cooling, offering a
compelling path forward for hyperscale data centers s

### 5. [DP2Guard: A Lightweight and Byzantine-Robust Privacy-Preserving Federated Learning Scheme for Industrial IoT](http://arxiv.org/pdf/2507.16134v1)

Authors: Baofu Han, Bing Li, Yining Qi, Raja Jurdak, Kaibin Huang, Chau Yuen

Privacy-Preserving Federated Learning (PPFL) has emerged as a secure
distributed Machine Learning (ML) paradigm that aggregates locally trained
gradients without exposing raw data. To defend against model poisoning threats,
several robustness-enhanced PPFL schemes have been proposed by integrating
anomaly detection. Nevertheless, they still face two major challenges: (1) the
reliance on heavyweight encryption techniques results in substantial
communication and computation overhead; and (2) single-strategy defense
mechanisms often fail to provide sufficient robustness against adaptive
adversaries. To overcome these challenges, we propose DP2Guard, a lightweight
PPFL framework that enhances both privacy and robustness. DP2Guard leverages a
lightweight gradient masking mechanism to replace costly cryptographic
operations while ensuring the privacy of local gradients. A hybrid defense
strategy is proposed, which extracts gradient features using singular value
decomposition and cosine similarity, and applies a clustering algorithm to
effectively identify malicious gradients. Additionally, DP2Guard adopts a trust
score-based adaptive aggregation scheme that adjusts client weights according
to historical behavior, while blockchain records aggregated results and trust
scores to ensure tamper-proof and auditable training. Extensive experiments
conducted on two public datasets demonstrate that DP2Guard effectively defends
against four advanced poisoning attacks while ensuring privacy with reduced
communication and computation costs.

### 6. [AcceleratedKernels.jl: Cross-Architecture Parallel Algorithms from a Unified, Transpiled Codebase](http://arxiv.org/pdf/2507.16710v1)

Authors: Andrei-Leonard Nicusan, Dominik Werner, Simon Branford, Simon Hartley, Andrew J. Morris, Kit Windows-Yule

AcceleratedKernels.jl is introduced as a backend-agnostic library for
parallel computing in Julia, natively targeting NVIDIA, AMD, Intel, and Apple
accelerators via a unique transpilation architecture. Written in a unified,
compact codebase, it enables productive parallel programming with minimised
implementation and usage complexities. Benchmarks of arithmetic-heavy kernels
show performance on par with C and OpenMP-multithreaded CPU implementations,
with Julia sometimes offering more consistent and predictable numerical
performance than conventional C compilers. Exceptional composability is
highlighted as simultaneous CPU-GPU co-processing is achievable - such as
CPU-GPU co-sorting - with transparent use of hardware-specialised MPI
implementations. Tests on the Baskerville Tier 2 UK HPC cluster achieved
world-class sorting throughputs of 538-855 GB/s using 200 NVIDIA A100 GPUs,
comparable to the highest literature-reported figure of 900 GB/s achieved on
262,144 CPU cores. The use of direct NVLink GPU-to-GPU interconnects resulted
in a 4.93x speedup on average; normalised by a combined capital, running and
environmental cost, communication-heavy HPC tasks only become economically
viable on GPUs if GPUDirect interconnects are employed.

### 7. [Parallel Ray Tracing of Black Hole Images Using the Schwarzschild Metric](http://arxiv.org/pdf/2507.16165v1)

Authors: Liam Naddell, Marcelo Ponce

Rendering images of black holes by utilizing ray tracing techniques is a
common methodology employed in many aspects of scientific and astrophysical
visualizations. Similarly, general ray tracing techniques are widely used in
areas related to computer graphics. In this work we describe the implementation
of a parallel open-source program that can ray trace images in the presence of
a black hole geometry. We do this by combining a couple of different techniques
usually present in parallel scientific computing, such as, mathematical
approximations, utilization of scientific libraries, shared-memory and
distributed-memory parallelism.

### 8. [Improved Wake-Up Time For Euclidean Freeze-Tag Problem](http://arxiv.org/pdf/2507.16269v1)

Authors: Sharareh Alipour, Arash Ahadi, Kajal Baghestani

The Freeze-Tag Problem (FTP) involves activating a set of initially asleep
robots as quickly as possible, starting from a single awake robot. Once
activated, a robot can assist in waking up other robots. Each active robot
moves at unit speed. The objective is to minimize the makespan, i.e., the time
required to activate the last robot. A key performance measure is the wake-up
ratio, defined as the maximum time needed to activate any number of robots in
any primary positions. This work focuses on the geometric (Euclidean) version
of FTP in $\mathbb{R}^d$ under the $\ell_p$ norm, where the initial distance
between each asleep robot and the single active robot is at most 1. For
$(\mathbb{R}^2, \ell_2)$, we improve the previous upper bound of 4.62 ([7],
CCCG 2024) to 4.31. Note that it is known that 3.82 is a lower bound for the
wake-up ratio. In $\mathbb{R}^3$, we propose a new strategy that achieves a
wake-up ratio of 12 for $(\mathbb{R}^3, \ell_1)$ and 12.76 for $(\mathbb{R}^3,
\ell_2)$, improving upon the previous bounds of 13 and $13\sqrt{3}$,
respectively, reported in [2].

### 9. [Reducing GPU Memory Fragmentation via Spatio-Temporal Planning for Efficient Large-Scale Model Training](http://arxiv.org/pdf/2507.16274v1)

Authors: Zixiao Huang, Junhao Hu, Hao Lin, Chunyang Zhu, Yueran Tang, Quanlu Zhang, Zhen Guo, Zhenhua Li, Shengen Yan, Zhenhua Zhu, Guohao Dai, Yu Wang

The rapid scaling of large language models (LLMs) has significantly increased
GPU memory pressure, which is further aggravated by training optimization
techniques such as virtual pipeline and recomputation that disrupt tensor
lifespans and introduce considerable memory fragmentation. Default GPU memory
allocators of popular deep learning frameworks like PyTorch use online
strategies without knowledge of tensor lifespans, which can waste up to 43\% of
memory and cause out-of-memory errors, rendering optimization techniques
ineffective or even unusable.
  To address this, we introduce STWeaver, a GPU memory allocator for deep
learning frameworks that reduces fragmentation by exploiting the spatial and
temporal regularity in memory allocation behaviors of training workloads.
STWeaver introduces a novel paradigm that combines offline planning with online
allocation. The offline planning leverages spatio-temporal regularities to
generate a near-optimal allocation plan, while the online allocation handles
complex and dynamic models such as Mixture-of-Experts (MoE). Built as a
pluggable PyTorch allocator, STWeaver reduces fragmentation ratio on average by
79.2\% (up to 100\%) across both dense and sparse models, with negligible
overhead. This enables more efficient, high-throughput training configurations
and improves performance by up to 32.5\%.

### 10. [An Experimental Study of Split-Learning TinyML on Ultra-Low-Power Edge/IoT Nodes](http://arxiv.org/pdf/2507.16594v1)

Authors: Zied Jenhani, Mounir Bensalem, Jasenka Dizdarević, Admela Jukan

Running deep learning inference directly on ultra-low-power edge/IoT nodes
has been limited by the tight memory and compute budgets of microcontrollers.
Split learning (SL) addresses this limitation in which it executes part of the
inference process on the sensor and off-loads the remainder to a companion
device. In the context of constrained devices and the related impact of
low-power, over-the-air transport protocols, the performance of split learning
remains largely unexplored. TO the best of our knowledge, this paper presents
the first end-to-end TinyML + SL testbed built on Espressif ESP32-S3 boards,
designed to benchmark the over-the-air performance of split learning TinyML in
edge/IoT environments. We benchmark the performance of a MobileNetV2 image
recognition model, which is quantized to 8-bit integers, partitioned, and
delivered to the nodes via over-the-air updates. The intermediate activations
are exchanged through different wireless communication methods: ESP-NOW, BLE,
and traditional UDP/IP and TCP/IP, enabling a head-to-head comparison on
identical hardware. Measurements show that splitting the model after
block_16_project_BN layer generates a 5.66 kB tensor that traverses the link in
3.2 ms, when UDP is used, achieving a steady-state round-trip latency of 5.8 s.
ESP-NOW presents the most favorable RTT performance 3.7 s; BLE extends battery
life further but increases latency beyond 10s.

### Discrete Mathematics

### 1. [An Exact Solver for Maximizing a Submodular Function Subject to a Knapsack Constraint](http://arxiv.org/pdf/2507.16149v1)

Authors: Sabine Münch, Stephen Raach

We study the problem of maximizing a monotone increasing submodular function
over a set of weighted elements subject to a knapsack constraint.
  Although this problem is NP-hard, many applications require exact solutions,
as approximate solutions are often insufficient in practice.
  To address this need, we propose an exact branch-and-bound algorithm tailored
for the submodular knapsack problem and introduce several acceleration
techniques to enhance its efficiency. We evaluate these techniques on instances
of three benchmark problems and compare the proposed solvers to two solvers by
Sakaue and Ishihata, which are considered state-of-the-art, demonstrating that
the presented methods outperform the existing methods.

### 2. [Algorithmic methods of finite discrete structures. Topological graph drawing (part IV)](http://arxiv.org/pdf/2507.16759v1)

Authors: Sergey Kurapov, Maxim Davidovsky

The chapter presents mathematical models intended for creating a topological
drawing of a non-separable non-planar graph based on the methods of G. Ringel's
vertex rotation theory. The induced system of cycles generates a topological
drawing of a certain thickness. A method for determining the location of
imaginary vertices by finding the intersection of connections on a plane is
presented. A topological drawing of a maximum planar subgraph is used as a
basis.

### 3. [An unconditional lower bound for the active-set method in convex quadratic maximization](http://arxiv.org/pdf/2507.16648v1)

Authors: Eleon Bach, Yann Disser, Sophie Huiberts, Nils Mosis

We prove that the active-set method needs an exponential number of iterations
in the worst-case to maximize a convex quadratic function subject to linear
constraints, regardless of the pivot rule used. This substantially improves
over the best previously known lower bound [IPCO 2025], which needs objective
functions of polynomial degrees $\omega(\log d)$ in dimension $d$, to a bound
using a convex polynomial of degree 2. In particular, our result firmly
resolves the open question [IPCO 2025] of whether a constant degree suffices,
and it represents significant progress towards linear objectives, where the
active-set method coincides with the simplex method and a lower bound for all
pivot rules would constitute a major breakthrough.
  Our result is based on a novel extended formulation, recursively constructed
using deformed products. Its key feature is that it projects onto a polygonal
approximation of a parabola while preserving all of its exponentially many
vertices. We define a quadratic objective that forces the active-set method to
follow the parabolic boundary of this projection, without allowing any
shortcuts along chords corresponding to edges of its full-dimensional preimage.

### Data Structures and Algorithms

### 1. [Longest Unbordered Factors on Run-Length Encoded Strings](http://arxiv.org/pdf/2507.16285v1)

Authors: Shoma Sekizaki, Takuya Mieno

A border of a string is a non-empty proper prefix of the string that is also
a suffix. A string is unbordered if it has no border. The longest unbordered
factor is a fundamental notion in stringology, closely related to string
periodicity. This paper addresses the longest unbordered factor problem: given
a string of length $n$, the goal is to compute its longest factor that is
unbordered. While recent work has achieved subquadratic and near-linear time
algorithms for this problem, the best known worst-case time complexity remains
$O(n \log n)$ [Kociumaka et al., ISAAC 2018]. In this paper, we investigate the
problem in the context of compressed string processing, particularly focusing
on run-length encoded (RLE) strings. We first present a simple yet crucial
structural observation relating unbordered factors and RLE-compressed strings.
Building on this, we propose an algorithm that solves the problem in $O(m^{1.5}
\log^2 m)$ time and $O(m \log^2 m)$ space, where $m$ is the size of the
RLE-compressed input string. To achieve this, our approach simulates a key idea
from the $O(n^{1.5})$-time algorithm by [Gawrychowski et al., SPIRE 2015],
adapting it to the RLE setting through new combinatorial insights. When the RLE
size $m$ is sufficiently small compared to $n$, our algorithm may show
linear-time behavior in $n$, potentially leading to improved performance over
existing methods in such cases.

### 2. [An Exact Solver for Maximizing a Submodular Function Subject to a Knapsack Constraint](http://arxiv.org/pdf/2507.16149v1)

Authors: Sabine Münch, Stephen Raach

We study the problem of maximizing a monotone increasing submodular function
over a set of weighted elements subject to a knapsack constraint.
  Although this problem is NP-hard, many applications require exact solutions,
as approximate solutions are often insufficient in practice.
  To address this need, we propose an exact branch-and-bound algorithm tailored
for the submodular knapsack problem and introduce several acceleration
techniques to enhance its efficiency. We evaluate these techniques on instances
of three benchmark problems and compare the proposed solvers to two solvers by
Sakaue and Ishihata, which are considered state-of-the-art, demonstrating that
the presented methods outperform the existing methods.

### 3. [Best-of-Both-Worlds Guarantees with Fairer Endings](http://arxiv.org/pdf/2507.16209v1)

Authors: Telikepalli Kavitha, Surya Panchapakesan, Rohit Vaish, Vignesh Viswanathan, Jatin Yadav

Fair allocation of indivisible goods is a fundamental problem at the
interface of economics and computer science. Traditional approaches focus
either on randomized allocations that are fair in expectation or deterministic
allocations that are approximately fair. Recent work reconciles both these
approaches via best-of-both-worlds guarantees, wherein one seeks randomized
allocations that are fair in expectation (ex-ante fair) while being supported
on approximately fair allocations (ex-post fair). Prior work has shown that
under additive valuations, there always exists a randomized allocation that is
ex-ante stochastic-dominance envy-free (sd-EF) and ex-post envy-free up to one
good (EF1).
  Our work is motivated by the goal of achieving stronger ex-post fairness
guarantees such as envy-freeness up to any good (EFX) along with meaningful
ex-ante guarantees. We make the following contributions:
  1) We first consider lexicographic preferences, a subdomain of additive
valuations where ex-post EFX allocations always exist and can be computed
efficiently. On the negative side, we show that ex-ante sd-EF is fundamentally
incompatible with ex-post EFX, prompting a relaxation of the ex-ante benchmark.
We then present a poly. time algorithm that achieves ex-post EFX and PO
together with ex-ante 9/10-EF. Our algorithm uses dependent rounding and
leverages structural properties of EFX and PO allocations.
  2)For monotone valuations, we study EFX-with-charity: a relaxation of EFX
where some goods remain unallocated, with no agent envying the unallocated
pool. We show that ex-post EFX-with-charity can be achieved alongside ex-ante
0.5-EF.
  3)Finally, for subadditive valuations, we strengthen our previous ex-post
guarantee to EFX-with-bounded-charity, where at most n-1 goods (n= no. of
agents) remain unallocated, at the price of weakening the ex-ante guarantee to
0.5-proportionality.

### 4. [Toward a Lightweight and Robust Design for Caching with Predictions](http://arxiv.org/pdf/2507.16242v1)

Authors: Peng Chen, Hailiang Zhao, Jiaji Zhang, Xueyan Tang, Yixuan Wang, Shuiguang Deng

The online caching problem aims to minimize cache misses when serving a
sequence of requests under a limited cache size. While naive learning-augmented
caching algorithms achieve ideal $1$-consistency, they lack robustness
guarantees. Existing robustification methods either sacrifice $1$-consistency
or introduce significant computational overhead. In this paper, we introduce
\textsc{Guard}, a lightweight robustification framework that enhances the
robustness of a broad class of learning-augmented caching algorithms to $2H_k +
2$, while preserving their $1$-consistency. \textsc{Guard} achieves the current
best-known trade-off between consistency and robustness, with only
$\mathcal{O}(1)$ additional per-request overhead, thereby maintaining the
original time complexity of the base algorithm. Extensive experiments across
multiple real-world datasets and prediction models validate the effectiveness
of \textsc{Guard} in practice.

### 5. [The Cost of Compression: Tight Quadratic Black-Box Attacks on Sketches for $\ell_2$ Norm Estimation](http://arxiv.org/pdf/2507.16345v1)

Authors: Sara Ahmadian, Edith Cohen, Uri Stemmer

Dimensionality reduction via linear sketching is a powerful and widely used
technique, but it is known to be vulnerable to adversarial inputs. We study the
black-box adversarial setting, where a fixed, hidden sketching matrix A in
$R^{k X n}$ maps high-dimensional vectors v $\in R^n$ to lower-dimensional
sketches A v in $R^k$, and an adversary can query the system to obtain
approximate ell2-norm estimates that are computed from the sketch.
  We present a universal, nonadaptive attack that, using tilde(O)($k^2$)
queries, either causes a failure in norm estimation or constructs an
adversarial input on which the optimal estimator for the query distribution
(used by the attack) fails. The attack is completely agnostic to the sketching
matrix and to the estimator: It applies to any linear sketch and any query
responder, including those that are randomized, adaptive, or tailored to the
query distribution.
  Our lower bound construction tightly matches the known upper bounds of
tilde(Omega)($k^2$), achieved by specialized estimators for Johnson
Lindenstrauss transforms and AMS sketches. Beyond sketching, our results
uncover structural parallels to adversarial attacks in image classification,
highlighting fundamental vulnerabilities of compressed representations.

### 6. [An unconditional lower bound for the active-set method in convex quadratic maximization](http://arxiv.org/pdf/2507.16648v1)

Authors: Eleon Bach, Yann Disser, Sophie Huiberts, Nils Mosis

We prove that the active-set method needs an exponential number of iterations
in the worst-case to maximize a convex quadratic function subject to linear
constraints, regardless of the pivot rule used. This substantially improves
over the best previously known lower bound [IPCO 2025], which needs objective
functions of polynomial degrees $\omega(\log d)$ in dimension $d$, to a bound
using a convex polynomial of degree 2. In particular, our result firmly
resolves the open question [IPCO 2025] of whether a constant degree suffices,
and it represents significant progress towards linear objectives, where the
active-set method coincides with the simplex method and a lower bound for all
pivot rules would constitute a major breakthrough.
  Our result is based on a novel extended formulation, recursively constructed
using deformed products. Its key feature is that it projects onto a polygonal
approximation of a parabola while preserving all of its exponentially many
vertices. We define a quadratic objective that forces the active-set method to
follow the parabolic boundary of this projection, without allowing any
shortcuts along chords corresponding to edges of its full-dimensional preimage.

### Emerging Technologies

### 1. [Quantum Annealing Hyperparameter Analysis for Optimal Sensor Placement in Production Environments](http://arxiv.org/pdf/2507.16584v1)

Authors: Nico Kraus, Marvin Erdmann, Alexander Kuzmany, Daniel Porawski, Jonas Stein

To increase efficiency in automotive manufacturing, newly produced vehicles
can move autonomously from the production line to the distribution area. This
requires an optimal placement of sensors to ensure full coverage while
minimizing the number of sensors used. The underlying optimization problem
poses a computational challenge due to its large-scale nature. Currently,
classical solvers rely on heuristics, often yielding non-optimal solutions for
large instances, resulting in suboptimal sensor distributions and increased
operational costs.
  We explore quantum computing methods that may outperform classical heuristics
in the future. We implemented quantum annealing with D-Wave, transforming the
problem into a quadratic unconstrained binary optimization formulation with
one-hot and binary encoding. Hyperparameters like the penalty terms and the
annealing time are optimized and the results are compared with default
parameter settings.
  Our results demonstrate that quantum annealing is capable of solving
instances derived from real-world scenarios. Through the use of decomposition
techniques, we are able to scale the problem size further, bringing it closer
to practical, industrial applicability. Through this work, we provide key
insights into the importance of quantum annealing parametrization,
demonstrating how quantum computing could contribute to cost-efficient,
large-scale optimization problems once the hardware matures.

### 2. [Towards Railway Domain Adaptation for LiDAR-based 3D Detection: Road-to-Rail and Sim-to-Real via SynDRA-BBox](http://arxiv.org/pdf/2507.16413v1)

Authors: Xavier Diaz, Gianluca D'Amico, Raul Dominguez-Sanchez, Federico Nesti, Max Ronecker, Giorgio Buttazzo

In recent years, interest in automatic train operations has significantly
increased. To enable advanced functionalities, robust vision-based algorithms
are essential for perceiving and understanding the surrounding environment.
However, the railway sector suffers from a lack of publicly available
real-world annotated datasets, making it challenging to test and validate new
perception solutions in this domain. To address this gap, we introduce
SynDRA-BBox, a synthetic dataset designed to support object detection and other
vision-based tasks in realistic railway scenarios. To the best of our
knowledge, is the first synthetic dataset specifically tailored for 2D and 3D
object detection in the railway domain, the dataset is publicly available at
https://syndra.retis.santannapisa.it. In the presented evaluation, a
state-of-the-art semi-supervised domain adaptation method, originally developed
for automotive perception, is adapted to the railway context, enabling the
transferability of synthetic data to 3D object detection. Experimental results
demonstrate promising performance, highlighting the effectiveness of synthetic
datasets and domain adaptation techniques in advancing perception capabilities
for railway environments.

### 3. [Active RISs: Modeling and Optimization](http://arxiv.org/pdf/2507.16499v1)

Authors: Recep Akif Tasci, Panagiotis Gavriilidis, Ertugrul Basar, George C. Alexandropoulos

Reconfigurable Intelligent Surfaces (RIS)-empowered communication has emerged
as a transformative technology for next generation wireless networks, enabling
the programmable shaping of the propagation environment. However, conventional
RISs are fundamentally limited by the double path loss effect, which severely
attenuates the reflected signals. To overcome this, active RIS architectures,
capable of amplifying impinging signals, have been proposed. This chapter
investigates the modeling, performance analysis, and optimization of active
RISs, focusing on two hardware designs: a dual-RIS structure with a single
Power Amplifier (PA), and a reflection amplification structure at the unit cell
level using tunnel diodes. For the PA-based design, a comprehensive
mathematical model is developed, and closed-form expressions for the received
signal-to-noise ratio, bit error probability, and Energy Efficiency (EE) are
derived. An optimization framework for configuring the phase shifts and
amplifier gain is proposed to maximize system capacity under power constraints.
Regarding the second design, the integration of a tunnel diode into the unit
cell is carefully studied by analyzing its I-V characteristic, enabling the
derivation of the negative resistance range and the power consumption model.
Furthermore, the intrinsic phase-amplitude coupling of the reflection
coefficient is characterized through compact linear algebra formulations,
enabling practical optimization of active RISs. Extensive numerical simulations
validate the theoretical analyses, demonstrating that active RISs can
effectively overcome the double path loss limitation and achieve favorable EE
trade-offs compared to passive RISs. Finally, the trade-off between the
available power budget and the number of active elements is examined, revealing
that a higher number of active elements does not always lead to optimal
performance.

### 4. [AI-enhanced conversational agents for personalized asthma support Factors for engagement, value and efficacy](http://arxiv.org/pdf/2507.16735v1)

Authors: Laura Moradbakhti, Dorian Peters, Jennifer K. Quint, Björn Schuller, Darren Cook, Rafael A. Calvo

Asthma-related deaths in the UK are the highest in Europe, and only 30% of
patients access basic care. There is a need for alternative approaches to
reaching people with asthma in order to provide health education,
self-management support and bridges to care. Automated conversational agents
(specifically, mobile chatbots) present opportunities for providing alternative
and individually tailored access to health education, self-management support
and risk self-assessment. But would patients engage with a chatbot, and what
factors influence engagement? We present results from a patient survey (N=1257)
devised by a team of asthma clinicians, patients, and technology developers,
conducted to identify optimal factors for efficacy, value and engagement for a
chatbot. Results indicate that most adults with asthma (53%) are interested in
using a chatbot and the patients most likely to do so are those who believe
their asthma is more serious and who are less confident about self-management.
Results also indicate enthusiasm for 24/7 access, personalisation, and for
WhatsApp as the preferred access method (compared to app, voice assistant, SMS
or website). Obstacles to uptake include security/privacy concerns and
skepticism of technological capabilities. We present detailed findings and
consolidate these into 7 recommendations for developers for optimising efficacy
of chatbot-based health support.

### 5. [Multi-RIS-Empowered Communication Systems: Capacity Analysis and Optimization](http://arxiv.org/pdf/2507.16767v1)

Authors: Aris L. Moustakas, George C. Alexandropoulos

In this chapter, using statistical physics methods, asymptotic closed-form
expressions for the mean and variance of the mutual information for a
multi-antenna transmitter-receiver pair in the presence of multiple
Reconfigurable Intelligent Surfaces (RISs) are presented. While nominally valid
in the large-system limit, it is shown that the derived Gaussian approximation
for the mutual information can be quite accurate, even for modest-sized antenna
arrays and metasurfaces. The above results are particularly useful when
fast-fading conditions are present, which renders channel estimation
challenging. The derived analysis indicates that, when the channel close to an
RIS is correlated, for instance due to small angle spread which is reasonable
for wireless systems with increasing carrier frequencies, the communication
link benefits significantly from statistical RIS optimization, resulting in
gains that are surprisingly higher than the nearly uncorrelated case. More
importantly, the presented novel asymptotic properties of the correlation
matrices of the impinging and outgoing signals at the RISs can be deployed to
optimize the metasurfaces without brute-force numerical optimization. The
numerical investigation demonstrates that, when the desired reflection from any
of the RISs departs significantly from geometrical optics, the metasurfaces can
be optimized to provide robust communication links, without significant need
for their optimal placement.

### 6. [Designing for Difference: How Human Characteristics Shape Perceptions of Collaborative Robots](http://arxiv.org/pdf/2507.16480v1)

Authors: Sabrina Livanec, Laura Londoño, Michael Gorki, Adrian Röfer, Abhinav Valada, Andrea Kiesel

The development of assistive robots for social collaboration raises critical
questions about responsible and inclusive design, especially when interacting
with individuals from protected groups such as those with disabilities or
advanced age. Currently, research is scarce on how participants assess varying
robot behaviors in combination with diverse human needs, likely since
participants have limited real-world experience with advanced domestic robots.
In the current study, we aim to address this gap while using methods that
enable participants to assess robot behavior, as well as methods that support
meaningful reflection despite limited experience. In an online study, 112
participants (from both experimental and control groups) evaluated 7 videos
from a total of 28 variations of human-robot collaboration types. The
experimental group first completed a cognitive-affective mapping (CAM) exercise
on human-robot collaboration before providing their ratings. Although CAM
reflection did not significantly affect overall ratings, it led to more
pronounced assessments for certain combinations of robot behavior and human
condition. Most importantly, the type of human-robot collaboration influences
the assessment. Antisocial robot behavior was consistently rated as the lowest,
while collaboration with aged individuals elicited more sensitive evaluations.
Scenarios involving object handovers were viewed more positively than those
without them. These findings suggest that both human characteristics and
interaction paradigms influence the perceived acceptability of collaborative
robots, underscoring the importance of prosocial design. They also highlight
the potential of reflective methods, such as CAM, to elicit nuanced feedback,
supporting the development of user-centered and socially responsible robotic
systems tailored to diverse populations.

### Graphics

### 1. [MMS Player: an open source software for parametric data-driven animation of Sign Language avatars](http://arxiv.org/pdf/2507.16463v1)

Authors: Fabrizio Nunnari, Shailesh Mishra, Patrick Gebhard

This paper describes the MMS-Player, an open source software able to
synthesise sign language animations from a novel sign language representation
format called MMS (MultiModal Signstream). The MMS enhances gloss-based
representations by adding information on parallel execution of signs, timing,
and inflections. The implementation consists of Python scripts for the popular
Blender 3D authoring tool and can be invoked via command line or HTTP API.
Animations can be rendered as videos or exported in other popular 3D animation
exchange formats. The software is freely available under GPL-3.0 license at
https://github.com/DFKI-SignLanguage/MMS-Player.

### 2. [Parallel Ray Tracing of Black Hole Images Using the Schwarzschild Metric](http://arxiv.org/pdf/2507.16165v1)

Authors: Liam Naddell, Marcelo Ponce

Rendering images of black holes by utilizing ray tracing techniques is a
common methodology employed in many aspects of scientific and astrophysical
visualizations. Similarly, general ray tracing techniques are widely used in
areas related to computer graphics. In this work we describe the implementation
of a parallel open-source program that can ray trace images in the presence of
a black hole geometry. We do this by combining a couple of different techniques
usually present in parallel scientific computing, such as, mathematical
approximations, utilization of scientific libraries, shared-memory and
distributed-memory parallelism.

### Computer Science and Game Theory

### 1. [Best-of-Both-Worlds Guarantees with Fairer Endings](http://arxiv.org/pdf/2507.16209v1)

Authors: Telikepalli Kavitha, Surya Panchapakesan, Rohit Vaish, Vignesh Viswanathan, Jatin Yadav

Fair allocation of indivisible goods is a fundamental problem at the
interface of economics and computer science. Traditional approaches focus
either on randomized allocations that are fair in expectation or deterministic
allocations that are approximately fair. Recent work reconciles both these
approaches via best-of-both-worlds guarantees, wherein one seeks randomized
allocations that are fair in expectation (ex-ante fair) while being supported
on approximately fair allocations (ex-post fair). Prior work has shown that
under additive valuations, there always exists a randomized allocation that is
ex-ante stochastic-dominance envy-free (sd-EF) and ex-post envy-free up to one
good (EF1).
  Our work is motivated by the goal of achieving stronger ex-post fairness
guarantees such as envy-freeness up to any good (EFX) along with meaningful
ex-ante guarantees. We make the following contributions:
  1) We first consider lexicographic preferences, a subdomain of additive
valuations where ex-post EFX allocations always exist and can be computed
efficiently. On the negative side, we show that ex-ante sd-EF is fundamentally
incompatible with ex-post EFX, prompting a relaxation of the ex-ante benchmark.
We then present a poly. time algorithm that achieves ex-post EFX and PO
together with ex-ante 9/10-EF. Our algorithm uses dependent rounding and
leverages structural properties of EFX and PO allocations.
  2)For monotone valuations, we study EFX-with-charity: a relaxation of EFX
where some goods remain unallocated, with no agent envying the unallocated
pool. We show that ex-post EFX-with-charity can be achieved alongside ex-ante
0.5-EF.
  3)Finally, for subadditive valuations, we strengthen our previous ex-post
guarantee to EFX-with-bounded-charity, where at most n-1 goods (n= no. of
agents) remain unallocated, at the price of weakening the ex-ante guarantee to
0.5-proportionality.

### 2. [Smooth Games of Configuration in the Linear-Quadratic Setting](http://arxiv.org/pdf/2507.16611v1)

Authors: Jesse Milzman, Jeffrey Mao, Giuseppe Loianno

Dynamic game theory offers a toolbox for formalizing and solving for both
cooperative and non-cooperative strategies in multi-agent scenarios. However,
the optimal configuration of such games remains largely unexplored. While there
is existing literature on the parametrization of dynamic games, little research
examines this parametrization from a strategic perspective where each agent's
configuration choice is influenced by the decisions of others. In this work, we
introduce the concept of a game of configuration, providing a framework for the
strategic fine-tuning of differential games. We define a game of configuration
as a two-stage game within the setting of finite-horizon, affine-quadratic, AQ,
differential games. In the first stage, each player chooses their corresponding
configuration parameter, which will impact their dynamics and costs in the
second stage. We provide the subgame perfect solution concept and a method for
computing first stage cost gradients over the configuration space. This then
allows us to formulate a gradient-based method for searching for local
solutions to the configuration game, as well as provide necessary conditions
for equilibrium configurations over their downstream (second stage)
trajectories. We conclude by demonstrating the effectiveness of our approach in
example AQ systems, both zero-sum and general-sum.

### Human-Computer Interaction

### 1. [BDIViz: An Interactive Visualization System for Biomedical Schema Matching with LLM-Powered Validation](http://arxiv.org/pdf/2507.16117v1)

Authors: Eden Wu, Dishita G Turakhia, Guande Wu, Christos Koutras, Sarah Keegan, Wenke Liu, Beata Szeitz, David Fenyo, Cláudio T. Silva, Juliana Freire

Biomedical data harmonization is essential for enabling exploratory analyses
and meta-studies, but the process of schema matching - identifying semantic
correspondences between elements of disparate datasets (schemas) - remains a
labor-intensive and error-prone task. Even state-of-the-art automated methods
often yield low accuracy when applied to biomedical schemas due to the large
number of attributes and nuanced semantic differences between them. We present
BDIViz, a novel visual analytics system designed to streamline the schema
matching process for biomedical data. Through formative studies with domain
experts, we identified key requirements for an effective solution and developed
interactive visualization techniques that address both scalability challenges
and semantic ambiguity. BDIViz employs an ensemble approach that combines
multiple matching methods with LLM-based validation, summarizes matches through
interactive heatmaps, and provides coordinated views that enable users to
quickly compare attributes and their values. Our method-agnostic design allows
the system to integrate various schema matching algorithms and adapt to
application-specific needs. Through two biomedical case studies and a
within-subject user study with domain experts, we demonstrate that BDIViz
significantly improves matching accuracy while reducing cognitive load and
curation time compared to baseline approaches.

### 2. [Animal Interaction with Autonomous Mobility Systems: Designing for Multi-Species Coexistence](http://arxiv.org/pdf/2507.16258v1)

Authors: Tram Thi Minh Tran, Xinyan Yu, Marius Hoggenmueller, Callum Paker, Paul Schmitt, Julie Stephany Berrio Perez, Stewart Worrall, Martin Tomitsch

Autonomous mobility systems increasingly operate in environments shared with
animals, from urban pets to wildlife. However, their design has largely focused
on human interaction, with limited understanding of how non-human species
perceive, respond to, or are affected by these systems. Motivated by research
in Animal-Computer Interaction (ACI) and more-than-human design, this study
investigates animal interactions with autonomous mobility through a
multi-method approach combining a scoping review (45 articles), online
ethnography (39 YouTube videos and 11 Reddit discussions), and expert
interviews (8 participants). Our analysis surfaces five key areas of concern:
Physical Impact (e.g., collisions, failures to detect), Behavioural Effects
(e.g., avoidance, stress), Accessibility Concerns (particularly for service
animals), Ethics and Regulations, and Urban Disturbance. We conclude with
design and policy directions aimed at supporting multispecies coexistence in
the age of autonomous systems. This work underscores the importance of
incorporating non-human perspectives to ensure safer, more inclusive futures
for all species.

### 3. [SceneLoom: Communicating Data with Scene Context](http://arxiv.org/pdf/2507.16466v1)

Authors: Lin Gao, Leixian Shen, Yuheng Zhao, Jiexiang Lan, Huamin Qu, Siming Chen

In data-driven storytelling contexts such as data journalism and data videos,
data visualizations are often presented alongside real-world imagery to support
narrative context. However, these visualizations and contextual images
typically remain separated, limiting their combined narrative expressiveness
and engagement. Achieving this is challenging due to the need for fine-grained
alignment and creative ideation. To address this, we present SceneLoom, a
Vision-Language Model (VLM)-powered system that facilitates the coordination of
data visualization with real-world imagery based on narrative intents. Through
a formative study, we investigated the design space of coordination
relationships between data visualization and real-world scenes from the
perspectives of visual alignment and semantic coherence. Guided by the derived
design considerations, SceneLoom leverages VLMs to extract visual and semantic
features from scene images and data visualization, and perform design mapping
through a reasoning process that incorporates spatial organization, shape
similarity, layout consistency, and semantic binding. The system generates a
set of contextually expressive, image-driven design alternatives that achieve
coherent alignments across visual, semantic, and data dimensions. Users can
explore these alternatives, select preferred mappings, and further refine the
design through interactive adjustments and animated transitions to support
expressive data communication. A user study and an example gallery validate
SceneLoom's effectiveness in inspiring creative design and facilitating design
externalization.

### 4. [The Effect of Scale Consistency between Real and Virtual Spaces on Immersion in Exhibition Hybrid Spaces](http://arxiv.org/pdf/2507.16542v1)

Authors: Qiong Wu, Yan Dong, Zipeng Zhang, Ruochen Hu

In exhibition hybrid spaces, scale consistency between real and virtual
spaces is crucial for user immersion. However, there is currently a lack of
systematic research to determine appropriate virtual-to-real mapping ratios.
This study developed an immersive interaction system based on Intel 3D Athlete
Tracking body mapping technology. Two experiments investigated the impact of
virtual space and virtual avatar scale on immersion. Experiment 1 investigated
30 participants' preferences for virtual space scale, while Experiment 2 tested
the effect of 6 different virtual avatar sizes (25%-150%) on immersion. A
5-point Likert scale was used to assess immersion, followed by analysis of
variance and Tukey HSD post-hoc tests. Experiment 1 showed that participants
preferred a virtual space ratio of 130% (mean 127.29%, SD 8.55%). Experiment 2
found that virtual avatar sizes within the 75%-100% range produced optimal
immersion (p < 0.05). Immersion decreased significantly when virtual avatar
sizes deviated from users' actual height (below 50% or above 125%).
Participants were more sensitive to size changes in the 25%-75% range, while
perception was weaker for changes in the 75%-100% range. Virtual environments
slightly larger than real space (130%) and virtual avatars slightly smaller
than users (75%-100%) optimize user immersion. These findings have been applied
in the Intel Global Trade Center exhibition hall, demonstrating actionable
insights for designing hybrid spaces that enhance immersion and coherence.

### 5. [Animated Transition between Node-Link and Parallel Coordinates Visualizations](http://arxiv.org/pdf/2507.16563v1)

Authors: Abdulhaq Adetunji Salako, Hannes Hagen, Christian Tominski

Multi-faceted data visualization typically involves several dedicated views.
To create a comprehensive understanding of the data, users have to mentally
integrate the information from the different views. This integration is
hindered by context switches between views and usually requires interactive
methods such as brushing and linking. Animated transitions have also been shown
to be able to mediate context switches and improve understanding. Yet, most
existing animated transitions consider only basic views showing the same data
facet. In this work, we study how the gap between node-link diagrams, showing
graph structure, and parallel coordinates plots, showing multivariate
attributes, can be narrowed via smooth animated transitions. Based on two
design goals (traceability and swiftness), we outline a partial design space
including several design options. These inform the implementation of two
alternative transition variants: a basic variant with plain interpolation and
an advanced variant that uses our design space and accepted animation
techniques, including staging and staggering. In a preliminary study, we asked
seven participants for qualitative feedback. We found that the swiftness of the
basic variant is preferred, while the traceability of data items is better with
the slower advanced variant.

### 6. [Emergent Cognitive Convergence via Implementation: A Structured Loop Reflecting Four Theories of Mind (A Position Paper)](http://arxiv.org/pdf/2507.16184v1)

Authors: Myung Ho Kim

We report the discovery of a structural convergence across four influential
theories of mind: Kahneman's dual-system theory, Friston's predictive
processing, Minsky's society of mind, and Clark's extended mind-emerging
unintentionally within a practical AI agent architecture called Agentic Flow.
Designed to address limitations in large language models (LLMs), Agentic Flow
comprises five interdependent modules such as Retrieval, Cognition, Control,
Memory, and Action arranged in a recurrent cognitive loop. Although originally
inspired only by Minsky and Clark, the system's structure retrospectively
aligns with computational motifs found in all four theories, including
predictive modeling, associative recall, and error-sensitive control.
  To assess this convergence, we conducted comparative experiments with
baseline LLM agents on multi-step reasoning tasks. The structured agent
achieved 95.8% task success and exhibited strong constraint adherence, while
the baseline system succeeded 62.3% of the time. These results were not aimed
at proving superiority, but at illustrating how theoretical structures may
emerge through practical design choices rather than top-down theory.
  We introduce PEACE as a descriptive meta-architecture that captures
design-level regularities observed in Agentic Flow. Not intended as a new
theory, PEACE provides a shared vocabulary for understanding architectures
shaped by real-world implementation demands. This paper should be read as a
position paper - an exploratory reflection on how implementation can surface
latent structural echoes of cognitive theory, without asserting theoretical
unification.

### 7. [AI or Human? Understanding Perceptions of Embodied Robots with LLMs](http://arxiv.org/pdf/2507.16398v1)

Authors: Lavinia Hriscu, Alberto Sanfeliu, Anais Garrell

The pursuit of artificial intelligence has long been associated to the the
challenge of effectively measuring intelligence. Even if the Turing Test was
introduced as a means of assessing a system intelligence, its relevance and
application within the field of human-robot interaction remain largely
underexplored. This study investigates the perception of intelligence in
embodied robots by performing a Turing Test within a robotic platform. A total
of 34 participants were tasked with distinguishing between AI- and
human-operated robots while engaging in two interactive tasks: an information
retrieval and a package handover. These tasks assessed the robot perception and
navigation abilities under both static and dynamic conditions. Results indicate
that participants were unable to reliably differentiate between AI- and
human-controlled robots beyond chance levels. Furthermore, analysis of
participant responses reveals key factors influencing the perception of
artificial versus human intelligence in embodied robotic systems. These
findings provide insights into the design of future interactive robots and
contribute to the ongoing discourse on intelligence assessment in AI-driven
systems.

### 8. [Introducing Quality Estimation to Machine Translation Post-editing Workflow: An Empirical Study on Its Usefulness](http://arxiv.org/pdf/2507.16515v1)

Authors: Siqi Liu, Guangrong Dai, Dechao Li

This preliminary study investigates the usefulness of sentence-level Quality
Estimation (QE) in English-Chinese Machine Translation Post-Editing (MTPE),
focusing on its impact on post-editing speed and student translators'
perceptions. It also explores the interaction effects between QE and MT
quality, as well as between QE and translation expertise. The findings reveal
that QE significantly reduces post-editing time. The examined interaction
effects were not significant, suggesting that QE consistently improves MTPE
efficiency across medium- and high-quality MT outputs and among student
translators with varying levels of expertise. In addition to indicating
potentially problematic segments, QE serves multiple functions in MTPE, such as
validating translators' evaluations of MT quality and enabling them to
double-check translation outputs. However, interview data suggest that
inaccurate QE may hinder post-editing processes. This research provides new
insights into the strengths and limitations of QE, facilitating its more
effective integration into MTPE workflows to enhance translators' productivity.

### 9. [Evaluating Social Acceptance of eXtended Reality (XR) Agent Technology: A User Study (Extended Version)](http://arxiv.org/pdf/2507.16562v1)

Authors: Megha Quamara, Viktor Schmuck, Cristina Iani, Axel Primavesi, Alexander Plaum, Luca Vigano

In this paper, we present the findings of a user study that evaluated the
social acceptance of eXtended Reality (XR) agent technology, focusing on a
remotely accessible, web-based XR training system developed for journalists.
This system involves user interaction with a virtual avatar, enabled by a
modular toolkit. The interactions are designed to provide tailored training for
journalists in digital-remote settings, especially for sensitive or dangerous
scenarios, without requiring specialized end-user equipment like headsets. Our
research adapts and extends the Almere model, representing social acceptance
through existing attributes such as perceived ease of use and perceived
usefulness, along with added ones like dependability and security in the
user-agent interaction. The XR agent was tested through a controlled experiment
in a real-world setting, with data collected on users' perceptions. Our
findings, based on quantitative and qualitative measurements involving
questionnaires, contribute to the understanding of user perceptions and
acceptance of XR agent solutions within a specific social context, while also
identifying areas for the improvement of XR systems.

### 10. [Disability Across Cultures: A Human-Centered Audit of Ableism in Western and Indic LLMs](http://arxiv.org/pdf/2507.16130v1)

Authors: Mahika Phutane, Aditya Vashistha

People with disabilities (PwD) experience disproportionately high levels of
discrimination and hate online, particularly in India, where entrenched stigma
and limited resources intensify these challenges. Large language models (LLMs)
are increasingly used to identify and mitigate online hate, yet most research
on online ableism focuses on Western audiences with Western AI models. Are
these models adequately equipped to recognize ableist harm in non-Western
places like India? Do localized, Indic language models perform better? To
investigate, we adopted and translated a publicly available ableist speech
dataset to Hindi, and prompted eight LLMs--four developed in the U.S. (GPT-4,
Gemini, Claude, Llama) and four in India (Krutrim, Nanda, Gajendra,
Airavata)--to score and explain ableism. In parallel, we recruited 175 PwD from
both the U.S. and India to perform the same task, revealing stark differences
between groups. Western LLMs consistently overestimated ableist harm, while
Indic LLMs underestimated it. Even more concerning, all LLMs were more tolerant
of ableism when it was expressed in Hindi and asserted Western framings of
ableist harm. In contrast, Indian PwD interpreted harm through intention,
relationality, and resilience--emphasizing a desire to inform and educate
perpetrators. This work provides groundwork for global, inclusive standards of
ableism, demonstrating the need to center local disability experiences in the
design and evaluation of AI systems.

### Information Retrieval

### 1. [Reinforce Lifelong Interaction Value of User-Author Pairs for Large-Scale Recommendation Systems](http://arxiv.org/pdf/2507.16253v1)

Authors: Yisha Li, Lexi Gao, Jingxin Liu, Xiang Gao, Xin Li, Haiyang Lu, Liyin Hong

Recommendation systems (RS) help users find interested content and connect
authors with their target audience. Most research in RS tends to focus either
on predicting users' immediate feedback (like click-through rate) accurately or
improving users' long-term engagement. However, they ignore the influence for
authors and the lifelong interaction value (LIV) of user-author pairs, which is
particularly crucial for improving the prosperity of social community in
short-video platforms. Currently, reinforcement learning (RL) can optimize
long-term benefits and has been widely applied in RS. In this paper, we
introduce RL to Reinforce Lifelong Interaction Value of User-Author pairs
(RLIV-UA) based on each interaction of UA pairs. To address the long intervals
between UA interactions and the large scale of the UA space, we propose a novel
Sparse Cross-Request Interaction Markov Decision Process (SCRI-MDP) and
introduce an Adjacent State Approximation (ASA) method to construct RL training
samples. Additionally, we introduce Multi-Task Critic Learning (MTCL) to
capture the progressive nature of UA interactions (click -> follow -> gift),
where denser interaction signals are leveraged to compensate for the learning
of sparse labels. Finally, an auxiliary supervised learning task is designed to
enhance the convergence of the RLIV-UA model. In offline experiments and online
A/B tests, the RLIV-UA model achieves both higher user satisfaction and higher
platform profits than compared methods.

### 2. [Enhancing patent retrieval using automated patent summarization](http://arxiv.org/pdf/2507.16371v1)

Authors: Eleni Kamateri, Renukswamy Chikkamath, Michail Salampasis, Linda Andersson, Markus Endres

Effective query formulation is a key challenge in long-document Information
Retrieval (IR). This challenge is particularly acute in domain-specific
contexts like patent retrieval, where documents are lengthy, linguistically
complex, and encompass multiple interrelated technical topics. In this work, we
present the application of recent extractive and abstractive summarization
methods for generating concise, purpose-specific summaries of patent documents.
We further assess the utility of these automatically generated summaries as
surrogate queries across three benchmark patent datasets and compare their
retrieval performance against conventional approaches that use entire patent
sections. Experimental results show that summarization-based queries
significantly improve prior-art retrieval effectiveness, highlighting their
potential as an efficient alternative to traditional query formulation
techniques.

### 3. [Generating Search Explanations using Large Language Models](http://arxiv.org/pdf/2507.16692v1)

Authors: Arif Laksito, Mark Stevenson

Aspect-oriented explanations in search results are typically concise text
snippets placed alongside retrieved documents to serve as explanations that
assist users in efficiently locating relevant information. While Large Language
Models (LLMs) have demonstrated exceptional performance for a range of
problems, their potential to generate explanations for search results has not
been explored. This study addresses that gap by leveraging both encoder-decoder
and decoder-only LLMs to generate explanations for search results. The
explanations generated are consistently more accurate and plausible
explanations than those produced by a range of baseline models.

### 4. [Biases in LLM-Generated Musical Taste Profiles for Recommendation](http://arxiv.org/pdf/2507.16708v1)

Authors: Bruno Sguerra, Elena V. Epure, Harin Lee, Manuel Moussallam

One particularly promising use case of Large Language Models (LLMs) for
recommendation is the automatic generation of Natural Language (NL) user taste
profiles from consumption data. These profiles offer interpretable and editable
alternatives to opaque collaborative filtering representations, enabling
greater transparency and user control. However, it remains unclear whether
users consider these profiles to be an accurate representation of their taste,
which is crucial for trust and usability. Moreover, because LLMs inherit
societal and data-driven biases, profile quality may systematically vary across
user and item characteristics. In this paper, we study this issue in the
context of music streaming, where personalization is challenged by a large and
culturally diverse catalog. We conduct a user study in which participants rate
NL profiles generated from their own listening histories. We analyze whether
identification with the profiles is biased by user attributes (e.g.,
mainstreamness, taste diversity) and item features (e.g., genre, country of
origin). We also compare these patterns to those observed when using the
profiles in a downstream recommendation task. Our findings highlight both the
potential and limitations of scrutable, LLM-based profiling in personalized
systems.

### 5. [EBaReT: Expert-guided Bag Reward Transformer for Auto Bidding](http://arxiv.org/pdf/2507.16186v1)

Authors: Kaiyuan Li, Pengyu Wang, Yunshan Peng, Pengjia Yuan, Yanxiang Zeng, Rui Xiang, Yanhua Cheng, Xialong Liu, Peng Jiang

Reinforcement learning has been widely applied in automated bidding.
Traditional approaches model bidding as a Markov Decision Process (MDP).
Recently, some studies have explored using generative reinforcement learning
methods to address long-term dependency issues in bidding environments.
Although effective, these methods typically rely on supervised learning
approaches, which are vulnerable to low data quality due to the amount of
sub-optimal bids and low probability rewards resulting from the low click and
conversion rates. Unfortunately, few studies have addressed these challenges.
  In this paper, we formalize the automated bidding as a sequence
decision-making problem and propose a novel Expert-guided Bag Reward
Transformer (EBaReT) to address concerns related to data quality and
uncertainty rewards. Specifically, to tackle data quality issues, we generate a
set of expert trajectories to serve as supplementary data in the training
process and employ a Positive-Unlabeled (PU) learning-based discriminator to
identify expert transitions. To ensure the decision also meets the expert
level, we further design a novel expert-guided inference strategy. Moreover, to
mitigate the uncertainty of rewards, we consider the transitions within a
certain period as a "bag" and carefully design a reward function that leads to
a smoother acquisition of rewards. Extensive experiments demonstrate that our
model achieves superior performance compared to state-of-the-art bidding
methods.

### 6. [LLM-Enhanced Reranking for Complementary Product Recommendation](http://arxiv.org/pdf/2507.16237v1)

Authors: Zekun Xu, Yudi Zhang

Complementary product recommendation, which aims to suggest items that are
used together to enhance customer value, is a crucial yet challenging task in
e-commerce. While existing graph neural network (GNN) approaches have made
significant progress in capturing complex product relationships, they often
struggle with the accuracy-diversity tradeoff, particularly for long-tail
items. This paper introduces a model-agnostic approach that leverages Large
Language Models (LLMs) to enhance the reranking of complementary product
recommendations. Unlike previous works that use LLMs primarily for data
preprocessing and graph augmentation, our method applies LLM-based prompting
strategies directly to rerank candidate items retrieved from existing
recommendation models, eliminating the need for model retraining. Through
extensive experiments on public datasets, we demonstrate that our approach
effectively balances accuracy and diversity in complementary product
recommendations, with at least 50% lift in accuracy metrics and 2% lift in
diversity metrics on average for the top recommended items across datasets.

### 7. [Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders](http://arxiv.org/pdf/2507.16289v1)

Authors: Danil Gusak, Anna Volodkevich, Anton Klenitskiy, Alexey Vasilev, Evgeny Frolov

Modern sequential recommender systems, ranging from lightweight
transformer-based variants to large language models, have become increasingly
prominent in academia and industry due to their strong performance in the
next-item prediction task. Yet common evaluation protocols for sequential
recommendations remain insufficiently developed: they often fail to reflect the
corresponding recommendation task accurately, or are not aligned with
real-world scenarios.
  Although the widely used leave-one-out split matches next-item prediction, it
permits the overlap between training and test periods, which leads to temporal
leakage and unrealistically long test horizon, limiting real-world relevance.
Global temporal splitting addresses these issues by evaluating on distinct
future periods. However, its applications to sequential recommendations remain
loosely defined, particularly in terms of selecting target interactions and
constructing a validation subset that provides necessary consistency between
validation and test metrics.
  In this paper, we demonstrate that evaluation outcomes can vary significantly
across splitting strategies, influencing model rankings and practical
deployment decisions. To improve reproducibility in both academic and
industrial settings, we systematically compare different splitting strategies
for sequential recommendations across multiple datasets and established
baselines. Our findings show that prevalent splits, such as leave-one-out, may
be insufficiently aligned with more realistic evaluation strategies. Code:
https://github.com/monkey0head/time-to-split

### 8. [Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning in Real-World Applications](http://arxiv.org/pdf/2507.16507v1)

Authors: Jean Lelong, Adnane Errazine, Annabelle Blangero

Conventional Retrieval-Augmented Generation (RAG) systems enhance Large
Language Models (LLMs) but often fall short on complex queries, delivering
limited, extractive answers and struggling with multiple targeted retrievals or
navigating intricate entity relationships. This is a critical gap in
knowledge-intensive domains. We introduce INRAExplorer, an agentic RAG system
for exploring the scientific data of INRAE (France's National Research
Institute for Agriculture, Food and Environment). INRAExplorer employs an
LLM-based agent with a multi-tool architecture to dynamically engage a rich
knowledge base, through a comprehensive knowledge graph derived from open
access INRAE publications. This design empowers INRAExplorer to conduct
iterative, targeted queries, retrieve exhaustive datasets (e.g., all
publications by an author), perform multi-hop reasoning, and deliver
structured, comprehensive answers. INRAExplorer serves as a concrete
illustration of enhancing knowledge interaction in specialized fields.

### 9. [RAVine: Reality-Aligned Evaluation for Agentic Search](http://arxiv.org/pdf/2507.16725v1)

Authors: Yilong Xu, Xiang Long, Zhi Zheng, Jinhua Gao

Agentic search, as a more autonomous and adaptive paradigm of retrieval
augmentation, is driving the evolution of intelligent search systems. However,
existing evaluation frameworks fail to align well with the goals of agentic
search. First, the complex queries commonly used in current benchmarks often
deviate from realistic user search scenarios. Second, prior approaches tend to
introduce noise when extracting ground truth for end-to-end evaluations,
leading to distorted assessments at a fine-grained level. Third, most current
frameworks focus solely on the quality of final answers, neglecting the
evaluation of the iterative process inherent to agentic search. To address
these limitations, we propose RAVine -- a Reality-Aligned eValuation framework
for agentic LLMs with search. RAVine targets multi-point queries and long-form
answers that better reflect user intents, and introduces an attributable ground
truth construction strategy to enhance the accuracy of fine-grained evaluation.
Moreover, RAVine examines model's interaction with search tools throughout the
iterative process, and accounts for factors of efficiency. We benchmark a
series of models using RAVine and derive several insights, which we hope will
contribute to advancing the development of agentic search systems. The code and
datasets are available at https://github.com/SwordFaith/RAVine.

### Machine Learning

### 1. [Optimization and generalization analysis for two-layer physics-informed neural networks without over-parametrization](http://arxiv.org/pdf/2507.16380v1)

Authors: Zhihan Zeng, Yiqi Gu

This work focuses on the behavior of stochastic gradient descent (SGD) in
solving least-squares regression with physics-informed neural networks (PINNs).
Past work on this topic has been based on the over-parameterization regime,
whose convergence may require the network width to increase vastly with the
number of training samples. So, the theory derived from over-parameterization
may incur prohibitive computational costs and is far from practical
experiments. We perform new optimization and generalization analysis for SGD in
training two-layer PINNs, making certain assumptions about the target function
to avoid over-parameterization. Given $\epsilon>0$, we show that if the network
width exceeds a threshold that depends only on $\epsilon$ and the problem, then
the training loss and expected loss will decrease below $O(\epsilon)$.

### 2. [Improving Predictions on Highly Unbalanced Data Using Open Source Synthetic Data Upsampling](http://arxiv.org/pdf/2507.16419v1)

Authors: Ivona Krchova, Michael Platzer, Paul Tiwald

Unbalanced tabular data sets present significant challenges for predictive
modeling and data analysis across a wide range of applications. In many
real-world scenarios, such as fraud detection, medical diagnosis, and rare
event prediction, minority classes are vastly underrepresented, making it
difficult for traditional machine learning algorithms to achieve high accuracy.
These algorithms tend to favor the majority class, leading to biased models
that struggle to accurately represent minority classes. Synthetic data holds
promise for addressing the under-representation of minority classes by
providing new, diverse, and highly realistic samples. This paper presents a
benchmark study on the use of AI-generated synthetic data for upsampling highly
unbalanced tabular data sets.
  We evaluate the effectiveness of an open-source solution, the Synthetic Data
SDK by MOSTLY AI, which provides a flexible and user-friendly approach to
synthetic upsampling for mixed-type data. We compare predictive models trained
on data sets upsampled with synthetic records to those using standard methods,
such as naive oversampling and SMOTE-NC. Our results demonstrate that synthetic
data can improve predictive accuracy for minority groups by generating diverse
data points that fill gaps in sparse regions of the feature space. We show that
upsampled synthetic training data consistently results in top-performing
predictive models, particularly for mixed-type data sets containing very few
minority samples.

### 3. [Improving Model Classification by Optimizing the Training Dataset](http://arxiv.org/pdf/2507.16729v1)

Authors: Morad Tukan, Loay Mualem, Eitan Netzer, Liran Sigalat

In the era of data-centric AI, the ability to curate high-quality training
data is as crucial as model design. Coresets offer a principled approach to
data reduction, enabling efficient learning on large datasets through
importance sampling. However, conventional sensitivity-based coreset
construction often falls short in optimizing for classification performance
metrics, e.g., $F1$ score, focusing instead on loss approximation. In this
work, we present a systematic framework for tuning the coreset generation
process to enhance downstream classification quality. Our method introduces new
tunable parameters--including deterministic sampling, class-wise allocation,
and refinement via active sampling, beyond traditional sensitivity scores.
Through extensive experiments on diverse datasets and classifiers, we
demonstrate that tuned coresets can significantly outperform both vanilla
coresets and full dataset training on key classification metrics, offering an
effective path towards better and more efficient model training.

### 4. [Equivariant Goal Conditioned Contrastive Reinforcement Learning](http://arxiv.org/pdf/2507.16139v1)

Authors: Arsh Tangri, Nichols Crawford Taylor, Haojie Huang, Robert Platt

Contrastive Reinforcement Learning (CRL) provides a promising framework for
extracting useful structured representations from unlabeled interactions. By
pulling together state-action pairs and their corresponding future states,
while pushing apart negative pairs, CRL enables learning nontrivial policies
without manually designed rewards. In this work, we propose Equivariant CRL
(ECRL), which further structures the latent space using equivariant
constraints. By leveraging inherent symmetries in goal-conditioned manipulation
tasks, our method improves both sample efficiency and spatial generalization.
Specifically, we formally define Goal-Conditioned Group-Invariant MDPs to
characterize rotation-symmetric robotic manipulation tasks, and build on this
by introducing a novel rotation-invariant critic representation paired with a
rotation-equivariant actor for Contrastive RL. Our approach consistently
outperforms strong baselines across a range of simulated tasks in both
state-based and image-based settings. Finally, we extend our method to the
offline RL setting, demonstrating its effectiveness across multiple tasks.

### 5. [Learning Patient-Specific Spatial Biomarker Dynamics via Operator Learning for Alzheimer's Disease Progression](http://arxiv.org/pdf/2507.16148v1)

Authors: Jindong Wang, Yutong Mao, Xiao Liu, Wenrui Hao

Alzheimer's disease (AD) is a complex, multifactorial neurodegenerative
disorder with substantial heterogeneity in progression and treatment response.
Despite recent therapeutic advances, predictive models capable of accurately
forecasting individualized disease trajectories remain limited. Here, we
present a machine learning-based operator learning framework for personalized
modeling of AD progression, integrating longitudinal multimodal imaging,
biomarker, and clinical data. Unlike conventional models with prespecified
dynamics, our approach directly learns patient-specific disease operators
governing the spatiotemporal evolution of amyloid, tau, and neurodegeneration
biomarkers. Using Laplacian eigenfunction bases, we construct geometry-aware
neural operators capable of capturing complex brain dynamics. Embedded within a
digital twin paradigm, the framework enables individualized predictions,
simulation of therapeutic interventions, and in silico clinical trials. Applied
to AD clinical data, our method achieves high prediction accuracy exceeding 90%
across multiple biomarkers, substantially outperforming existing approaches.
This work offers a scalable, interpretable platform for precision modeling and
personalized therapeutic optimization in neurodegenerative diseases.

### 6. [LLM Data Selection and Utilization via Dynamic Bi-level Optimization](http://arxiv.org/pdf/2507.16178v1)

Authors: Yang Yu, Kai Han, Hang Zhou, Yehui Tang, Kaiqi Huang, Yunhe Wang, Dacheng Tao

While large-scale training data is fundamental for developing capable large
language models (LLMs), strategically selecting high-quality data has emerged
as a critical approach to enhance training efficiency and reduce computational
costs. Current data selection methodologies predominantly rely on static,
training-agnostic criteria, failing to account for the dynamic model training
and data interactions. In this paper, we propose a new Data Weighting Model
(DWM) to adjust the weight of selected data within each batch to achieve a
dynamic data utilization during LLM training. Specially, to better capture the
dynamic data preference of the trained model, a bi-level optimization framework
is implemented to update the weighting model. Our experiments demonstrate that
DWM enhances the performance of models trained with randomly-selected data, and
the learned weighting model can be transferred to enhance other data selection
methods and models of different sizes. Moreover, we further analyze how a
model's data preferences evolve throughout training, providing new insights
into the data preference of the model during training.

### 7. [The Impact of Pseudo-Science in Financial Loans Risk Prediction](http://arxiv.org/pdf/2507.16182v1)

Authors: Bruno Scarone, Ricardo Baeza-Yates

We study the societal impact of pseudo-scientific assumptions for predicting
the behavior of people in a straightforward application of machine learning to
risk prediction in financial lending. This use case also exemplifies the impact
of survival bias in loan return prediction. We analyze the models in terms of
their accuracy and social cost, showing that the socially optimal model may not
imply a significant accuracy loss for this downstream task. Our results are
verified for commonly used learning methods and datasets. Our findings also
show that there is a natural dynamic when training models that suffer survival
bias where accuracy slightly deteriorates, and whose recall and precision
improves with time. These results act as an illusion, leading the observer to
believe that the system is getting better, when in fact the model is suffering
from increasingly more unfairness and survival bias.

### 8. [EBaReT: Expert-guided Bag Reward Transformer for Auto Bidding](http://arxiv.org/pdf/2507.16186v1)

Authors: Kaiyuan Li, Pengyu Wang, Yunshan Peng, Pengjia Yuan, Yanxiang Zeng, Rui Xiang, Yanhua Cheng, Xialong Liu, Peng Jiang

Reinforcement learning has been widely applied in automated bidding.
Traditional approaches model bidding as a Markov Decision Process (MDP).
Recently, some studies have explored using generative reinforcement learning
methods to address long-term dependency issues in bidding environments.
Although effective, these methods typically rely on supervised learning
approaches, which are vulnerable to low data quality due to the amount of
sub-optimal bids and low probability rewards resulting from the low click and
conversion rates. Unfortunately, few studies have addressed these challenges.
  In this paper, we formalize the automated bidding as a sequence
decision-making problem and propose a novel Expert-guided Bag Reward
Transformer (EBaReT) to address concerns related to data quality and
uncertainty rewards. Specifically, to tackle data quality issues, we generate a
set of expert trajectories to serve as supplementary data in the training
process and employ a Positive-Unlabeled (PU) learning-based discriminator to
identify expert transitions. To ensure the decision also meets the expert
level, we further design a novel expert-guided inference strategy. Moreover, to
mitigate the uncertainty of rewards, we consider the transitions within a
certain period as a "bag" and carefully design a reward function that leads to
a smoother acquisition of rewards. Extensive experiments demonstrate that our
model achieves superior performance compared to state-of-the-art bidding
methods.

### 9. [RealBench: Benchmarking Verilog Generation Models with Real-World IP Designs](http://arxiv.org/pdf/2507.16200v1)

Authors: Pengwei Jin, Di Huang, Chongxiao Li, Shuyao Cheng, Yang Zhao, Xinyao Zheng, Jiaguo Zhu, Shuyi Xing, Bohan Dou, Rui Zhang, Zidong Du, Qi Guo, Xing Hu

The automatic generation of Verilog code using Large Language Models (LLMs)
has garnered significant interest in hardware design automation. However,
existing benchmarks for evaluating LLMs in Verilog generation fall short in
replicating real-world design workflows due to their designs' simplicity,
inadequate design specifications, and less rigorous verification environments.
To address these limitations, we present RealBench, the first benchmark aiming
at real-world IP-level Verilog generation tasks. RealBench features complex,
structured, real-world open-source IP designs, multi-modal and formatted design
specifications, and rigorous verification environments, including 100% line
coverage testbenches and a formal checker. It supports both module-level and
system-level tasks, enabling comprehensive assessments of LLM capabilities.
Evaluations on various LLMs and agents reveal that even one of the
best-performing LLMs, o1-preview, achieves only a 13.3% pass@1 on module-level
tasks and 0% on system-level tasks, highlighting the need for stronger Verilog
generation models in the future. The benchmark is open-sourced at
https://github.com/IPRC-DIP/RealBench.

### 10. [METER: Multi-modal Evidence-based Thinking and Explainable Reasoning -- Algorithm and Benchmark](http://arxiv.org/pdf/2507.16206v1)

Authors: Xu Yang, Qi Zhang, Shuming Jiang, Yaowen Xu, Zhaofan Zou, Hao Sun, Xuelong Li

With the rapid advancement of generative AI, synthetic content across images,
videos, and audio has become increasingly realistic, amplifying the risk of
misinformation. Existing detection approaches predominantly focus on binary
classification while lacking detailed and interpretable explanations of
forgeries, which limits their applicability in safety-critical scenarios.
Moreover, current methods often treat each modality separately, without a
unified benchmark for cross-modal forgery detection and interpretation. To
address these challenges, we introduce METER, a unified, multi-modal benchmark
for interpretable forgery detection spanning images, videos, audio, and
audio-visual content. Our dataset comprises four tracks, each requiring not
only real-vs-fake classification but also evidence-chain-based explanations,
including spatio-temporal localization, textual rationales, and forgery type
tracing. Compared to prior benchmarks, METER offers broader modality coverage
and richer interpretability metrics such as spatial/temporal IoU, multi-class
tracing, and evidence consistency. We further propose a human-aligned,
three-stage Chain-of-Thought (CoT) training strategy combining SFT, DPO, and a
novel GRPO stage that integrates a human-aligned evaluator with CoT reasoning.
We hope METER will serve as a standardized foundation for advancing
generalizable and interpretable forgery detection in the era of generative
media.

### Neural and Evolutionary Computing

### 1. [Spiking neurons as predictive controllers of linear systems](http://arxiv.org/pdf/2507.16495v1)

Authors: Paolo Agliati, André Urbano, Pablo Lanillos, Nasir Ahmad, Marcel van Gerven, Sander Keemink

Neurons communicate with downstream systems via sparse and incredibly brief
electrical pulses, or spikes. Using these events, they control various targets
such as neuromuscular units, neurosecretory systems, and other neurons in
connected circuits. This gave rise to the idea of spiking neurons as
controllers, in which spikes are the control signal. Using instantaneous events
directly as the control inputs, also called `impulse control', is challenging
as it does not scale well to larger networks and has low analytical
tractability. Therefore, current spiking control usually relies on filtering
the spike signal to approximate analog control. This ultimately means spiking
neural networks (SNNs) have to output a continuous control signal,
necessitating continuous energy input into downstream systems. Here, we
circumvent the need for rate-based representations, providing a scalable method
for task-specific spiking control with sparse neural activity. In doing so, we
take inspiration from both optimal control and neuroscience theory, and define
a spiking rule where spikes are only emitted if they bring a dynamical system
closer to a target. From this principle, we derive the required connectivity
for an SNN, and show that it can successfully control linear systems. We show
that for physically constrained systems, predictive control is required, and
the control signal ends up exploiting the passive dynamics of the downstream
system to reach a target. Finally, we show that the control method scales to
both high-dimensional networks and systems. Importantly, in all cases, we
maintain a closed-form mathematical derivation of the network connectivity, the
network dynamics and the control objective. This work advances the
understanding of SNNs as biologically-inspired controllers, providing insight
into how real neurons could exert control, and enabling applications in
neuromorphic hardware design.

### Networking and Internet Architecture

### 1. [The Sweet Danger of Sugar: Debunking Representation Learning for Encrypted Traffic Classification](http://arxiv.org/pdf/2507.16438v1)

Authors: Yuqi Zhao, Giovanni Dettori, Matteo Boffa, Luca Vassio, Marco Mellia

Recently we have witnessed the explosion of proposals that, inspired by
Language Models like BERT, exploit Representation Learning models to create
traffic representations. All of them promise astonishing performance in
encrypted traffic classification (up to 98% accuracy). In this paper, with a
networking expert mindset, we critically reassess their performance. Through
extensive analysis, we demonstrate that the reported successes are heavily
influenced by data preparation problems, which allow these models to find easy
shortcuts - spurious correlation between features and labels - during
fine-tuning that unrealistically boost their performance. When such shortcuts
are not present - as in real scenarios - these models perform poorly. We also
introduce Pcap-Encoder, an LM-based representation learning model that we
specifically design to extract features from protocol headers. Pcap-Encoder
appears to be the only model that provides an instrumental representation for
traffic classification. Yet, its complexity questions its applicability in
practical settings. Our findings reveal flaws in dataset preparation and model
training, calling for a better and more conscious test design. We propose a
correct evaluation methodology and stress the need for rigorous benchmarking.

### 2. [An Experimental Study of Split-Learning TinyML on Ultra-Low-Power Edge/IoT Nodes](http://arxiv.org/pdf/2507.16594v1)

Authors: Zied Jenhani, Mounir Bensalem, Jasenka Dizdarević, Admela Jukan

Running deep learning inference directly on ultra-low-power edge/IoT nodes
has been limited by the tight memory and compute budgets of microcontrollers.
Split learning (SL) addresses this limitation in which it executes part of the
inference process on the sensor and off-loads the remainder to a companion
device. In the context of constrained devices and the related impact of
low-power, over-the-air transport protocols, the performance of split learning
remains largely unexplored. TO the best of our knowledge, this paper presents
the first end-to-end TinyML + SL testbed built on Espressif ESP32-S3 boards,
designed to benchmark the over-the-air performance of split learning TinyML in
edge/IoT environments. We benchmark the performance of a MobileNetV2 image
recognition model, which is quantized to 8-bit integers, partitioned, and
delivered to the nodes via over-the-air updates. The intermediate activations
are exchanged through different wireless communication methods: ESP-NOW, BLE,
and traditional UDP/IP and TCP/IP, enabling a head-to-head comparison on
identical hardware. Measurements show that splitting the model after
block_16_project_BN layer generates a 5.66 kB tensor that traverses the link in
3.2 ms, when UDP is used, achieving a steady-state round-trip latency of 5.8 s.
ESP-NOW presents the most favorable RTT performance 3.7 s; BLE extends battery
life further but increases latency beyond 10s.

### 3. [Latent Space Alignment for AI-Native MIMO Semantic Communications](http://arxiv.org/pdf/2507.16680v1)

Authors: Mario Edoardo Pandolfo, Simone Fiorellino, Emilio Calvanese Strinati, Paolo Di Lorenzo

Semantic communications focus on prioritizing the understanding of the
meaning behind transmitted data and ensuring the successful completion of tasks
that motivate the exchange of information. However, when devices rely on
different languages, logic, or internal representations, semantic mismatches
may occur, potentially hindering mutual understanding. This paper introduces a
novel approach to addressing latent space misalignment in semantic
communications, exploiting multiple-input multiple-output (MIMO)
communications. Specifically, our method learns a MIMO precoder/decoder pair
that jointly performs latent space compression and semantic channel
equalization, mitigating both semantic mismatches and physical channel
impairments. We explore two solutions: (i) a linear model, optimized by solving
a biconvex optimization problem via the alternating direction method of
multipliers (ADMM); (ii) a neural network-based model, which learns semantic
MIMO precoder/decoder under transmission power budget and complexity
constraints. Numerical results demonstrate the effectiveness of the proposed
approach in a goal-oriented semantic communication scenario, illustrating the
main trade-offs between accuracy, communication burden, and complexity of the
solutions.

### Robotics

### 1. [FTIN: Frequency-Time Integration Network for Inertial Odometry](http://arxiv.org/pdf/2507.16120v1)

Authors: Shanshan Zhang, Qi Zhang, Siyue Wang, Tianshui Wen, Ziheng Zhou, Lingxiang Zheng, Yu Yang

In recent years, machine learning has achieved significant advancements in
inertial odometry. However, most existing inertial odometry methods primarily
rely on CNNs in the time domain. These methods often struggle to capture
long-term dependency in inertial measurement unit data, thereby constraining
the potential for further improvements in localization accuracy. To address
these issues, we propose a novel network architecture that integrates both
frequency-domain and time-domain information. Specifically, we leverage the
global view and energy compaction properties of frequency-domain learning to
effectively model long-term dependency and reduce redundancy in IMU data.
Additionally, we introduce a Scalar LSTM to capture sequential dependencies in
the time domain, enabling cross-domain information fusion and providing a
stable and reliable reference for localization. Experimental evaluations on
multiple public datasets (e.g., RIDI, RoNIN, OxIOD, RNIN, TLIO, and IMUNet)
demonstrate the effectiveness of the proposed frequency-time domain fusion
strategy. Notably, on the RoNIN dataset, our method achieves a 43.0% reduction
in absolute trajectory error and a 13.1% reduction in relative trajectory error
compared to RoNIN ResNet.

### 2. [DWSFormer: A Lightweight Inertial Odometry Network for Complex Motion Modeling](http://arxiv.org/pdf/2507.16121v1)

Authors: Shanshan Zhang, Qi Zhang, Siyue Wang, Tianshui Wen, Ziheng Zhou, Lingxiang Zheng, Yu Yang

Inertial odometry (IO) directly estimates the position of a carrier from
inertial sensor measurements and serves as a core technology for the widespread
deployment of consumer grade localization systems. While existing IO methods
can accurately reconstruct simple and near linear motion trajectories, they
often fail to account for drift errors caused by complex motion patterns such
as turning. This limitation significantly degrades localization accuracy and
restricts the applicability of IO systems in real world scenarios. To address
these challenges, we propose a lightweight IO framework. Specifically, inertial
data is projected into a high dimensional implicit nonlinear feature space
using the Star Operation method, enabling the extraction of complex motion
features that are typically overlooked. We further introduce a collaborative
attention mechanism that jointly models global motion dynamics across both
channel and temporal dimensions. In addition, we design Multi Scale Gated
Convolution Units to capture fine grained dynamic variations throughout the
motion process, thereby enhancing the model's ability to learn rich and
expressive motion representations. Extensive experiments demonstrate that our
proposed method consistently outperforms SOTA baselines across six widely used
inertial datasets. Compared to baseline models on the RoNIN dataset, it
achieves reductions in ATE ranging from 2.26% to 65.78%, thereby establishing a
new benchmark in the field.

### 3. [Scanning Bot: Efficient Scan Planning using Panoramic Cameras](http://arxiv.org/pdf/2507.16175v1)

Authors: Euijeong Lee, Kyung Min Han, Young J. Kim

Panoramic RGB-D cameras are known for their ability to produce high quality
3D scene reconstructions. However, operating these cameras involves manually
selecting viewpoints and physically transporting the camera, making the
generation of a 3D model time consuming and tedious. Additionally, the process
can be challenging for novice users due to spatial constraints, such as
ensuring sufficient feature overlap between viewpoint frames. To address these
challenges, we propose a fully autonomous scan planning that generates an
efficient tour plan for environment scanning, ensuring collision-free
navigation and adequate overlap between viewpoints within the plan. Extensive
experiments conducted in both synthetic and real-world environments validate
the performance of our planner against state-of-the-art view planners. In
particular, our method achieved an average scan coverage of 99 percent in the
real-world experiment, with our approach being up to 3 times faster than
state-of-the-art planners in total scan time.

### 4. [GFM-Planner: Perception-Aware Trajectory Planning with Geometric Feature Metric](http://arxiv.org/pdf/2507.16233v1)

Authors: Yue Lin, Xiaoxuan Zhang, Yang Liu, Dong Wang, Huchuan Lu

Like humans who rely on landmarks for orientation, autonomous robots depend
on feature-rich environments for accurate localization. In this paper, we
propose the GFM-Planner, a perception-aware trajectory planning framework based
on the geometric feature metric, which enhances LiDAR localization accuracy by
guiding the robot to avoid degraded areas. First, we derive the Geometric
Feature Metric (GFM) from the fundamental LiDAR localization problem. Next, we
design a 2D grid-based Metric Encoding Map (MEM) to efficiently store GFM
values across the environment. A constant-time decoding algorithm is further
proposed to retrieve GFM values for arbitrary poses from the MEM. Finally, we
develop a perception-aware trajectory planning algorithm that improves LiDAR
localization capabilities by guiding the robot in selecting trajectories
through feature-rich areas. Both simulation and real-world experiments
demonstrate that our approach enables the robot to actively select trajectories
that significantly enhance LiDAR localization accuracy.

### 5. [Trajectory Planning of a Curtain Wall Installation Robot Based on Biomimetic Mechanisms](http://arxiv.org/pdf/2507.16305v1)

Authors: Xiao Liu, Weijun Wang, Tianlun Huang, Zhiyong Wang, Wei Feng

As the robotics market rapidly evolves, energy consumption has become a
critical issue, particularly restricting the application of construction
robots. To tackle this challenge, our study innovatively draws inspiration from
the mechanics of human upper limb movements during weight lifting, proposing a
bio-inspired trajectory planning framework that incorporates human energy
conversion principles. By collecting motion trajectories and electromyography
(EMG) signals during dumbbell curls, we construct an anthropomorphic trajectory
planning that integrates human force exertion patterns and energy consumption
patterns. Utilizing the Particle Swarm Optimization (PSO) algorithm, we achieve
dynamic load distribution for robotic arm trajectory planning based on
human-like movement features. In practical application, these bio-inspired
movement characteristics are applied to curtain wall installation tasks,
validating the correctness and superiority of our trajectory planning method.
Simulation results demonstrate a 48.4% reduction in energy consumption through
intelligent conversion between kinetic and potential energy. This approach
provides new insights and theoretical support for optimizing energy use in
curtain wall installation robots during actual handling tasks.

### 6. [Design and Dimensional Optimization of Legged Structures for Construction Robots](http://arxiv.org/pdf/2507.16328v1)

Authors: Xiao Liu, Xianlong Yang, Weijun Wang, Wei Feng

Faced with complex and unstructured construction environments, wheeled and
tracked robots exhibit significant limitations in terrain adaptability and
flexibility, making it difficult to meet the requirements of autonomous
operation. Inspired by ants in nature, this paper proposes a leg configuration
design and optimization method tailored for construction scenarios, aiming to
enhance the autonomous mobility of construction robots. This paper analyzes the
full operational motion performance of the leg during both swing and stance
phases. First, based on kinematic modeling and multi-dimensional workspace
analysis, the concept of an "improved workspace" is introduced, and graphical
methods are used to optimize the leg dimensions during the swing phase.
Furthermore, a new concept of "average manipulability" is introduced based on
the velocity Jacobian matrix, and numerical solutions are applied to obtain the
leg segment ratio that maximizes manipulability. To overcome the difficulties
associated with traditional analytical methods, virtual prototype simulations
are conducted in ADAMS to explore the relationship between the robot body's
optimal flexibility and leg segment proportions. In summary, the leg segment
proportions with the best comprehensive motion performance are obtained. This
study presents the first multi-dimensional quantitative evaluation framework
for leg motion performance tailored for construction environments, providing a
structural design foundation for legged construction robots to achieve
autonomous mobility in complex terrains.

### 7. [Topology Optimization of Leg Structures for Construction Robots Based on Variable Density Method](http://arxiv.org/pdf/2507.16335v1)

Authors: Xiao Liu, Xianlong Yang, Weijun Wang, Wei Feng

In complex terrain construction environments, there are high demands for
robots to achieve both high payload capacity and mobility flexibility. As the
key load-bearing component, the optimization of robotic leg structures is of
particular importance. Therefore, this study focuses on the optimization of leg
structures for construction robots, proposing a topology optimization strategy
based on the SIMP (Solid Isotropic Microstructures with Penalization) variable
density method along with a structural re-design approach. The design
performance is comprehensively validated through finite element analysis using
ANSYS. First, static and modal analyses are conducted to evaluate the
rationality of the initial design. Then, topology optimization using the
SIMP-based variable density method is applied to the femur section, which
accounts for the largest proportion of the leg's weight. Based on iterative
calculations, the femur undergoes secondary structural reconstruction. After
optimization, the mass of the femur is reduced by 19.45\%, and the overall leg
mass decreases by 7.92\%, achieving the goal of lightweight design. Finally,
static and modal analyses are conducted on the reconstructed leg. The results
demonstrate that the optimized leg still meets structural performance
requirements, validating the feasibility of lightweight design. This research
provides robust theoretical and technical support for lightweight construction
robot design and lays a foundation for their efficient operation in complex
construction environments.

### 8. [Humanoid Robot Whole-body Geometric Calibration with Embedded Sensors and a Single Plane](http://arxiv.org/pdf/2507.16369v1)

Authors: Thanh D V Nguyen, Vincent Bonnet, Pierre Fernbach, David Daney, Florent Lamiraux

Whole-body geometric calibration of humanoid robots using classical robot
calibration methods is a timeconsuming and experimentally burdensome task.
However, despite its significance for accurate control and simulation, it is
often overlooked in the humanoid robotics community. To address this issue, we
propose a novel practical method that utilizes a single plane, embedded force
sensors, and an admittance controller to calibrate the whole-body kinematics of
humanoids without requiring manual intervention. Given the complexity of
humanoid robots, it is crucial to generate and determine a minimal set of
optimal calibration postures. To do so, we propose a new algorithm called IROC
(Information Ranking algorithm for selecting Optimal Calibration postures).
IROC requires a pool of feasible candidate postures to build a normalized
weighted information matrix for each posture. Then, contrary to other
algorithms from the literature, IROC will determine the minimal number of
optimal postures that are to be played onto a robot for its calibration. Both
IROC and the single-plane calibration method were experimentally validated on a
TALOS humanoid robot. The total whole-body kinematics chain was calibrated
using solely 31 optimal postures with 3-point contacts on a table by the robot
gripper. In a cross-validation experiment, the average root-mean-square (RMS)
error was reduced by a factor of 2.3 compared to the manufacturer's model.

### 9. [Morpheus: A Neural-driven Animatronic Face with Hybrid Actuation and Diverse Emotion Control](http://arxiv.org/pdf/2507.16645v1)

Authors: Zongzheng Zhang, Jiawen Yang, Ziqiao Peng, Meng Yang, Jianzhu Ma, Lin Cheng, Huazhe Xu, Hang Zhao, Hao Zhao

Previous animatronic faces struggle to express emotions effectively due to
hardware and software limitations. On the hardware side, earlier approaches
either use rigid-driven mechanisms, which provide precise control but are
difficult to design within constrained spaces, or tendon-driven mechanisms,
which are more space-efficient but challenging to control. In contrast, we
propose a hybrid actuation approach that combines the best of both worlds. The
eyes and mouth-key areas for emotional expression-are controlled using rigid
mechanisms for precise movement, while the nose and cheek, which convey subtle
facial microexpressions, are driven by strings. This design allows us to build
a compact yet versatile hardware platform capable of expressing a wide range of
emotions. On the algorithmic side, our method introduces a self-modeling
network that maps motor actions to facial landmarks, allowing us to
automatically establish the relationship between blendshape coefficients for
different facial expressions and the corresponding motor control signals
through gradient backpropagation. We then train a neural network to map speech
input to corresponding blendshape controls. With our method, we can generate
distinct emotional expressions such as happiness, fear, disgust, and anger,
from any given sentence, each with nuanced, emotion-specific control signals-a
feature that has not been demonstrated in earlier systems. We release the
hardware design and code at https://github.com/ZZongzheng0918/Morpheus-Hardware
and https://github.com/ZZongzheng0918/Morpheus-Software.

### 10. [Benchmarking LLM Privacy Recognition for Social Robot Decision Making](http://arxiv.org/pdf/2507.16124v1)

Authors: Dakota Sullivan, Shirley Zhang, Jennica Li, Heather Kirkorian, Bilge Mutlu, Kassem Fawaz

Social robots are embodied agents that interact with people while following
human communication norms. These robots interact using verbal and non-verbal
cues, and share the physical environments of people. While social robots have
previously utilized rule-based systems or probabilistic models for user
interaction, the rapid evolution of large language models (LLMs) presents new
opportunities to develop LLM-empowered social robots for enhanced human-robot
interaction. To fully realize these capabilities, however, robots need to
collect data such as audio, fine-grained images, video, and locations. As a
result, LLMs often process sensitive personal information, particularly within
home environments. Given the tension between utility and privacy risks,
evaluating how current LLMs manage sensitive data is critical. Specifically, we
aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the
context of household social robots. In this study, we present a set of
privacy-relevant scenarios crafted through the lens of Contextual Integrity
(CI). We first survey users' privacy preferences regarding in-home social robot
behaviors and then examine how their privacy orientation affects their choices
of these behaviors (N = 450). We then provide the same set of scenarios and
questions to state-of-the-art LLMs (N = 10) and find that the agreement between
humans and LLMs is low. To further investigate the capabilities of LLMs as a
potential privacy controller, we implement four additional prompting strategies
and compare their results. Finally, we discuss the implications and potential
of AI privacy awareness in human-robot interaction.

### Software Engineering

### 1. [Ten Essential Guidelines for Building High-Quality Research Software](http://arxiv.org/pdf/2507.16166v1)

Authors: Nasir U. Eisty, David E. Bernholdt, Alex Koufos, David J. Luet, Miranda Mundt

High-quality research software is a cornerstone of modern scientific
progress, enabling researchers to analyze complex data, simulate phenomena, and
share reproducible results. However, creating such software requires adherence
to best practices that ensure robustness, usability, and sustainability. This
paper presents ten guidelines for producing high-quality research software,
covering every stage of the development lifecycle. These guidelines emphasize
the importance of planning, writing clean and readable code, using version
control, and implementing thorough testing strategies. Additionally, they
address key principles such as modular design, reproducibility, performance
optimization, and long-term maintenance. The paper also highlights the role of
documentation and community engagement in enhancing software usability and
impact. By following these guidelines, researchers can create software that
advances their scientific objectives and contributes to a broader ecosystem of
reliable and reusable research tools. This work serves as a practical resource
for researchers and developers aiming to elevate the quality and impact of
their research software.

### 2. [Search-based Generation of Waypoints for Triggering Self-Adaptations in Maritime Autonomous Vessels](http://arxiv.org/pdf/2507.16327v1)

Authors: Karoline Nylænder, Aitor Arrieta, Shaukat Ali, Paolo Arcaini

Self-adaptation in maritime autonomous vessels (AVs) enables them to adapt
their behaviors to address unexpected situations while maintaining
dependability requirements. During the design of such AVs, it is crucial to
understand and identify the settings that should trigger adaptations, enabling
validation of their implementation. To this end, we focus on the navigation
software of AVs, which must adapt their behavior during operation through
adaptations. AVs often rely on predefined waypoints to guide them along
designated routes, ensuring safe navigation. We propose a multiobjective
search-based approach, called WPgen, to generate minor modifications to the
predefined set of waypoints, keeping them as close as possible to the original
waypoints, while causing the AV to navigate inappropriately when navigating
with the generated waypoints. WPgen uses NSGA-II as the multi-objective search
algorithm with three seeding strategies for its initial population, resulting
in three variations of WPgen. We evaluated these variations on three AVs (one
overwater tanker and two underwater). We compared the three variations of WPgen
with Random Search as the baseline and with each other. Experimental results
showed that the effectiveness of these variations varied depending on the AV.
Based on the results, we present the research and practical implications of
WPgen.

### 3. [Improving Code LLM Robustness to Prompt Perturbations via Layer-Aware Model Editing](http://arxiv.org/pdf/2507.16407v1)

Authors: Shuhan Liu, Xing Hu, Kerui Huang, Xiaohu Yang, David Lo, Xin Xia

Large language models (LLMs) have demonstrated impressive capabilities in
code generation, where the natural language prompt plays a crucial role in
conveying user intent to the model. However, prior studies have shown that LLMs
are highly sensitive to prompt perturbations. Minor modifications in wording,
syntax, or formatting can significantly reduce the functional correctness of
generated code. As perturbations frequently occur in real-world scenarios,
improving the robustness of LLMs to prompt perturbations is essential for
ensuring reliable performance in practical code generation. In this paper, we
introduce CREME (Code Robustness Enhancement via Model Editing), a novel
approach that enhances LLM robustness through targeted parameter updates. CREME
first identifies robustness-sensitive layers by comparing hidden states between
an original prompt and its perturbed variant. Then, it performs lightweight
parameter editing at the identified layer to reduce performance degradation. We
evaluate CREME on two widely used code generation benchmarks (HumanEval and
MBPP) along with their perturbed counterparts. Experimental results show that
CREME improves Pass@1 accuracy by 63% on perturbed prompts while maintaining
stable performance on clean inputs, with accuracy deviations within 1%. Further
analysis reveals that robustness-sensitive layers are primarily concentrated in
the middle and deeper layers of the network, and their locations vary across
different model architectures. These insights provide a valuable foundation for
developing future robustness-oriented editing strategies.

### 4. [Exploring Large Language Models for Analyzing and Improving Method Names in Scientific Code](http://arxiv.org/pdf/2507.16439v1)

Authors: Gunnar Larsen, Carol Wong, Anthony Peruma

Research scientists increasingly rely on implementing software to support
their research. While previous research has examined the impact of identifier
names on program comprehension in traditional programming environments, limited
work has explored this area in scientific software, especially regarding the
quality of method names in the code. The recent advances in Large Language
Models (LLMs) present new opportunities for automating code analysis tasks,
such as identifier name appraisals and recommendations. Our study evaluates
four popular LLMs on their ability to analyze grammatical patterns and suggest
improvements for 496 method names extracted from Python-based Jupyter
Notebooks. Our findings show that the LLMs are somewhat effective in analyzing
these method names and generally follow good naming practices, like starting
method names with verbs. However, their inconsistent handling of
domain-specific terminology and only moderate agreement with human annotations
indicate that automated suggestions require human evaluation. This work
provides foundational insights for improving the quality of scientific code
through AI automation.

### 5. [On the Effectiveness of LLM-as-a-judge for Code Generation and Summarization](http://arxiv.org/pdf/2507.16587v1)

Authors: Giuseppe Crupi, Rosalia Tufano, Alejandro Velasco, Antonio Mastropaolo, Denys Poshyvanyk, Gabriele Bavota

Large Language Models have been recently exploited as judges for complex
natural language processing tasks, such as Q&A. The basic idea is to delegate
to an LLM the assessment of the "quality" of the output provided by an
automated technique for tasks for which: (i) quantitative metrics would only
tell part of the story, and; (ii) a large-scale human-based evaluation would be
too expensive. LLMs-as-a-judge, if proven effective for a specific task, can
also unlock new possibilities for automation, with several LLMs proposing a
solution for a given instance of the task and others judging and deciding what
is the best output to show the user. We study the effectiveness of
LLMs-as-a-judge for two code-related tasks, namely code generation and code
summarization. The rationale for choosing these tasks is two-fold. First,
quantitative metrics are usually not enough for the assessment of code
summarizers/generators. For example, it is well documented that metrics such as
BLEU are quite weak proxies for the quality of the generated summaries. Second,
even state-of-the-art techniques still struggle with handling complex instances
of these tasks, making them good candidates for benefiting from more advanced
solutions envisioning collaboration among LLMs. For code generation, we check
whether eight LLMs are able to judge the correctness of 1,405 Java methods and
1,281 Python functions generated by the same LLMs or implemented by humans. For
code summarization, we compare the judgment of five LLMs to those provided by
nine humans for ~1.2k summaries, related to both Java and Python functions. Our
findings show that GPT-4-turbo is the best LLM in terms of judging capabilities
for both tasks, with "smaller" LLMs featuring tens of billions parameters not
being able to cope with judging tasks. However, even the best-performing LLM
frequently misjudges the correctness of the code and summary quality.

### 6. [VulCoCo: A Simple Yet Effective Method for Detecting Vulnerable Code Clones](http://arxiv.org/pdf/2507.16661v1)

Authors: Tan Bui, Yan Naing Tun, Thanh Phuc Nguyen, Yindu Su, Ferdian Thung, Yikun Li, Han Wei Ang, Yide Yin, Frank Liauw, Lwin Khin Shar, Eng Lieh Ouh, Ting Zhang, David Lo

Code reuse is common in modern software development, but it can also spread
vulnerabilities when developers unknowingly copy risky code. The code fragments
that preserve the logic of known vulnerabilities are known as vulnerable code
clones (VCCs). Detecting those VCCs is a critical but challenging task.
Existing VCC detection tools often rely on syntactic similarity or produce
coarse vulnerability predictions without clear explanations, limiting their
practical utility. In this paper, we propose VulCoCo, a lightweight and
scalable approach that combines embedding-based retrieval with large language
model (LLM) validation. Starting from a set of known vulnerable functions, we
retrieve syntactically or semantically similar candidate functions from a large
corpus and use an LLM to assess whether the candidates retain the
vulnerability. Given that there is a lack of reproducible vulnerable code clone
benchmarks, we first construct a synthetic benchmark that spans various clone
types.
  Our experiments on the benchmark show that VulCoCo outperforms prior
state-of-the-art methods in terms of Precision@k and mean average precision
(MAP). In addition, we also demonstrate VulCoCo's effectiveness in real-world
projects by submitting 400 pull requests (PRs) to 284 open-source projects.
Among them, 75 PRs were merged, and 15 resulted in newly published CVEs. We
also provide insights to inspire future work to further improve the precision
of vulnerable code clone detection.

### 7. [VulGuard: An Unified Tool for Evaluating Just-In-Time Vulnerability Prediction Models](http://arxiv.org/pdf/2507.16685v1)

Authors: Duong Nguyen, Manh Tran-Duc, Thanh Le-Cong, Triet Huynh Minh Le, M. Ali Babar, Quyet-Thang Huynh

We present VulGuard, an automated tool designed to streamline the extraction,
processing, and analysis of commits from GitHub repositories for Just-In-Time
vulnerability prediction (JIT-VP) research. VulGuard automatically mines commit
histories, extracts fine-grained code changes, commit messages, and software
engineering metrics, and formats them for downstream analysis. In addition, it
integrates several state-of-the-art vulnerability prediction models, allowing
researchers to train, evaluate, and compare models with minimal setup. By
supporting both repository-scale mining and model-level experimentation within
a unified framework, VulGuard addresses key challenges in reproducibility and
scalability in software security research. VulGuard can also be easily
integrated into the CI/CD pipeline. We demonstrate the effectiveness of the
tool in two influential open-source projects, FFmpeg and the Linux kernel,
highlighting its potential to accelerate real-world JIT-VP research and promote
standardized benchmarking. A demo video is available at:
https://youtu.be/j96096-pxbs

### 8. [LOCOFY Large Design Models -- Design to code conversion solution](http://arxiv.org/pdf/2507.16208v1)

Authors: Sohaib Muhammad, Ashwati Vipin, Karan Shetti, Honey Mittal

Despite rapid advances in Large Language Models and Multimodal Large Language
Models (LLMs), numerous challenges related to interpretability, scalability,
resource requirements and repeatability remain, related to their application in
the design-to-code space. To address this, we introduce the Large Design Models
(LDMs) paradigm specifically trained on designs and webpages to enable seamless
conversion from design-to-code. We have developed a training and inference
pipeline by incorporating data engineering and appropriate model architecture
modification. The training pipeline consists of the following: 1)Design
Optimiser: developed using a proprietary ground truth dataset and addresses
sub-optimal designs; 2)Tagging and feature detection: using pre-trained and
fine-tuned models, this enables the accurate detection and classification of UI
elements; and 3)Auto Components: extracts repeated UI structures into reusable
components to enable creation of modular code, thus reducing redundancy while
enhancing code reusability. In this manner, each model addresses distinct but
key issues for design-to-code conversion. Separately, our inference pipeline
processes real-world designs to produce precise and interpretable instructions
for code generation and ensures reliability. Additionally, our models
illustrated exceptional end-to-end design-to-code conversion accuracy using a
novel preview match score metric. Comparative experiments indicated superior
performance of LDMs against LLMs on accuracy of node positioning,
responsiveness and reproducibility. Moreover, our custom-trained tagging and
feature detection model demonstrated high precision and consistency in
identifying UI elements across a wide sample of test designs. Thus, our
proposed LDMs are a reliable and superior solution to understanding designs
that subsequently enable the generation of efficient and reliable
production-ready code.

### 9. [LLM-Driven Collaborative Model for Untangling Commits via Explicit and Implicit Dependency Reasoning](http://arxiv.org/pdf/2507.16395v1)

Authors: Bo Hou, Xin Tan, Kai Zheng, Fang Liu, Yinghao Zhu, Li Zhang

Atomic commits, each of which addresses a single development concern, are a
best practice in software development. However, developers frequently produce
tangled commits that mix unrelated changes due to practical constraints or
unclear boundaries, negatively impacting code review and maintenance. Although
prior commit untangling approaches: rule-based, feature-based, or graph-based,
have made progress, they often rely on shallow signals and fail to distinguish
between explicit dependencies (e.g., control/data flow) and implicit ones
(e.g., semantic or conceptual relationships). In this paper, we propose
ColaUntangle, a new collaborative consultation framework for commit untangling
that models both explicit and implicit dependencies among code changes.
ColaUntangle integrates Large Language Model (LLM)-driven agents in a
multi-agent architecture: one agent specializes in explicit dependencies,
another in implicit ones, and a reviewer agent synthesizes their perspectives
through iterative consultation. To capture explicit and implicit contextual
information, we construct multi-version Program Dependency Graphs (delta-PDG),
enabling agents to reason over code relationships with both symbolic and
semantic depth. We evaluate ColaUntangle on two widely-used datasets (1,612 C#
and 14k Java tangled commits). Experimental results show that ColaUntangle
outperforms the best-performing baseline, achieving an improvement of 44% on
the C# dataset and 100% on the Java dataset. These findings highlight the
potential of LLM-based collaborative frameworks for advancing automated commit
untangling tasks.

### 10. [ACT: Bridging the Gap in Code Translation through Synthetic Data Generation & Adaptive Training](http://arxiv.org/pdf/2507.16478v1)

Authors: Shreya Saxena, Siva Prasad, Zishan Ahmad, Vishal Vaddina

Code translation is a crucial process in software development and migration
projects, enabling interoperability between different programming languages and
enhancing software adaptability and thus longevity. Traditional automated
translation methods rely heavily on handcrafted transformation rules, which
often lack flexibility and scalability. Meanwhile, advanced language models
present promising alternatives but are often limited by proprietary, API-based
implementations that raise concerns over data security and reliance. In this
paper, we present Auto-Train for Code Translation (ACT), an innovative
framework that aims to improve code translation capabilities by enabling
in-house finetuning of open-source Large Language Models (LLMs). ACT's
automated pipeline significantly boosts the performance of these models,
narrowing the gap between open-source accessibility and the high performance of
closed-source solutions. Central to ACT is its synthetic data generation
module, which builds extensive, high-quality datasets from initial code
samples, incorporating unit tests to ensure functional accuracy and diversity.
ACT's evaluation framework incorporates execution-level checks, offering a
comprehensive assessment of translation quality. A key feature in ACT is its
controller module, which manages the entire pipeline by dynamically adjusting
hyperparameters, orchestrating iterative data generation, and finetuning based
on real-time evaluations. This enables ACT to intelligently optimize when to
continue training, generate additional targeted training data, or stop the
process. Our results demonstrate that ACT consistently enhances the
effectiveness of open-source models, offering businesses and developers a
secure and reliable alternative. Additionally, applying our data generation
pipeline to industry-scale migration projects has led to a notable increase in
developer acceleration.

### Social and Information Networks

### 1. [WhatsApp Tiplines and Multilingual Claims in the 2021 Indian Assembly Elections](http://arxiv.org/pdf/2507.16298v1)

Authors: Gautam Kishore Shahi, Scot A. Hale

WhatsApp tiplines, first launched in 2019 to combat misinformation, enable
users to interact with fact-checkers to verify misleading content. This study
analyzes 580 unique claims (tips) from 451 users, covering both high-resource
languages (English, Hindi) and a low-resource language (Telugu) during the 2021
Indian assembly elections using a mixed-method approach. We categorize the
claims into three categories, election, COVID-19, and others, and observe
variations across languages. We compare content similarity through frequent
word analysis and clustering of neural sentence embeddings. We also investigate
user overlap across languages and fact-checking organizations. We measure the
average time required to debunk claims and inform tipline users. Results reveal
similarities in claims across languages, with some users submitting tips in
multiple languages to the same fact-checkers. Fact-checkers generally require a
couple of days to debunk a new claim and share the results with users. Notably,
no user submits claims to multiple fact-checking organizations, indicating that
each organization maintains a unique audience. We provide practical
recommendations for using tiplines during elections with ethical consideration
of users' information.

### 2. [A Comprehensive Data-centric Overview of Federated Graph Learning](http://arxiv.org/pdf/2507.16541v1)

Authors: Zhengyu Wu, Xunkai Li, Yinlin Zhu, Zekai Chen, Guochen Yan, Yanyu Yan, Hao Zhang, Yuming Ai, Xinmo Jin, Rong-Hua Li, Guoren Wang

In the era of big data applications, Federated Graph Learning (FGL) has
emerged as a prominent solution that reconcile the tradeoff between optimizing
the collective intelligence between decentralized datasets holders and
preserving sensitive information to maximum. Existing FGL surveys have
contributed meaningfully but largely focus on integrating Federated Learning
(FL) and Graph Machine Learning (GML), resulting in early stage taxonomies that
emphasis on methodology and simulated scenarios. Notably, a data centric
perspective, which systematically examines FGL methods through the lens of data
properties and usage, remains unadapted to reorganize FGL research, yet it is
critical to assess how FGL studies manage to tackle data centric constraints to
enhance model performances. This survey propose a two-level data centric
taxonomy: Data Characteristics, which categorizes studies based on the
structural and distributional properties of datasets used in FGL, and Data
Utilization, which analyzes the training procedures and techniques employed to
overcome key data centric challenges. Each taxonomy level is defined by three
orthogonal criteria, each representing a distinct data centric configuration.
Beyond taxonomy, this survey examines FGL integration with Pretrained Large
Models, showcases realistic applications, and highlights future direction
aligned with emerging trends in GML.

### 3. [SASH: Decoding Community Structure in Graphs](http://arxiv.org/pdf/2507.16583v1)

Authors: Allison Beemer, Jessalyn Bolkema

Detection of communities in a graph entails identifying clusters of densely
connected vertices; the area has a variety of important applications and a rich
literature. The problem has previously been situated in the realm of error
correcting codes by viewing a graph as a noisy version of the assumed
underlying communities. In this paper, we introduce an encoding of community
structure along with the resulting code's parameters. We then present a novel
algorithm, SASH, to decode to estimated communities given an observed dataset.
We demonstrate the performance of SASH via simulations on an assortative
planted partition model and on the Zachary's Karate Club dataset.

### Systems and Control

### 1. [Design and Implementation of a Lightweight Object Detection System for Resource-Constrained Edge Environments](http://arxiv.org/pdf/2507.16155v1)

Authors: Jiyue Jiang, Mingtong Chen, Zhengbao Yang

This project aims to develop a system to run the object detection model under
low power consumption conditions. The detection scene is set as an outdoor
traveling scene, and the detection categories include people and vehicles. In
this system, users data does not need to be uploaded to the cloud, which is
suitable for use in environments with portable needs and strict requirements
for data privacy. The MCU device used in this system is STM32H7, which has
better performance among low power devices. The YOLOv5 system is selected to
train the object detection model. To overcome the resource limitation of the
embedded devices, this project uses several model compression techniques such
as pruned, quantization, and distillation, which could improve the performance
and efficiency of the detection model. Through these processes, the model s
computation and the quantity of model parameters could be reduced, in order to
run computer vision models on micro-controller devices for the development of
embedded vision applications.

### 2. [Design and Optimization of Wearables for Human Motion Energy Harvesting](http://arxiv.org/pdf/2507.16157v1)

Authors: Weijia Peng, Mingtong Chen, Zhengbao Yang

As wearable electronics become increasingly prevalent, there is a rise in
interest and demand for sustainably designed systems that are also energy
self-sufficient. The research described in this paper investigated a shoe-worn
energy harvesting system designed use the mechanical energy from walking to
output electrical energy. A spring is attached to electromagnetic generator
embedded in the heel of the shoe to recover the vertical pressure caused by the
foot strike. The simulated prototype consisted of a standard EM generator
designed in MATLAB demonstrating a maximum voltage of 12V. The initial low
fidelity prototype demonstrated testing the relationship between the EM
generator and a simple electrical circuit, with energy output observed. Future
research will explore enhancing the overall generator design, integrate a power
management IC for battery protect and regulation, and combine the system into a
final product, wearable footwear. This research lays a foundation for
self-powered footwear and energy independent wearable electronic devices.

### 3. [The Bode Plots for Sliding-Mode Control Design](http://arxiv.org/pdf/2507.16281v1)

Authors: Ulises Pérez-Ventura

This paper develops a unified frequency-domain framework for the analysis of
sliding-mode control systems, encompassing both discontinuous and
Lipschitz-continuous implementations. Using describing function (DF) theory,
closed-form expressions are derived for the amplitude and frequency of
chattering oscillations, as well as equivalent gain (EG) models that enable
closed-loop sensitivity analysis. The proposed methodology captures the
influence of actuator dynamics, control parameters, and disturbance profiles on
steady-state performance.
  Theoretical predictions for bias and oscillatory components are validated
through simulations under both constant and sinusoidal perturbations. In the
low-frequency regime, the EG-based sensitivity functions accurately predict the
amplitude and phase of the system response, with tracking errors remaining
within a 15\% margin, provided that the DF assumptions hold. The framework also
incorporates orbital stability considerations via Loebs criterion, ensuring
that chattering remains bounded.
  Overall, the results offer practical insight into the robust design of
sliding-mode controllers, enabling systematic gain tuning that balances
disturbance rejection and chattering attenuation, while accounting for actuator
and sensor constraints.

### 4. [Polarforming Design for Movable Antenna Systems](http://arxiv.org/pdf/2507.16311v1)

Authors: Zijian Zhou, Jingze Ding, Rui Zhang

Polarforming has emerged as a promising technique to enable the antenna to
shape its polarization into a desired state for aligning with that of the
received electromagnetic (EM) wave or reconfiguring that of the transmitted EM
wave. In this letter, we investigate polarforming design for the movable
antenna (MA)-enabled communication system. Specifically, we consider a
single-input single-output (SISO) system with reconfigurable antenna positions
and polarizations to leverage both spatial and polarization degrees of freedom
(DoFs). First, we present a polarized channel model and characterize the
channel response as a function of antenna positions and polarforming phase
shifts. To maximize the achievable rate of the proposed system, we then develop
a successive convex approximation (SCA)-based optimization algorithm by
iteratively optimizing the antenna positions and phase shifts at both the
transmitter and receiver. Furthermore, simulation results demonstrate the
performance gains of the proposed system over conventional systems in
mitigating channel depolarization and adapting to channel fading.

### 5. [Derivative-Agnostic Inference of Nonlinear Hybrid Systems](http://arxiv.org/pdf/2507.16426v1)

Authors: Hengzhi Yu, Bohan Ma, Mingshuai Chen, Jie An, Bin Gu, Naijun Zhan, Jianwei Yin

This paper addresses the problem of inferring a hybrid automaton from a set
of input-output traces of a hybrid system exhibiting discrete mode switching
between continuously evolving dynamics. Existing approaches mainly adopt a
derivative-based method where (i) the occurrence of mode switching is
determined by a drastic variation in derivatives and (ii) the clustering of
trace segments relies on signal similarity -- both subject to user-supplied
thresholds. We present a derivative-agnostic approach, named Dainarx, to infer
nonlinear hybrid systems where the dynamics are captured by nonlinear
autoregressive exogenous (NARX) models. Dainarx employs NARX models as a
unified, threshold-free representation through the detection of mode switching
and trace-segment clustering. We show that Dainarx suffices to learn models
that closely approximate a general class of hybrid systems featuring high-order
nonlinear dynamics with exogenous inputs, nonlinear guard conditions, and
linear resets. Experimental results on a collection of benchmarks indicate that
our approach can effectively and efficiently infer nontrivial hybrid automata
with high-order dynamics yielding significantly more accurate approximations
than state-of-the-art techniques.

### 6. [Arbitrage Tactics in the Local Markets via Hierarchical Multi-agent Reinforcement Learning](http://arxiv.org/pdf/2507.16479v1)

Authors: Haoyang Zhang, Mina Montazeri, Philipp Heer, Koen Kok, Nikolaos G. Paterakis

Strategic bidding tactics employed by prosumers in local markets, including
the Local Electricity Market (LEM) and Local Flexibility Market (LFM), have
attracted significant attention due to their potential to enhance economic
benefits for market participants through optimized energy management and
bidding. While existing research has explored strategic bidding in a single
market with multi-agent reinforcement learning (MARL) algorithms, arbitrage
opportunities across local markets remain unexplored. This paper introduces a
hierarchical MARL (HMARL) algorithm designed to enable aggregator arbitrage
across multiple local markets. The strategic behavior of these aggregators in
local markets is modeled as a two-stage Markov game: the first stage involves
the LEM, while the second stage encompasses both the LFM and the balancing
market. To solve this two-stage Markov game, the HMARL framework assigns two
sub-agents to each aggregator, a primary sub-agent and a secondary sub-agent.
Without the arbitrage strategy, these sub-agents operate in silos, with the
primary sub-agent focusing on first-stage profits and the secondary sub-agent
on second-stage profits, each employing independent MARLs. On the contrary,
when implementing the arbitrage strategy with the proposed HMARL, the
sub-agents communicate and coordinate to perform arbitrage across multiple
local markets, enhancing overall efficiency. The case study, conducted under a
scenario where all aggregators employ the arbitrage strategy, shows that
despite higher initial costs in the LEM, this strategy generates substantial
savings in the LFM and the balancing market, resulting in a total profit
increase of $40.6\%$ on average. This highlights the capability of the proposed
HMARL to address the two-stage Markov game and facilitate arbitrage across
local markets, thereby enhancing profitability for participants.

### 7. [A Distributed Actor-Critic Algorithm for Fixed-Time Consensus in Nonlinear Multi-Agent Systems](http://arxiv.org/pdf/2507.16520v1)

Authors: Aria Delshad, Maryam Babazadeh

This paper proposes a reinforcement learning (RL)-based backstepping control
strategy to achieve fixed time consensus in nonlinear multi-agent systems with
strict feedback dynamics. Agents exchange only output information with their
neighbors over a directed communication graph, without requiring full state
measurements or symmetric communication. Achieving fixed time consensus, where
convergence occurs within a pre-specified time bound that is independent of
initial conditions is faced with significant challenges due to the presence of
unknown nonlinearities, inter-agent couplings, and external disturbances. This
work addresses these challenges by integrating actor critic reinforcement
learning with a novel fixed time adaptation mechanism. Each agent employs an
actor critic architecture supported by two estimator networks designed to
handle system uncertainties and unknown perturbations. The adaptation laws are
developed to ensure that all agents track the leader within a fixed time
regardless of their initial conditions. The consensus and tracking errors are
guaranteed to converge to a small neighborhood of the origin, with the
convergence radius adjustable through control parameters. Simulation results
demonstrate the effectiveness of the proposed approach and highlight its
advantages over state-of-the-art methods in terms of convergence speed and
robustness.

### 8. [Dynamic Activation and Assignment of SDN Controllers in LEO Satellite Constellations](http://arxiv.org/pdf/2507.16774v1)

Authors: Wafa Hasanain, Pablo G. Madoery, Halim Yanikomeroglu, Gunes Karabulut Kurt, Sameera Siddiqui, Stephane Martel, Khaled Ahmed, Colin Bellinger

Software-defined networking (SDN) has emerged as a promising approach for
managing traditional satellite communication. This enhances opportunities for
future services, including integrating satellite and terrestrial networks. In
this paper, we have developed an SDN-enabled framework for Low Earth Orbit
(LEO) satellite networks, incorporating the OpenFlow protocol, all within an
OMNeT++ simulation environment. Dynamic controller assignment is one of the
most significant challenges for large LEO constellations. Due to the movement
of LEO satellites, satellite-controller assignments must be updated frequently
to maintain low propagation delays. To address this issue, we present a dynamic
satellite-to-controller assignment (DSCA) optimization problem that
continuously adjusts these assignments. Our optimal DSCA (Opt-DSCA) approach
minimizes propagation delay and optimizes the number of active controllers. Our
preliminary results demonstrate that the DSCA approach significantly
outperforms the static satellite-to-controller assignment (SSCA) approach.
While SSCA may perform better with more controllers, this scheme fails to adapt
to satellite movements. Our DSCA approach consistently improves network
efficiency by dynamically reassigning satellites based on propagation delays.
Further, we found diminishing returns when the number of controllers is
increased beyond a certain point, suggesting optimal performance with a limited
number of controllers. Opt-DSCA lowers propagation delays and improves network
performance by optimizing satellite assignments and reducing active
controllers.

### 9. [Nd3+ Doping-induced Leakage Currents Suppression in High-temperature 0.7BiFeO3-0.3BaTiO3 Lead-free Piezoceramics](http://arxiv.org/pdf/2507.16156v1)

Authors: Jinming Liu, Mingtong Chen, Zhengbao Yang

BiFeO3 has attracted much attention as a potential candidate for replacing
conventional, lead based piezoelectric materials due to its remarkable
spontaneous polarization and high Curie temperature. However, its inherent high
leakage currents, which lead to low piezoelectric response and poor temperature
stability, have severely limited its practical applications. In this study,
lead free piezoelectric ceramics of the 0.7BiFeO3-0.3BaTiO3 (BF-BT) system were
prepared, and their microstructures along with high-temperature electrical
performance were modulated by introducing Nd3+. The results indicate that
moderate Nd doping improves lattice symmetry and reduces oxygen vacancy-related
defect dipoles, thereby effectively suppressing leakage currents at
temperatures above 75{\deg}C. The Nddoped samples exhibit significantly lower
leakage current densities, reduced by over 99% compared to the undoped
ceramics, reaching values as low as 10-5Acm-2. They also show higher
resistivity under elevated temperatures and electric fields, offering notable
improvements in thermal stability over the undoped counterparts. In addition,
the Nd-doped samples achieved piezoelectric coefficients as high as 172 pC N -1
at room temperature while still maintaining high dielectric and piezoelectric
responses at elevated temperatures. This work not only provides an effective
way to solve the leakage current problem of BF-BT ceramics in high temperature
applications but also indicates a new design strategy for optimizing the high
temperature stability of lead free piezoelectric materials, which shows a broad
application prospect in the field of high-temperature sensors and actuators.

### 10. [Unbeatable imitation of a friend](http://arxiv.org/pdf/2507.16221v1)

Authors: Masahiko Ueda

Imitation sometimes achieves success in multi-agent situations even though it
is very simple. In game theory, success of imitation has been characterized by
unbeatability against other agents. Previous studies specified conditions under
which imitation is unbeatable in repeated games, and clarified that the
existence of unbeatable imitation is strongly related to the existence of
payoff-controlling strategies, called zero-determinant strategies. However, the
previous studies mainly focused on ``imitation of opponents''. It was pointed
out that imitation of other players in the same group and imitation of other
players in the same role in other groups generally result in different
outcomes. Here, we investigate the existence condition of unbeatable imitation
in the latter ``imitation of friends'' situations. We find that it is stronger
than the existence condition of unbeatable zero-determinant strategies, whereas
both are very limited. Our findings suggest a strong relation between them even
in the `imitation of friends'' situations.

### Machine Learning (Statistics Category)

### 1. [PAC Off-Policy Prediction of Contextual Bandits](http://arxiv.org/pdf/2507.16236v1)

Authors: Yilong Wan, Yuqiang Li, Xianyi Wu

This paper investigates off-policy evaluation in contextual bandits, aiming
to quantify the performance of a target policy using data collected under a
different and potentially unknown behavior policy. Recently, methods based on
conformal prediction have been developed to construct reliable prediction
intervals that guarantee marginal coverage in finite samples, making them
particularly suited for safety-critical applications. To further achieve
coverage conditional on a given offline data set, we propose a novel algorithm
that constructs probably approximately correct prediction intervals. Our method
builds upon a PAC-valid conformal prediction framework, and we strengthen its
theoretical guarantees by establishing PAC-type bounds on coverage. We analyze
both finite-sample and asymptotic properties of the proposed method, and
compare its empirical performance with existing methods in simulations.

### 2. [Families of Optimal Transport Kernels for Cell Complexes](http://arxiv.org/pdf/2507.16569v1)

Authors: Rahul Khorana

Recent advances have discussed cell complexes as ideal learning
representations. However, there is a lack of available machine learning methods
suitable for learning on CW complexes. In this paper, we derive an explicit
expression for the Wasserstein distance between cell complex signal
distributions in terms of a Hodge-Laplacian matrix. This leads to a
structurally meaningful measure to compare CW complexes and define the optimal
transportation map. In order to simultaneously include both feature and
structure information, we extend the Fused Gromov-Wasserstein distance to CW
complexes. Finally, we introduce novel kernels over the space of probability
measures on CW complexes based on the dual formulation of optimal transport.

### 3. [Bootstrapped Control Limits for Score-Based Concept Drift Control Charts](http://arxiv.org/pdf/2507.16749v1)

Authors: Jiezhong Wu, Daniel W. Apley

Monitoring for changes in a predictive relationship represented by a fitted
supervised learning model (aka concept drift detection) is a widespread
problem, e.g., for retrospective analysis to determine whether the predictive
relationship was stable over the training data, for prospective analysis to
determine when it is time to update the predictive model, for quality control
of processes whose behavior can be characterized by a predictive relationship,
etc. A general and powerful Fisher score-based concept drift approach has
recently been proposed, in which concept drift detection reduces to detecting
changes in the mean of the model's score vector using a multivariate
exponentially weighted moving average (MEWMA). To implement the approach, the
initial data must be split into two subsets. The first subset serves as the
training sample to which the model is fit, and the second subset serves as an
out-of-sample test set from which the MEWMA control limit (CL) is determined.
In this paper, we develop a novel bootstrap procedure for computing the CL. Our
bootstrap CL provides much more accurate control of false-alarm rate,
especially when the sample size and/or false-alarm rate is small. It also
allows the entire initial sample to be used for training, resulting in a more
accurate fitted supervised learning model. We show that a standard nested
bootstrap (inner loop accounting for future data variability and outer loop
accounting for training sample variability) substantially underestimates
variability and develop a 632-like correction that appropriately accounts for
this. We demonstrate the advantages with numerical examples.

### 4. [A Distributional View of High Dimensional Optimization](http://arxiv.org/pdf/2507.16315v1)

Authors: Felix Benning

This PhD thesis presents a distributional view of optimization in place of a
worst-case perspective. We motivate this view with an investigation of the
failure point of classical optimization. Subsequently we consider the
optimization of a randomly drawn objective function. This is the setting of
Bayesian Optimization. After a review of Bayesian optimization we outline how
such a distributional view may explain predictable progress of optimization in
high dimension. It further turns out that this distributional view provides
insights into optimal step size control of gradient descent. To enable these
results, we develop mathematical tools to deal with random input to random
functions and a characterization of non-stationary isotropic covariance
kernels. Finally, we outline how assumptions about the data, specifically
exchangability, can lead to random objective functions in machine learning and
analyze their landscape.

### 5. [Meta-learning of Gibbs states for many-body Hamiltonians with applications to Quantum Boltzmann Machines](http://arxiv.org/pdf/2507.16373v1)

Authors: Ruchira V Bhat, Rahul Bhowmick, Avinash Singh, Krishna Kumar Sabapathy

The preparation of quantum Gibbs states is a fundamental challenge in quantum
computing, essential for applications ranging from modeling open quantum
systems to quantum machine learning. Building on the Meta-Variational Quantum
Eigensolver framework proposed by Cervera-Lierta et al.(2021) and a problem
driven ansatz design, we introduce two meta-learning algorithms:
Meta-Variational Quantum Thermalizer (Meta-VQT) and Neural Network Meta-VQT
(NN-Meta VQT) for efficient thermal state preparation of parametrized
Hamiltonians on Noisy Intermediate-Scale Quantum (NISQ) devices. Meta-VQT
utilizes a fully quantum ansatz, while NN Meta-VQT integrates a quantum
classical hybrid architecture. Both leverage collective optimization over
training sets to generalize Gibbs state preparation to unseen parameters. We
validate our methods on upto 8-qubit Transverse Field Ising Model and the
2-qubit Heisenberg model with all field terms, demonstrating efficient thermal
state generation beyond training data. For larger systems, we show that our
meta-learned parameters when combined with appropriately designed ansatz serve
as warm start initializations, significantly outperforming random
initializations in the optimization tasks. Furthermore, a 3- qubit Kitaev ring
example showcases our algorithm's effectiveness across finite-temperature
crossover regimes. Finally, we apply our algorithms to train a Quantum
Boltzmann Machine (QBM) on a 2-qubit Heisenberg model with all field terms,
achieving enhanced training efficiency, improved Gibbs state accuracy, and a
30-fold runtime speedup over existing techniques such as variational quantum
imaginary time (VarQITE)-based QBM highlighting the scalability and
practicality of meta-algorithm-based QBMs.

### 6. [Estimating Treatment Effects with Independent Component Analysis](http://arxiv.org/pdf/2507.16467v1)

Authors: Patrik Reizinger, Lester Mackey, Wieland Brendel, Rahul Krishnan

The field of causal inference has developed a variety of methods to
accurately estimate treatment effects in the presence of nuisance. Meanwhile,
the field of identifiability theory has developed methods like Independent
Component Analysis (ICA) to identify latent sources and mixing weights from
data. While these two research communities have developed largely
independently, they aim to achieve similar goals: the accurate and
sample-efficient estimation of model parameters. In the partially linear
regression (PLR) setting, Mackey et al. (2018) recently found that estimation
consistency can be improved with non-Gaussian treatment noise. Non-Gaussianity
is also a crucial assumption for identifying latent factors in ICA. We provide
the first theoretical and empirical insights into this connection, showing that
ICA can be used for causal effect estimation in the PLR model. Surprisingly, we
find that linear ICA can accurately estimate multiple treatment effects even in
the presence of Gaussian confounders or nonlinear nuisance.

### 7. [Deep Unfolding Network for Nonlinear Multi-Frequency Electrical Impedance Tomography](http://arxiv.org/pdf/2507.16678v1)

Authors: Giovanni S. Alberti, Damiana Lazzaro, Serena Morigi, Luca Ratti, Matteo Santacesaria

Multi-frequency Electrical Impedance Tomography (mfEIT) represents a
promising biomedical imaging modality that enables the estimation of tissue
conductivities across a range of frequencies. Addressing this challenge, we
present a novel variational network, a model-based learning paradigm that
strategically merges the advantages and interpretability of classical iterative
reconstruction with the power of deep learning. This approach integrates graph
neural networks (GNNs) within the iterative Proximal Regularized Gauss Newton
(PRGN) framework. By unrolling the PRGN algorithm, where each iteration
corresponds to a network layer, we leverage the physical insights of nonlinear
model fitting alongside the GNN's capacity to capture inter-frequency
correlations. Notably, the GNN architecture preserves the irregular triangular
mesh structure used in the solution of the nonlinear forward model, enabling
accurate reconstruction of overlapping tissue fraction concentrations.

### 8. [A Partitioned Sparse Variational Gaussian Process for Fast, Distributed Spatial Modeling](http://arxiv.org/pdf/2507.16771v1)

Authors: Michael Grosskopf, Kellin Rumsey, Ayan Biswas, Earl Lawrence

The next generation of Department of Energy supercomputers will be capable of
exascale computation. For these machines, far more computation will be possible
than that which can be saved to disk. As a result, users will be unable to rely
on post-hoc access to data for uncertainty quantification and other statistical
analyses and there will be an urgent need for sophisticated machine learning
algorithms which can be trained in situ. Algorithms deployed in this setting
must be highly scalable, memory efficient and capable of handling data which is
distributed across nodes as spatially contiguous partitions. One suitable
approach involves fitting a sparse variational Gaussian process (SVGP) model
independently and in parallel to each spatial partition. The resulting model is
scalable, efficient and generally accurate, but produces the undesirable effect
of constructing discontinuous response surfaces due to the disagreement between
neighboring models at their shared boundary. In this paper, we extend this idea
by allowing for a small amount of communication between neighboring spatial
partitions which encourages better alignment of the local models, leading to
smoother spatial predictions and a better fit in general. Due to our
decentralized communication scheme, the proposed extension remains highly
scalable and adds very little overhead in terms of computation (and none, in
terms of memory). We demonstrate this Partitioned SVGP (PSVGP) approach for the
Energy Exascale Earth System Model (E3SM) and compare the results to the
independent SVGP case.

### 9. [Structural Effect and Spectral Enhancement of High-Dimensional Regularized Linear Discriminant Analysis](http://arxiv.org/pdf/2507.16682v1)

Authors: Yonghan Zhang, Zhangni Pu, Lu Yan, Jiang Hu

Regularized linear discriminant analysis (RLDA) is a widely used tool for
classification and dimensionality reduction, but its performance in
high-dimensional scenarios is inconsistent. Existing theoretical analyses of
RLDA often lack clear insight into how data structure affects classification
performance. To address this issue, we derive a non-asymptotic approximation of
the misclassification rate and thus analyze the structural effect and
structural adjustment strategies of RLDA. Based on this, we propose the
Spectral Enhanced Discriminant Analysis (SEDA) algorithm, which optimizes the
data structure by adjusting the spiked eigenvalues of the population covariance
matrix. By developing a new theoretical result on eigenvectors in random matrix
theory, we derive an asymptotic approximation on the misclassification rate of
SEDA. The bias correction algorithm and parameter selection strategy are then
obtained. Experiments on synthetic and real datasets show that SEDA achieves
higher classification accuracy and dimensionality reduction compared to
existing LDA methods.

### 10. [Testing the variety hypothesis](http://arxiv.org/pdf/2507.16705v1)

Authors: A. Lerario, P. Roos Hoefgeest, M. Scolamiero, A. Tamai

Given a probability measure on the unit disk, we study the problem of
deciding whether, for some threshold probability, this measure is supported
near a real algebraic variety of given dimension and bounded degree. We call
this "testing the variety hypothesis". We prove an upper bound on the so-called
"sample complexity" of this problem and show how it can be reduced to a
semialgebraic decision problem. This is done by studying in a quantitative way
the Hausdorff geometry of the space of real algebraic varieties of a given
dimension and degree.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-23 PST.

### 1. [A novel ensemble Wasserstein GAN framework for effective anomaly detection in industrial internet of things environments](https://www.nature.com/articles/s41598-025-07533-1)

Authors: Rubina Riaz et al.

### 2. [Multilingual identification of nuanced dimensions of hope speech in social media texts](https://www.nature.com/articles/s41598-025-10683-x)

Authors: Grigori Sidorov et al.

### 3. [Frame topology fusion-based hierarchical graph convolution for automatic assessment of physical rehabilitation exercises](https://www.nature.com/articles/s41598-025-12020-8)

Authors: Shaohui Zhang et al.

### 4. [Unmasking insider threats using a robust hybrid optimized generative pretrained neural network approach](https://www.nature.com/articles/s41598-025-12127-y)

Authors: P. Lavanya et al.

### 5. [Contextualizing ancient texts with generative neural networks](https://www.nature.com/articles/s41586-025-09292-5)

Authors: Yannis Assael et al.

### 6. [A novel and secure artificial intelligence enabled zero trust intrusion detection in industrial internet of things architecture](https://www.nature.com/articles/s41598-025-11738-9)

Authors: Asif Ali Laghari et al.

### 7. [An explainable transformer model for Alzheimer’s disease detection using retinal imaging](https://www.nature.com/articles/s41598-025-12498-2)

Authors: Saeed Jamshidiha et al.

### 8. [Texture-preserving and information loss minimization method for infrared and visible image fusion](https://www.nature.com/articles/s41598-025-11482-0)

Authors: Qiyuan He et al.

### 9. [Multi-scale spatio-temporal graph neural network for urban traffic flow prediction](https://www.nature.com/articles/s41598-025-11072-0)

Authors: Hui Chen et al.

### 10. [A hybrid approach combining Bayesian networks and logistic regression for enhancing risk assessment](https://www.nature.com/articles/s41598-025-10291-9)

Authors: Xueyuan Wei et al.

### 11. [Tackling inter-subject variability in smartwatch data using factorization models](https://www.nature.com/articles/s41598-025-12102-7)

Authors: Arman Naseri et al.

