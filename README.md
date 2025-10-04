# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-03 17:00:24.950419 PST.

### Artificial Intelligence

### 1. [LOGicalThought: Logic-Based Ontological Grounding of LLMs for High-Assurance Reasoning](http://arxiv.org/pdf/2510.01530v1)

Authors: Navapat Nananukul, Yue Zhang, Ryan Lee, Eric Boxer, Jonathan May, Vibhav Giridhar Gogate, Jay Pujara, Mayank Kejriwal

High-assurance reasoning, particularly in critical domains such as law and
medicine, requires conclusions that are accurate, verifiable, and explicitly
grounded in evidence. This reasoning relies on premises codified from rules,
statutes, and contracts, inherently involving defeasible or non-monotonic logic
due to numerous exceptions, where the introduction of a single fact can
invalidate general rules, posing significant challenges. While large language
models (LLMs) excel at processing natural language, their capabilities in
standard inference tasks do not translate to the rigorous reasoning required
over high-assurance text guidelines. Core reasoning challenges within such
texts often manifest specific logical structures involving negation,
implication, and, most critically, defeasible rules and exceptions. In this
paper, we propose a novel neurosymbolically-grounded architecture called
LOGicalThought (LogT) that uses an advanced logical language and reasoner in
conjunction with an LLM to construct a dual symbolic graph context and
logic-based context. These two context representations transform the problem
from inference over long-form guidelines into a compact grounded evaluation.
Evaluated on four multi-domain benchmarks against four baselines, LogT improves
overall performance by 11.84% across all LLMs. Performance improves
significantly across all three modes of reasoning: by up to +10.2% on negation,
+13.2% on implication, and +5.5% on defeasible reasoning compared to the
strongest baseline.

### 2. [Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models](http://arxiv.org/pdf/2510.01544v1)

Authors: Shaoan Xie, Lingjing Kong, Xiangchen Song, Xinshuai Dong, Guangyi Chen, Eric P. Xing, Kun Zhang

Diffusion language models (dLLMs) offer a promising, non-autoregressive
paradigm for text generation, yet training them for complex reasoning remains a
key challenge. Current reinforcement learning approaches often rely on sparse,
outcome-based rewards, which can reinforce flawed reasoning paths that lead to
coincidentally correct answers. We argue that this stems from a fundamental
mismatch with the natural structure of reasoning. We first propose a
theoretical framework that formalizes complex problem solving as a hierarchical
selection process, where an intractable global constraint is decomposed into a
series of simpler, localized logical steps. This framework provides a
principled foundation for algorithm design, including theoretical insights into
the identifiability of this latent reasoning structure. Motivated by this
theory, we identify unstructured refinement -- a failure mode where a model's
iterative steps do not contribute meaningfully to the solution -- as a core
deficiency in existing methods. We then introduce Step-Aware Policy
Optimization (SAPO), a novel RL algorithm that aligns the dLLM's denoising
process with the latent reasoning hierarchy. By using a process-based reward
function that encourages incremental progress, SAPO guides the model to learn
structured, coherent reasoning paths. Our empirical results show that this
principled approach significantly improves performance on challenging reasoning
benchmarks and enhances the interpretability of the generation process.

### 3. [AdvEvo-MARL: Shaping Internalized Safety through Adversarial Co-Evolution in Multi-Agent Reinforcement Learning](http://arxiv.org/pdf/2510.01586v1)

Authors: Zhenyu Pan, Yiting Zhang, Zhuo Liu, Yolo Yunlong Tang, Zeliang Zhang, Haozheng Luo, Yuwei Han, Jianshu Zhang, Dennis Wu, Hong-Yu Chen, Haoran Lu, Haoyang Fang, Manling Li, Chenliang Xu, Philip S. Yu, Han Liu

LLM-based multi-agent systems excel at planning, tool use, and role
coordination, but their openness and interaction complexity also expose them to
jailbreak, prompt-injection, and adversarial collaboration. Existing defenses
fall into two lines: (i) self-verification that asks each agent to pre-filter
unsafe instructions before execution, and (ii) external guard modules that
police behaviors. The former often underperforms because a standalone agent
lacks sufficient capacity to detect cross-agent unsafe chains and
delegation-induced risks; the latter increases system overhead and creates a
single-point-of-failure-once compromised, system-wide safety collapses, and
adding more guards worsens cost and complexity. To solve these challenges, we
propose AdvEvo-MARL, a co-evolutionary multi-agent reinforcement learning
framework that internalizes safety into task agents. Rather than relying on
external guards, AdvEvo-MARL jointly optimizes attackers (which synthesize
evolving jailbreak prompts) and defenders (task agents trained to both
accomplish their duties and resist attacks) in adversarial learning
environments. To stabilize learning and foster cooperation, we introduce a
public baseline for advantage estimation: agents within the same functional
group share a group-level mean-return baseline, enabling lower-variance updates
and stronger intra-group coordination. Across representative attack scenarios,
AdvEvo-MARL consistently keeps attack-success rate (ASR) below 20%, whereas
baselines reach up to 38.33%, while preserving-and sometimes improving-task
accuracy (up to +3.67% on reasoning tasks). These results show that safety and
utility can be jointly improved without relying on extra guard agents or added
system overhead.

### 4. [AgentRec: Next-Generation LLM-Powered Multi-Agent Collaborative Recommendation with Adaptive Intelligence](http://arxiv.org/pdf/2510.01609v1)

Authors: Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Lau

Interactive conversational recommender systems have gained significant
attention for their ability to capture user preferences through natural
language interactions. However, existing approaches face substantial challenges
in handling dynamic user preferences, maintaining conversation coherence, and
balancing multiple ranking objectives simultaneously. This paper introduces
AgentRec, a next-generation LLM-powered multi-agent collaborative
recommendation framework that addresses these limitations through hierarchical
agent networks with adaptive intelligence. Our approach employs specialized
LLM-powered agents for conversation understanding, preference modeling, context
awareness, and dynamic ranking, coordinated through an adaptive weighting
mechanism that learns from interaction patterns. We propose a three-tier
learning strategy combining rapid response for simple queries, intelligent
reasoning for complex preferences, and deep collaboration for challenging
scenarios. Extensive experiments on three real-world datasets demonstrate that
AgentRec achieves consistent improvements over state-of-the-art baselines, with
2.8\% enhancement in conversation success rate, 1.9\% improvement in
recommendation accuracy (NDCG@10), and 3.2\% better conversation efficiency
while maintaining comparable computational costs through intelligent agent
coordination.

### 5. [Learning to Decide with Just Enough: Information-Theoretic Context Summarization for CDMPs](http://arxiv.org/pdf/2510.01620v1)

Authors: Peidong Liu, Junjiang Lin, Shaowen Wang, Yao Xu, Haiqing Li, Xuhao Xie, Siyi Wu, Hao Li

Contextual Markov Decision Processes (CMDPs) offer a framework for sequential
decision-making under external signals, but existing methods often fail to
generalize in high-dimensional or unstructured contexts, resulting in excessive
computation and unstable performance. We propose an information-theoretic
summarization approach that uses large language models (LLMs) to compress
contextual inputs into low-dimensional, semantically rich summaries. These
summaries augment states by preserving decision-critical cues while reducing
redundancy. Building on the notion of approximate context sufficiency, we
provide, to our knowledge, the first regret bounds and a latency-entropy
trade-off characterization for CMDPs. Our analysis clarifies how
informativeness impacts computational cost. Experiments across discrete,
continuous, visual, and recommendation benchmarks show that our method
outperforms raw-context and non-context baselines, improving reward, success
rate, and sample efficiency, while reducing latency and memory usage. These
findings demonstrate that LLM-based summarization offers a scalable and
interpretable solution for efficient decision-making in context-rich,
resource-constrained environments.

### 6. [Understanding the Geospatial Reasoning Capabilities of LLMs: A Trajectory Recovery Perspective](http://arxiv.org/pdf/2510.01639v1)

Authors: Thinh Hung Truong, Jey Han Lau, Jianzhong Qi

We explore the geospatial reasoning capabilities of Large Language Models
(LLMs), specifically, whether LLMs can read road network maps and perform
navigation. We frame trajectory recovery as a proxy task, which requires models
to reconstruct masked GPS traces, and introduce GLOBALTRACE, a dataset with
over 4,000 real-world trajectories across diverse regions and transportation
modes. Using road network as context, our prompting framework enables LLMs to
generate valid paths without accessing any external navigation tools.
Experiments show that LLMs outperform off-the-shelf baselines and specialized
trajectory recovery models, with strong zero-shot generalization. Fine-grained
analysis shows that LLMs have strong comprehension of the road network and
coordinate systems, but also pose systematic biases with respect to regions and
transportation modes. Finally, we demonstrate how LLMs can enhance navigation
experiences by reasoning over maps in flexible ways to incorporate user
preferences.

### 7. [GuruAgents: Emulating Wise Investors with Prompt-Guided LLM Agents](http://arxiv.org/pdf/2510.01664v1)

Authors: Yejin Kim, Youngbin Lee, Juhyeong Kim, Yongjae Lee

This study demonstrates that GuruAgents, prompt-guided AI agents, can
systematically operationalize the strategies of legendary investment gurus. We
develop five distinct GuruAgents, each designed to emulate an iconic investor,
by encoding their distinct philosophies into LLM prompts that integrate
financial tools and a deterministic reasoning pipeline. In a backtest on
NASDAQ-100 constituents from Q4 2023 to Q2 2025, the GuruAgents exhibit unique
behaviors driven by their prompted personas. The Buffett GuruAgent achieves the
highest performance, delivering a 42.2\% CAGR that significantly outperforms
benchmarks, while other agents show varied results. These findings confirm that
prompt engineering can successfully translate the qualitative philosophies of
investment gurus into reproducible, quantitative strategies, highlighting a
novel direction for automated systematic investing. The source code and data
are available at https://github.com/yejining99/GuruAgents.

### 8. [MetaboT: AI-based agent for natural language-based interaction with metabolomics knowledge graphs](http://arxiv.org/pdf/2510.01724v1)

Authors: Madina Bekbergenova, Lucas Pradi, Benjamin Navet, Emma Tysinger, Franck Michel, Matthieu Feraud, Yousouf Taghzouti, Yan Zhou Chen, Olivier Kirchhoffer, Florence Mehl, Martin Legrand, Tao Jiang, Marco Pagni, Soha Hassoun, Jean-Luc Wolfender, Wout Bittremieux, Fabien Gandon, Louis-Félix Nothias

Mass spectrometry metabolomics generates vast amounts of data requiring
advanced methods for interpretation. Knowledge graphs address these challenges
by structuring mass spectrometry data, metabolite information, and their
relationships into a connected network (Gaudry et al. 2024). However, effective
use of a knowledge graph demands an in-depth understanding of its ontology and
its query language syntax. To overcome this, we designed MetaboT, an AI system
utilizing large language models (LLMs) to translate user questions into SPARQL
semantic query language for operating on knowledge graphs (Steve Harris 2013).
We demonstrate its effectiveness using the Experimental Natural Products
Knowledge Graph (ENPKG), a large-scale public knowledge graph for plant natural
products (Gaudry et al. 2024).MetaboT employs specialized AI agents for
handling user queries and interacting with the knowledge graph by breaking down
complex tasks into discrete components, each managed by a specialised agent
(Fig. 1a). The multi-agent system is constructed using the LangChain and
LangGraph libraries, which facilitate the integration of LLMs with external
tools and information sources (LangChain, n.d.). The query generation process
follows a structured workflow. First, the Entry Agent determines if the
question is new or a follow-up to previous interactions. New questions are
forwarded to the Validator Agent, which verifies if the question is related to
the knowledge graph. Then, the valid question is sent to the Supervisor Agent,
which identifies if the question requires chemical conversions or standardized
identifiers. In this case it delegates the question to the Knowledge Graph
Agent, which can use tools to extract necessary details, such as URIs or
taxonomies of chemical names, from the user query. Finally, an agent
responsible for crafting the SPARQL queries equipped with the ontology of the
knowledge graph uses the provided identifiers to generate the query. Then, the
system executes the generated query against the metabolomics knowledge graph
and returns structured results to the user (Fig. 1b). To assess the performance
of MetaboT we have curated 50 metabolomics-related questions and their expected
answers. In addition to submitting these questions to MetaboT, we evaluated a
baseline by submitting them to a standard LLM (GPT-4o) with a prompt that
incorporated the knowledge graph ontology but did not provide specific entity
IDs. This baseline achieved only 8.16% accuracy, compared to MetaboT's 83.67%,
underscoring the necessity of our multi-agent system for accurately retrieving
entities and generating correct SPARQL queries. MetaboT demonstrates promising
performance as a conversational question-answering assistant, enabling
researchers to retrieve structured metabolomics data through natural language
queries. By automating the generation and execution of SPARQL queries, it
removes technical barriers that have traditionally hindered access to knowledge
graphs. Importantly, MetaboT leverages the capabilities of LLMs while
maintaining experimentally grounded query generation, ensuring that outputs
remain aligned with domain-specific standards and data structures. This
approach facilitates data-driven discoveries by bridging the gap between
complex semantic technologies and user-friendly interaction. MetaboT is
accessible at [https://metabot.holobiomicslab.eu/], and its source code is
available at [https://github.com/HolobiomicsLab/MetaboT].

### 9. [A cybersecurity AI agent selection and decision support framework](http://arxiv.org/pdf/2510.01751v1)

Authors: Masike Malatji

This paper presents a novel, structured decision support framework that
systematically aligns diverse artificial intelligence (AI) agent architectures,
reactive, cognitive, hybrid, and learning, with the comprehensive National
Institute of Standards and Technology (NIST) Cybersecurity Framework (CSF) 2.0.
By integrating agent theory with industry guidelines, this framework provides a
transparent and stepwise methodology for selecting and deploying AI solutions
to address contemporary cyber threats. Employing a granular decomposition of
NIST CSF 2.0 functions into specific tasks, the study links essential AI agent
properties such as autonomy, adaptive learning, and real-time responsiveness to
each subcategory's security requirements. In addition, it outlines graduated
levels of autonomy (assisted, augmented, and fully autonomous) to accommodate
organisations at varying stages of cybersecurity maturity. This holistic
approach transcends isolated AI applications, providing a unified detection,
incident response, and governance strategy. Through conceptual validation, the
framework demonstrates how tailored AI agent deployments can align with
real-world constraints and risk profiles, enhancing situational awareness,
accelerating response times, and fortifying long-term resilience via adaptive
risk management. Ultimately, this research bridges the gap between theoretical
AI constructs and operational cybersecurity demands, establishing a foundation
for robust, empirically validated multi-agent systems that adhere to industry
standards.

### 10. [REBot: From RAG to CatRAG with Semantic Enrichment and Graph Routing](http://arxiv.org/pdf/2510.01800v1)

Authors: Thanh Ma, Tri-Tam La, Lam-Thu Le Huu, Minh-Nghi Nguyen, Khanh-Van Pham Luu, Huu-Hoa Nguyen

Academic regulation advising is essential for helping students interpret and
comply with institutional policies, yet building effective systems requires
domain specific regulatory resources. To address this challenge, we propose
REBot, an LLM enhanced advisory chatbot powered by CatRAG, a hybrid retrieval
reasoning framework that integrates retrieval augmented generation with graph
based reasoning. CatRAG unifies dense retrieval and graph reasoning, supported
by a hierarchical, category labeled knowledge graph enriched with semantic
features for domain alignment. A lightweight intent classifier routes queries
to the appropriate retrieval modules, ensuring both factual accuracy and
contextual depth. We construct a regulation specific dataset and evaluate REBot
on classification and question answering tasks, achieving state of the art
performance with an F1 score of 98.89%. Finally, we implement a web application
that demonstrates the practical value of REBot in real world academic advising
scenarios.

### Hardware Architecture

### 1. [Edge GPU Aware Multiple AI Model Pipeline for Accelerated MRI Reconstruction and Analysis](http://arxiv.org/pdf/2510.01730v1)

Authors: Ashiyana Abdul Majeed, Mahmoud Meribout, Safa Mohammed Sali

Advancements in AI have greatly enhanced the medical imaging process, making
it quicker to diagnose patients. However, very few have investigated the
optimization of a multi-model system with hardware acceleration. As specialized
edge devices emerge, the efficient use of their accelerators is becoming
increasingly crucial. This paper proposes a hardware-accelerated method for
simultaneous reconstruction and diagnosis of \ac{MRI} from \ac{CT} images.
Real-time performance of achieving a throughput of nearly 150 frames per second
was achieved by leveraging hardware engines available in modern NVIDIA edge
GPU, along with scheduling techniques. This includes the GPU and the \ac{DLA}
available in both Jetson AGX Xavier and Jetson AGX Orin, which were considered
in this paper. The hardware allocation of different layers of the multiple AI
models was done in such a way that the ideal time between the hardware engines
is reduced. In addition, the AI models corresponding to the \ac{GAN} model were
fine-tuned in such a way that no fallback execution into the GPU engine is
required without compromising accuracy. Indeed, the accuracy corresponding to
the fine-tuned edge GPU-aware AI models exhibited an accuracy enhancement of
5\%. A further hardware allocation of two fine-tuned GPU-aware GAN models
proves they can double the performance over the original model, leveraging
adequate partitioning on the NVIDIA Jetson AGX Xavier and Orin devices. The
results prove the effectiveness of employing hardware-aware models in parallel
for medical image analysis and diagnosis.

### 2. [Multiplier-free In-Memory Vector-Matrix Multiplication Using Distributed Arithmetic](http://arxiv.org/pdf/2510.02099v1)

Authors: Felix Zeller, John Reuben, Dietmar Fey

Vector-Matrix Multiplication (VMM) is the fundamental and frequently required
computation in inference of Neural Networks (NN). Due to the large data
movement required during inference, VMM can benefit greatly from in-memory
computing. However, ADC/DACs required for in-memory VMM consume significant
power and area. `Distributed Arithmetic (DA)', a technique in computer
architecture prevalent in 1980s was used to achieve inner product or dot
product of two vectors without using a hard-wired multiplier when one of the
vectors is a constant. In this work, we extend the DA technique to multiply an
input vector with a constant matrix. By storing the sum of the weights in
memory, DA achieves VMM using shift-and-add circuits in the periphery of ReRAM
memory. We verify functional and also estimate non-functional properties
(latency, energy, area) by performing transistor-level simulations. Using
energy-efficient sensing and fine grained pipelining, our approach achieves 4.5
x less latency and 12 x less energy than VMM performed in memory conventionally
by bit slicing. Furthermore, DA completely eliminated the need for power-hungry
ADCs which are the main source of area and energy consumption in the current
VMM implementations in memory.

### Computational Complexity

### 1. [Positive Univariate Polynomials: SOS certificates, algorithms, bit complexity, and T-systems](http://arxiv.org/pdf/2510.01701v1)

Authors: Matías Bender, Philipp Di Dio, Elias Tsigaridas

We study certificates of positivity for univariate polynomials with rational
coefficients that are positive over (an interval of) $\mathbb{R}$, given as
weighted sums of squares (SOS) of rational polynomials. We build on the
algorithm of Chevillard, Harrison, Joldes, and Lauter~\cite{chml-usos-alg-11},
which we call \usos. For a polynomial of degree~$d$ and coefficient
bitsize~$\tau$, we show that a rational weighted SOS representation can be
computed in $\widetilde{\mathcal{O}}_B(d^3 + d^2 \tau)$ bit operations, and the
certificate has bitsize $\widetilde{\mathcal{O}}(d^2 \tau)$. This improves the
best-known bounds by a factor~$d$ and completes previous analyses. We also
extend the method to positivity over arbitrary rational intervals, again saving
a factor~$d$. For univariate rational polynomials we further introduce
\emph{perturbed SOS certificates}. These consist of a sum of two rational
squares approximating the input polynomial so that nonnegativity of the
approximation implies that of the original. Their computation has the same bit
complexity and certificate size as in the weighted SOS case. We also
investigate structural properties of these SOS decompositions. Using the
classical fact that any nonnegative univariate real polynomial is a sum of two
real squares, we prove that the summands form an interlacing pair. Their real
roots correspond to the Karlin points of the original polynomial, linking our
construction to the T-systems of Karlin~\cite{Karlin-repr-pos-63}. This enables
explicit computation of such decompositions, whereas only existential results
were previously known. We obtain analogous results for positivity over
$(0,\infty)$ and thus over arbitrary real intervals. Finally, we present an
open-source Maple implementation of \usos and report experiments on diverse
inputs that demonstrate its efficiency.

### 2. [Minimum Selective Subset on Unit Disk Graphs and Circle Graphs](http://arxiv.org/pdf/2510.01931v1)

Authors: Bubai Manna

In a connected simple graph G = (V(G),E(G)), each vertex is assigned one of c
colors, where V(G) can be written as a union of a total of c subsets
V_{1},...,V_{c} and V_{i} denotes the set of vertices of color i. A subset S of
V(G) is called a selective subset if, for every i, every vertex v in V_{i} has
at least one nearest neighbor in $S \cup (V(G) \setminus V_{i})$ that also lies
in V_{i}. The Minimum Selective Subset (MSS) problem asks for a selective
subset of minimum size.
  We show that the MSS problem is log-APX-hard on general graphs, even when
c=2. As a consequence, the problem does not admit a polynomial-time
approximation scheme (PTAS) unless P = NP. On the positive side, we present a
PTAS for unit disk graphs, which works without requiring a geometric
representation and applies for arbitrary c. We further prove that MSS remains
NP-complete in unit disk graphs for arbitrary c. In addition, we show that the
MSS problem is log-APX-hard on circle graphs, even when c=2.

### 3. [Formal Framework for Quantum Advantage](http://arxiv.org/pdf/2510.01953v1)

Authors: Harry Buhrman, Niklas Galke, Konstantinos Meichanetzidis

Motivated by notions of quantum heuristics and by average-case rather than
worst-case algorithmic analysis, we define quantum computational advantage in
terms of individual problem instances. Inspired by the classical notions of
Kolmogorov complexity and instance complexity, we define their quantum
versions. This allows us to define queasy instances of computational problems,
like e.g. Satisfiability and Factoring, as those whose quantum instance
complexity is significantly smaller than their classical instance complexity.
These instances indicate quantum advantage: they are easy to solve on a quantum
computer, but classical algorithms struggle (they feel queasy). Via a reduction
from Factoring, we prove the existence of queasy Satisfiability instances;
specifically, these instances are maximally queasy (under reasonable
complexity-theoretic assumptions). Further, we show that there is exponential
algorithmic utility in the queasiness of a quantum algorithm. This formal
framework serves as a beacon that guides the hunt for quantum advantage in
practice, and moreover, because its focus lies on single instances, it can lead
to new ways of designing quantum algorithms.

### Computational Engineering

### 1. [A Copula-Based Variational Autoencoder for Uncertainty Quantification in Inverse Problems: Application to Damage Identification in an Offshore Wind Turbine](http://arxiv.org/pdf/2510.02013v1)

Authors: Ana Fernandez-Navamuel, Matteo Croci, Martin Alberto Diaz Viera

Structural Health Monitoring of Floating Offshore Wind Turbines (FOWTs) is
critical for ensuring operational safety and efficiency. However, identifying
damage in components like mooring systems from limited sensor data poses a
challenging inverse problem, often characterized by multimodal solutions where
various damage states could explain the observed response. To overcome it, we
propose a Variational Autoencoder (VAE) architecture, where the encoder
approximates the inverse operator, while the decoder approximates the forward.
The posterior distribution of the latent space variables is probabilistically
modeled, describing the uncertainties in the estimates. This work tackles the
limitations of conventional Gaussian Mixtures used within VAEs, which can be
either too restrictive or computationally prohibitive for high-dimensional
spaces. We propose a novel Copula-based VAE architecture that decouples the
marginal distribution of the variables from their dependence structure,
offering a flexible method for representing complex, correlated posterior
distributions. We provide a comprehensive comparison of three different
approaches for approximating the posterior: a Gaussian Mixture with a diagonal
covariance matrix, a Gaussian Mixture with a full covariance matrix, and a
Gaussian Copula. Our analysis, conducted on a high-fidelity synthetic dataset,
demonstrates that the Copula VAE offers a promising and tractable solution in
high-dimensional spaces. Although the present work remains in the
two-dimensional space, the results suggest efficient scalability to higher
dimensions. It achieves superior performance with significantly fewer
parameters than the Gaussian Mixture alternatives, whose parametrization grows
prohibitively with the dimensionality. The results underscore the potential of
Copula-based VAEs as a tool for uncertainty-aware damage identification in FOWT
mooring systems.

### 2. [ShapeGen3DCP: A Deep Learning Framework for Layer Shape Prediction in 3D Concrete Printing](http://arxiv.org/pdf/2510.02009v1)

Authors: Giacomo Rizzieri, Federico Lanteri, Liberato Ferrara, Massimiliano Cremonesi

This work introduces ShapeGen3DCP, a deep learning framework for fast and
accurate prediction of filament cross-sectional geometry in 3D Concrete
Printing (3DCP). The method is based on a neural network architecture that
takes as input both material properties in the fluid state (density, yield
stress, plastic viscosity) and process parameters (nozzle diameter, nozzle
height, printing and flow velocities) to directly predict extruded layer
shapes. To enhance generalization, some inputs are reformulated into
dimensionless parameters that capture underlying physical principles. Predicted
geometries are compactly represented using Fourier descriptors, which enforce
smooth, closed, and symmetric profiles while reducing the prediction task to a
small set of coefficients. The training dataset was synthetically generated
using a well-established Particle Finite Element (PFEM) model of 3DCP,
overcoming the scarcity of experimental data. Validation against diverse
numerical and experimental cases shows strong agreement, confirming the
framework's accuracy and reliability. This opens the way to practical uses
ranging from pre-calibration of print settings, minimizing or even eliminating
trial-and-error adjustments, to toolpath optimization for more advanced
designs. Looking ahead, coupling the framework with simulations and sensor
feedback could enable closed-loop digital twins for 3DCP, driving real-time
process optimization, defect detection, and adaptive control of printing
parameters.

### 3. [CardioRAG: A Retrieval-Augmented Generation Framework for Multimodal Chagas Disease Detection](http://arxiv.org/pdf/2510.01558v1)

Authors: Zhengyang Shen, Xuehao Zhai, Hua Tu, Mayue Shi

Chagas disease affects nearly 6 million people worldwide, with Chagas
cardiomyopathy representing its most severe complication. In regions where
serological testing capacity is limited, AI-enhanced electrocardiogram (ECG)
screening provides a critical diagnostic alternative. However, existing machine
learning approaches face challenges such as limited accuracy, reliance on large
labeled datasets, and more importantly, weak integration with evidence-based
clinical diagnostic indicators. We propose a retrieval-augmented generation
framework, CardioRAG, integrating large language models with interpretable
ECG-based clinical features, including right bundle branch block, left anterior
fascicular block, and heart rate variability metrics. The framework uses
variational autoencoder-learned representations for semantic case retrieval,
providing contextual cases to guide clinical reasoning. Evaluation demonstrated
high recall performance of 89.80%, with a maximum F1 score of 0.68 for
effective identification of positive cases requiring prioritized serological
testing. CardioRAG provides an interpretable, clinical evidence-based approach
particularly valuable for resource-limited settings, demonstrating a pathway
for embedding clinical indicators into trustworthy medical AI systems.

### 4. [LLM-Enhanced, Data-Driven Personalized and Equitable Clinician Scheduling: A Predict-then-Optimize Approach](http://arxiv.org/pdf/2510.02047v1)

Authors: Anjali Jha, Wanqing Chen, Maxim Eckmann, Ian Stockwell, Jianwu Wang, Kai Sun

Clinician scheduling remains a persistent challenge due to limited clinical
resources and fluctuating demands. This complexity is especially acute in large
academic anesthesiology departments as physicians balance responsibilities
across multiple clinical sites with conflicting priorities. Further, scheduling
must account for individual clinical and lifestyle preferences to ensure job
satisfaction and well-being. Traditional approaches, often based on statistical
or rule-based optimization models, rely on structured data and explicit domain
knowledge. However, these methods often overlook unstructured information,
e.g., free-text notes from routinely administered clinician well-being surveys
and scheduling platforms. These notes may reveal implicit and underutilized
clinical resources. Neglecting such information can lead to misaligned
schedules, increased burnout, overlooked staffing flexibility, and suboptimal
utilization of available resources. To address this gap, we propose a
predict-then-optimize framework that integrates classification-based clinician
availability predictions with a mixed-integer programming schedule optimization
model. Large language models (LLMs) are employed to extract actionable
preferences and implicit constraints from unstructured schedule notes,
enhancing the reliability of availability predictions. These predictions then
inform the schedule optimization considering four objectives: first, ensuring
clinical full-time equivalent compliance, second, reducing workload imbalances
by enforcing equitable proportions of shift types, third, maximizing clinician
availability for assigned shifts, and fourth, schedule consistency. By
combining the interpretive power of LLMs with the rigor of mathematical
optimization, our framework provides a robust, data-driven solution that
enhances operational efficiency while supporting equity and clinician
well-being.

### Computational Geometry

### 1. [Minimum Selective Subset on Unit Disk Graphs and Circle Graphs](http://arxiv.org/pdf/2510.01931v1)

Authors: Bubai Manna

In a connected simple graph G = (V(G),E(G)), each vertex is assigned one of c
colors, where V(G) can be written as a union of a total of c subsets
V_{1},...,V_{c} and V_{i} denotes the set of vertices of color i. A subset S of
V(G) is called a selective subset if, for every i, every vertex v in V_{i} has
at least one nearest neighbor in $S \cup (V(G) \setminus V_{i})$ that also lies
in V_{i}. The Minimum Selective Subset (MSS) problem asks for a selective
subset of minimum size.
  We show that the MSS problem is log-APX-hard on general graphs, even when
c=2. As a consequence, the problem does not admit a polynomial-time
approximation scheme (PTAS) unless P = NP. On the positive side, we present a
PTAS for unit disk graphs, which works without requiring a geometric
representation and applies for arbitrary c. We further prove that MSS remains
NP-complete in unit disk graphs for arbitrary c. In addition, we show that the
MSS problem is log-APX-hard on circle graphs, even when c=2.

### 2. [Bifurcation: How to Explore a Tree](http://arxiv.org/pdf/2510.01939v1)

Authors: Sariel Har-Peled

Avraham et al. [AFK+15] presented an alternative approach to parametric
search, called \emph{bifurcation}, that performs faster under certain
circumstances. Intuitively, when the underlying decider execution can be rolled
back cheaply and the decider has a near-linear running time. For some problems,
this leads to fast algorithms that beat the seemingly natural lower bound
arising from distance selection.
  Bifurcation boils down to a tree exploration problem. You are given a binary
(unfortunately implicit) tree of height $n$ and $k$ internal nodes with two
children (all other internal nodes have a single child), and assume each node
has an associated parameter value. These values are sorted in the inorder
traversal of the tree. Assume there is (say) a node (not necessarily a leaf)
that is the target node that the exploration needs to discover.
  The player starts from the root. At each step, the player can move to
adjacent nodes to the current location (i.e., one of the children or the
parent). Alternatively, the player can call an oracle on the current node,
which returns either that it is the target (thus, mission accomplished!) or
whether the target value is strictly smaller or larger than the current one.
  A naive algorithm explores the whole tree, in $O(n k)$ time, then performs
$O(\log k n)$ calls to the oracle to find the desired leaf. Avraham \etal
showed that this can be improved to $O(n \sqrt{k} )$ time, and $O( \sqrt{k}
\log n)$ oracle calls.
  Here, we improve this to $O(n \sqrt{k} )$ time, with only $ O( \sqrt{k} +
\log n)$ oracle calls. We also show matching lower bounds, under certain
assumptions. We believe our interpretation of bifurcation as a tree exploration
problem, and the associated algorithm, are of independent interest.

### Computation and Language

### 1. [CLUE: Non-parametric Verification from Experience via Hidden-State Clustering](http://arxiv.org/pdf/2510.01591v1)

Authors: Zhenwen Liang, Ruosen Li, Yujun Zhou, Linfeng Song, Dian Yu, Xinya Du, Haitao Mi, Dong Yu

Assessing the quality of Large Language Model (LLM) outputs presents a
critical challenge. Previous methods either rely on text-level information
(e.g., reward models, majority voting), which can overfit to superficial cues,
or on calibrated confidence from token probabilities, which would fail on
less-calibrated models. Yet both of these signals are, in fact, partial
projections of a richer source of information: the model's internal hidden
states. Early layers, closer to token embeddings, preserve semantic and lexical
features that underpin text-based judgments, while later layers increasingly
align with output logits, embedding confidence-related information. This paper
explores hidden states directly as a unified foundation for verification. We
show that the correctness of a solution is encoded as a geometrically separable
signature within the trajectory of hidden activations. To validate this, we
present Clue (Clustering and Experience-based Verification), a deliberately
minimalist, non-parametric verifier. With no trainable parameters, CLUE only
summarizes each reasoning trace by an hidden state delta and classifies
correctness via nearest-centroid distance to ``success'' and ``failure''
clusters formed from past experience. The simplicity of this method highlights
the strength of the underlying signal. Empirically, CLUE consistently
outperforms LLM-as-a-judge baselines and matches or exceeds modern
confidence-based methods in reranking candidates, improving both top-1 and
majority-vote accuracy across AIME 24/25 and GPQA. As a highlight, on AIME 24
with a 1.5B model, CLUE boosts accuracy from 56.7% (majority@64) to 70.0%
(top-maj@16).

### 2. [Efficient Training of Robust Traditional Chinese LLaMA-1B on a Single Consumer GPU: Continual Pre-training, SFT, and DPO](http://arxiv.org/pdf/2510.01616v1)

Authors: Yu-Cheng Chih, Ming-Tao Duan, Yong-Hao Hou

Small Language Models (SLMs) enable cost-effective, on-device and
latency-sensitive AI applications, yet their deployment in Traditional Chinese
(TC) remains hindered by token-level instability - models unpredictably emit
non-TC characters or code-switch into other languages. We address this
practical reliability gap by creating PureTC-1B, a three-stage stabilization
pipeline for Llama-3.2-1B-Instruct (an open-weight, instruction-tuned model
released by Meta) using parameter-efficient LoRA adapters. Our method combines
Continual Pre-Training (CPT) on TC-centric corpora, Supervised Fine-Tuning
(SFT) with instruction data, and Direct Preference Optimization (DPO) using
TC-adherence preferences to improve monolingual robustness without full-model
retraining. On a benchmark designed to simulate real-world usage, PureTC-1B
achieves a 51.3% relative reduction (micro-average) in non-TC output tokens
versus the base model. On a Named Entity Translation (NET) task, PureTC-1B
further reduces incorrect-language tokens by 77.2% relative to Llama-3B and
57.2% relative to Qwen-1.5B, indicating that robust TC adherence is attainable
even at the 1B scale. The pipeline is reproducible, adapter-only, and
hardware-friendly, offering practitioners a practical recipe to enhance
language stability for TC and potentially other non-English languages.

### 3. [AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System](http://arxiv.org/pdf/2510.01617v1)

Authors: Hui Yi Leong, Yuheng Li, Yuqing Wu, Wenwen Ouyang, Wei Zhu, Jiechao Gao

Although large language models (LLMs) have revolutionized natural language
processing capabilities, their practical implementation as autonomous
multi-agent systems (MAS) for industrial problem-solving encounters persistent
barriers. Conventional MAS architectures are fundamentally restricted by
inflexible, hand-crafted graph topologies that lack contextual responsiveness,
resulting in diminished efficacy across varied academic and commercial
workloads. To surmount these constraints, we introduce AMAS, a
paradigm-shifting framework that redefines LLM-based MAS through a novel
dynamic graph designer. This component autonomously identifies task-specific
optimal graph configurations via lightweight LLM adaptation, eliminating the
reliance on monolithic, universally applied structural templates. Instead, AMAS
exploits the intrinsic properties of individual inputs to intelligently direct
query trajectories through task-optimized agent pathways. Rigorous validation
across question answering, mathematical deduction, and code generation
benchmarks confirms that AMAS systematically exceeds state-of-the-art
single-agent and multi-agent approaches across diverse LLM architectures. Our
investigation establishes that context-sensitive structural adaptability
constitutes a foundational requirement for high-performance LLM MAS
deployments.

### 4. [Learning to Look at the Other Side: A Semantic Probing Study of Word Embeddings in LLMs with Enabled Bidirectional Attention](http://arxiv.org/pdf/2510.01652v1)

Authors: Zhaoxin Feng, Jianfei Ma, Emmanuele Chersoni, Xiaojing Zhao, Xiaoyi Bao

Autoregressive Large Language Models (LLMs) demonstrate exceptional
performance in language understanding and generation. However, their
application in text embedding tasks has been relatively slow, along with the
analysis of their semantic representation in probing tasks, due to the
constraints of the unidirectional attention mechanism.
  This paper aims to explore whether such constraints can be overcome by
enabling bidirectional attention in LLMs. We tested different variants of the
Llama architecture through additional training steps, progressively enabling
bidirectional attention and unsupervised/supervised contrastive learning.

### 5. [What MLLMs Learn about When they Learn about Multimodal Reasoning: Perception, Reasoning, or their Integration?](http://arxiv.org/pdf/2510.01719v1)

Authors: Jiwan Chung, Neel Joshi, Pratyusha Sharma, Youngjae Yu, Vibhav Vineet

Multimodal reasoning models have recently shown promise on challenging
domains such as olympiad-level geometry, yet their evaluation remains dominated
by aggregate accuracy, a single score that obscures where and how models are
improving. We introduce MathLens, a benchmark designed to disentangle the
subskills of multimodal reasoning while preserving the complexity of
textbook-style geometry problems. The benchmark separates performance into
three components: Perception: extracting information from raw inputs,
Reasoning: operating on available information, and Integration: selecting
relevant perceptual evidence and applying it within reasoning. To support each
test, we provide annotations: visual diagrams, textual descriptions to evaluate
reasoning in isolation, controlled questions that require both modalities, and
probes for fine-grained perceptual skills, all derived from symbolic
specifications of the problems to ensure consistency and robustness. Our
analysis reveals that different training approaches have uneven effects: First,
reinforcement learning chiefly strengthens perception, especially when
supported by textual supervision, while textual SFT indirectly improves
perception through reflective reasoning. Second, reasoning improves only in
tandem with perception. Third, integration remains the weakest capacity, with
residual errors concentrated there once other skills advance. Finally,
robustness diverges: RL improves consistency under diagram variation, whereas
multimodal SFT reduces it through overfitting. We will release all data and
experimental logs.

### 6. [Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network](http://arxiv.org/pdf/2510.01801v1)

Authors: Xin Liu, Rongwu Xu, Xinyi Jia, Jason Liao, Jiao Sun, Ling Huang, Wei Xu

The rise of large language models (LLMs) has enabled the generation of highly
persuasive spam reviews that closely mimic human writing. These reviews pose
significant challenges for existing detection systems and threaten the
credibility of online platforms. In this work, we first create three realistic
LLM-generated spam review datasets using three distinct LLMs, each guided by
product metadata and genuine reference reviews. Evaluations by GPT-4.1 confirm
the high persuasion and deceptive potential of these reviews. To address this
threat, we propose FraudSquad, a hybrid detection model that integrates text
embeddings from a pre-trained language model with a gated graph transformer for
spam node classification. FraudSquad captures both semantic and behavioral
signals without relying on manual feature engineering or massive training
resources. Experiments show that FraudSquad outperforms state-of-the-art
baselines by up to 44.22% in precision and 43.01% in recall on three
LLM-generated datasets, while also achieving promising results on two
human-written spam datasets. Furthermore, FraudSquad maintains a modest model
size and requires minimal labeled training data, making it a practical solution
for real-world applications. Our contributions include new synthetic datasets,
a practical detection framework, and empirical evidence highlighting the
urgency of adapting spam detection to the LLM era. Our code and datasets are
available at: https://anonymous.4open.science/r/FraudSquad-5389/.

### 7. [Syntactic Blind Spots: How Misalignment Leads to LLMs Mathematical Errors](http://arxiv.org/pdf/2510.01831v1)

Authors: Dane Williamson, Yangfeng Ji, Matthew Dwyer

Large Language Models (LLMs) demonstrate strong mathematical problem-solving
abilities but frequently fail on problems that deviate syntactically from their
training distribution. We identify a systematic failure mode, syntactic blind
spots, in which models misapply familiar reasoning strategies to problems that
are semantically straightforward but phrased in unfamiliar ways. These errors
are not due to gaps in mathematical competence, but rather reflect a brittle
coupling between surface form and internal representation. To test this, we
rephrase incorrectly answered questions using syntactic templates drawn from
correct examples. These rephrasings, which preserve semantics while reducing
structural complexity, often lead to correct answers. We quantify syntactic
complexity using a metric based on Dependency Locality Theory (DLT), and show
that higher DLT scores are associated with increased failure rates across
multiple datasets. Our findings suggest that many reasoning errors stem from
structural misalignment rather than conceptual difficulty, and that
syntax-aware interventions can reveal and mitigate these inductive failures.

### 8. [SCRIBES: Web-Scale Script-Based Semi-Structured Data Extraction with Reinforcement Learning](http://arxiv.org/pdf/2510.01832v1)

Authors: Shicheng Liu, Kai Sun, Lisheng Fu, Xilun Chen, Xinyuan Zhang, Zhaojiang Lin, Rulin Shao, Yue Liu, Anuj Kumar, Wen-tau Yih, Xin Luna Dong

Semi-structured content in HTML tables, lists, and infoboxes accounts for a
substantial share of factual data on the web, yet the formatting complicates
usage, and reliably extracting structured information from them remains
challenging. Existing methods either lack generalization or are
resource-intensive due to per-page LLM inference. In this paper, we introduce
SCRIBES (SCRIpt-Based Semi-Structured Content Extraction at Web-Scale), a novel
reinforcement learning framework that leverages layout similarity across
webpages within the same site as a reward signal. Instead of processing each
page individually, SCRIBES generates reusable extraction scripts that can be
applied to groups of structurally similar webpages. Our approach further
improves by iteratively training on synthetic annotations from in-the-wild
CommonCrawl data. Experiments show that our approach outperforms strong
baselines by over 13% in script quality and boosts downstream question
answering accuracy by more than 4% for GPT-4o, enabling scalable and
resource-efficient web information extraction.

### 9. [Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey](http://arxiv.org/pdf/2510.01925v1)

Authors: Qiyuan Liu, Hao Xu, Xuhong Chen, Wei Chen, Yee Whye Teh, Ning Miao

Reward models (RMs) play a critical role in enhancing the reasoning
performance of LLMs. For example, they can provide training signals to finetune
LLMs during reinforcement learning (RL) and help select the best answer from
multiple candidates during inference. In this paper, we provide a systematic
introduction to RMs, along with a comprehensive survey of their applications in
LLM reasoning. We first review fundamental concepts of RMs, including their
architectures, training methodologies, and evaluation techniques. Then, we
explore their key applications: (1) guiding generation and selecting optimal
outputs during LLM inference, (2) facilitating data synthesis and iterative
self-improvement for LLMs, and (3) providing training signals in RL-based
finetuning. Finally, we address critical open questions regarding the
selection, generalization, evaluation, and enhancement of RMs, based on
existing research and our own empirical findings. Our analysis aims to provide
actionable insights for the effective deployment and advancement of RMs for LLM
reasoning.

### 10. [Inverse Language Modeling towards Robust and Grounded LLMs](http://arxiv.org/pdf/2510.01929v1)

Authors: Davide Gabrielli, Simone Sestito, Iacopo Masi

The current landscape of defensive mechanisms for LLMs is fragmented and
underdeveloped, unlike prior work on classifiers. To further promote
adversarial robustness in LLMs, we propose Inverse Language Modeling (ILM), a
unified framework that simultaneously 1) improves the robustness of LLMs to
input perturbations, and, at the same time, 2) enables native grounding by
inverting model outputs to identify potentially toxic or unsafe input triggers.
ILM transforms LLMs from static generators into analyzable and robust systems,
potentially helping RED teaming. ILM can lay the foundation for next-generation
LLMs that are not only robust and grounded but also fundamentally more
controllable and trustworthy. The code is publicly available at
github.com/davegabe/pag-llm.

### Cryptography and Security

### 1. [Towards Imperceptible Adversarial Defense: A Gradient-Driven Shield against Facial Manipulations](http://arxiv.org/pdf/2510.01699v1)

Authors: Yue Li, Linying Xue, Dongdong Lin, Qiushi Li, Hui Tian, Hongxia Wang

With the flourishing prosperity of generative models, manipulated facial
images have become increasingly accessible, raising concerns regarding privacy
infringement and societal trust. In response, proactive defense strategies
embed adversarial perturbations into facial images to counter deepfake
manipulation. However, existing methods often face a tradeoff between
imperceptibility and defense effectiveness-strong perturbations may disrupt
forgeries but degrade visual fidelity. Recent studies have attempted to address
this issue by introducing additional visual loss constraints, yet often
overlook the underlying gradient conflicts among losses, ultimately weakening
defense performance. To bridge the gap, we propose a gradient-projection-based
adversarial proactive defense (GRASP) method that effectively counters facial
deepfakes while minimizing perceptual degradation. GRASP is the first approach
to successfully integrate both structural similarity loss and low-frequency
loss to enhance perturbation imperceptibility. By analyzing gradient conflicts
between defense effectiveness loss and visual quality losses, GRASP pioneers
the design of the gradient-projection mechanism to mitigate these conflicts,
enabling balanced optimization that preserves image fidelity without
sacrificing defensive performance. Extensive experiments validate the efficacy
of GRASP, achieving a PSNR exceeding 40 dB, SSIM of 0.99, and a 100% defense
success rate against facial attribute manipulations, significantly
outperforming existing approaches in visual quality.

### 2. [Constructions of Efficiently Implementable Boolean Functions with Provable Nonlinearity/Resiliency/Algebraic Immunity Trade-Offs](http://arxiv.org/pdf/2510.01720v1)

Authors: Palash Sarkar

We describe several families of efficiently implementable Boolean functions
achieving provable trade-offs between resiliency, nonlinearity, and algebraic
immunity. In concrete terms, the following result holds for each of the
function families that we propose. Given integers $m_0\geq 0$, $x_0\geq 1$, and
$a_0\geq 1$, it is possible to construct an $n$-variable function which has
resiliency at least $m_0$, linear bias (which is an equivalent method of
expressing nonlinearity) at most $2^{-x_0}$ and algebraic immunity at least
$a_0$; further, $n$ is linear in $m_0$, $x_0$ and $a_0$, and the function can
be implemented using $O(n)$ gates.

### 3. [Bypassing Prompt Guards in Production with Controlled-Release Prompting](http://arxiv.org/pdf/2510.01529v1)

Authors: Jaiden Fairoze, Sanjam Garg, Keewoo Lee, Mingyuan Wang

As large language models (LLMs) advance, ensuring AI safety and alignment is
paramount. One popular approach is prompt guards, lightweight mechanisms
designed to filter malicious queries while being easy to implement and update.
In this work, we introduce a new attack that circumvents such prompt guards,
highlighting their limitations. Our method consistently jailbreaks production
models while maintaining response quality, even under the highly protected chat
interfaces of Google Gemini (2.5 Flash/Pro), DeepSeek Chat (DeepThink), Grok
(3), and Mistral Le Chat (Magistral). The attack exploits a resource asymmetry
between the prompt guard and the main LLM, encoding a jailbreak prompt that
lightweight guards cannot decode but the main model can. This reveals an attack
surface inherent to lightweight prompt guards in modern LLM architectures and
underscores the need to shift defenses from blocking malicious inputs to
preventing malicious outputs. We additionally identify other critical alignment
issues, such as copyrighted data extraction, training data extraction, and
malicious response leakage during thinking.

### 4. [POLAR: Automating Cyber Threat Prioritization through LLM-Powered Assessment](http://arxiv.org/pdf/2510.01552v1)

Authors: Luoxi Tang, Yuqiao Meng, Ankita Patra, Weicheng Ma, Muchao Ye, Zhaohan Xi

Large Language Models (LLMs) are intensively used to assist security analysts
in counteracting the rapid exploitation of cyber threats, wherein LLMs offer
cyber threat intelligence (CTI) to support vulnerability assessment and
incident response. While recent work has shown that LLMs can support a wide
range of CTI tasks such as threat analysis, vulnerability detection, and
intrusion defense, significant performance gaps persist in practical
deployments. In this paper, we investigate the intrinsic vulnerabilities of
LLMs in CTI, focusing on challenges that arise from the nature of the threat
landscape itself rather than the model architecture. Using large-scale
evaluations across multiple CTI benchmarks and real-world threat reports, we
introduce a novel categorization methodology that integrates stratification,
autoregressive refinement, and human-in-the-loop supervision to reliably
analyze failure instances. Through extensive experiments and human inspections,
we reveal three fundamental vulnerabilities: spurious correlations,
contradictory knowledge, and constrained generalization, that limit LLMs in
effectively supporting CTI. Subsequently, we provide actionable insights for
designing more robust LLM-powered CTI systems to facilitate future research.

### 5. [Evaluating the Robustness of a Production Malware Detection System to Transferable Adversarial Attacks](http://arxiv.org/pdf/2510.01676v1)

Authors: Milad Nasr, Yanick Fratantonio, Luca Invernizzi, Ange Albertini, Loua Farah, Alex Petit-Bianco, Andreas Terzis, Kurt Thomas, Elie Bursztein, Nicholas Carlini

As deep learning models become widely deployed as components within larger
production systems, their individual shortcomings can create system-level
vulnerabilities with real-world impact. This paper studies how adversarial
attacks targeting an ML component can degrade or bypass an entire
production-grade malware detection system, performing a case study analysis of
Gmail's pipeline where file-type identification relies on a ML model.
  The malware detection pipeline in use by Gmail contains a machine learning
model that routes each potential malware sample to a specialized malware
classifier to improve accuracy and performance. This model, called Magika, has
been open sourced. By designing adversarial examples that fool Magika, we can
cause the production malware service to incorrectly route malware to an
unsuitable malware detector thereby increasing our chance of evading detection.
Specifically, by changing just 13 bytes of a malware sample, we can
successfully evade Magika in 90% of cases and thereby allow us to send malware
files over Gmail. We then turn our attention to defenses, and develop an
approach to mitigate the severity of these types of attacks. For our defended
production model, a highly resourced adversary requires 50 bytes to achieve
just a 20% attack success rate. We implement this defense, and, thanks to a
collaboration with Google engineers, it has already been deployed in production
for the Gmail classifier.

### 6. [Mirage Fools the Ear, Mute Hides the Truth: Precise Targeted Adversarial Attacks on Polyphonic Sound Event Detection Systems](http://arxiv.org/pdf/2510.02158v1)

Authors: Junjie Su, Weifei Jin, Yuxin Cao, Derui Wang, Kai Ye, Jie Hao

Sound Event Detection (SED) systems are increasingly deployed in
safety-critical applications such as industrial monitoring and audio
surveillance. However, their robustness against adversarial attacks has not
been well explored. Existing audio adversarial attacks targeting SED systems,
which incorporate both detection and localization capabilities, often lack
effectiveness due to SED's strong contextual dependencies or lack precision by
focusing solely on misclassifying the target region as the target event,
inadvertently affecting non-target regions. To address these challenges, we
propose the Mirage and Mute Attack (M2A) framework, which is designed for
targeted adversarial attacks on polyphonic SED systems. In our optimization
process, we impose specific constraints on the non-target output, which we
refer to as preservation loss, ensuring that our attack does not alter the
model outputs for non-target region, thus achieving precise attacks.
Furthermore, we introduce a novel evaluation metric Editing Precison (EP) that
balances effectiveness and precision, enabling our method to simultaneously
enhance both. Comprehensive experiments show that M2A achieves 94.56% and
99.11% EP on two state-of-the-art SED models, demonstrating that the framework
is sufficiently effective while significantly enhancing attack precision.

### 7. [NoMod: A Non-modular Attack on Module Learning With Errors](http://arxiv.org/pdf/2510.02162v1)

Authors: Cristian Bassotto, Ermes Franch, Marina Krček, Stjepan Picek

The advent of quantum computing threatens classical public-key cryptography,
motivating NIST's adoption of post-quantum schemes such as those based on the
Module Learning With Errors (Module-LWE) problem. We present NoMod ML-Attack, a
hybrid white-box cryptanalytic method that circumvents the challenge of
modeling modular reduction by treating wrap-arounds as statistical corruption
and casting secret recovery as robust linear estimation. Our approach combines
optimized lattice preprocessing--including reduced-vector saving and algebraic
amplification--with robust estimators trained via Tukey's Biweight loss.
Experiments show NoMod achieves full recovery of binary secrets for dimension
$n = 350$, recovery of sparse binomial secrets for $n = 256$, and successful
recovery of sparse secrets in CRYSTALS-Kyber settings with parameters $(n, k) =
(128, 3)$ and $(256, 2)$. We release our implementation in an anonymous
repository https://anonymous.4open.science/r/NoMod-3BD4.

### 8. [TAIBOM: Bringing Trustworthiness to AI-Enabled Systems](http://arxiv.org/pdf/2510.02169v1)

Authors: Vadim Safronov, Anthony McCaigue, Nicholas Allott, Andrew Martin

The growing integration of open-source software and AI-driven technologies
has introduced new layers of complexity into the software supply chain,
challenging existing methods for dependency management and system assurance.
While Software Bills of Materials (SBOMs) have become critical for enhancing
transparency and traceability, current frameworks fall short in capturing the
unique characteristics of AI systems -- namely, their dynamic, data-driven
nature and the loosely coupled dependencies across datasets, models, and
software components. These challenges are compounded by fragmented governance
structures and the lack of robust tools for ensuring integrity, trust, and
compliance in AI-enabled environments.
  In this paper, we introduce Trusted AI Bill of Materials (TAIBOM) -- a novel
framework extending SBOM principles to the AI domain. TAIBOM provides (i) a
structured dependency model tailored for AI components, (ii) mechanisms for
propagating integrity statements across heterogeneous AI pipelines, and (iii) a
trust attestation process for verifying component provenance. We demonstrate
how TAIBOM supports assurance, security, and compliance across AI workflows,
highlighting its advantages over existing standards such as SPDX and CycloneDX.
This work lays the foundation for trustworthy and verifiable AI systems through
structured software transparency.

### 9. [Authentication Security of PRF GNSS Ranging](http://arxiv.org/pdf/2510.02196v1)

Authors: Jason Anderson

This work derives the authentication security of pseudorandom function (PRF)
GNSS ranging under multiple GNSS spoofing models, including the Security Code
Estimation and Replay (SCER) spoofer. When GNSS ranging codes derive from a PRF
utilizing a secret known only to the broadcaster, the spoofer cannot predict
the ranging code before broadcast. Therefore, PRF ranging can be used to
establish trust in the GNSS pseudoranges and the resulting receiver position,
navigation, and timing (PNT) solution. I apply the methods herein to Galileo's
Signal Authentication Service (SAS) utilizing the encrypted Galileo E6-C signal
to compute that, at most, 400 ms of Galileo E6-C data to assert 128-bit
authentication security under non-SCER models. For the SCER adversary, I
predict the adversary's needed receiving radio equipment to break
authentication security. One can use this work to design a PRF GNSS ranging
protocol to meet useful authentication security requirements by computing the
probability of missed detection.

### 10. [Reproducible Builds for Quantum Computing](http://arxiv.org/pdf/2510.02251v1)

Authors: Iyán Méndez Veiga, Esther Hänggi

Reproducible builds are a set of software development practices that
establish an independently verifiable path from source code to binary
artifacts, helping to detect and mitigate certain classes of supply chain
attacks. Although quantum computing is a rapidly evolving field of research, it
can already benefit from adopting reproducible builds. This paper aims to
bridge the gap between the quantum computing and reproducible builds
communities. We propose a generalization of the definition of reproducible
builds in the quantum setting, motivated by two threat models: one targeting
the confidentiality of end users' data during circuit preparation and
submission to a quantum computer, and another compromising the integrity of
quantum computation results. This work presents three examples that show how
classical information can be hidden in transpiled quantum circuits, and two
cases illustrating how even minimal modifications to these circuits can lead to
incorrect quantum computation results. Our work provides initial steps towards
a framework for reproducibility in quantum software toolchains.

### Computer Vision and Pattern Recognition

### 1. [MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](http://arxiv.org/pdf/2510.01532v1)

Authors: Meilong Xu, Xiaoling Hu, Shahira Abousamra, Chen Li, Chao Chen

In semi-supervised segmentation, capturing meaningful semantic structures
from unlabeled data is essential. This is particularly challenging in
histopathology image analysis, where objects are densely distributed. To
address this issue, we propose a semi-supervised segmentation framework
designed to robustly identify and preserve relevant topological features. Our
method leverages multiple perturbed predictions obtained through stochastic
dropouts and temporal training snapshots, enforcing topological consistency
across these varied outputs. This consistency mechanism helps distinguish
biologically meaningful structures from transient and noisy artifacts. A key
challenge in this process is to accurately match the corresponding topological
features across the predictions in the absence of ground truth. To overcome
this, we introduce a novel matching strategy that integrates spatial overlap
with global structural alignment, minimizing discrepancies among predictions.
Extensive experiments demonstrate that our approach effectively reduces
topological errors, resulting in more robust and accurate segmentations
essential for reliable downstream analysis. Code is available at
\href{https://github.com/Melon-Xu/MATCH}{https://github.com/Melon-Xu/MATCH}.

### 2. [Towards Better Optimization For Listwise Preference in Diffusion Models](http://arxiv.org/pdf/2510.01540v1)

Authors: Jiamu Bai, Xin Yu, Meilong Xu, Weitao Lu, Xin Pan, Kiwan Maeng, Daniel Kifer, Jian Wang, Yu Wang

Reinforcement learning from human feedback (RLHF) has proven effectiveness
for aligning text-to-image (T2I) diffusion models with human preferences.
Although Direct Preference Optimization (DPO) is widely adopted for its
computational efficiency and avoidance of explicit reward modeling, its
applications to diffusion models have primarily relied on pairwise preferences.
The precise optimization of listwise preferences remains largely unaddressed.
In practice, human feedback on image preferences often contains implicit ranked
information, which conveys more precise human preferences than pairwise
comparisons. In this work, we propose Diffusion-LPO, a simple and effective
framework for Listwise Preference Optimization in diffusion models with
listwise data. Given a caption, we aggregate user feedback into a ranked list
of images and derive a listwise extension of the DPO objective under the
Plackett-Luce model. Diffusion-LPO enforces consistency across the entire
ranking by encouraging each sample to be preferred over all of its lower-ranked
alternatives. We empirically demonstrate the effectiveness of Diffusion-LPO
across various tasks, including text-to-image generation, image editing, and
personalized preference alignment. Diffusion-LPO consistently outperforms
pairwise DPO baselines on visual quality and preference alignment.

### 3. [Consistent Assistant Domains Transformer for Source-free Domain Adaptation](http://arxiv.org/pdf/2510.01559v1)

Authors: Renrong Shao, Wei Zhang, Kangyang Luo, Qin Li, and Jun Wang

Source-free domain adaptation (SFDA) aims to address the challenge of
adapting to a target domain without accessing the source domain directly.
However, due to the inaccessibility of source domain data, deterministic
invariable features cannot be obtained. Current mainstream methods primarily
focus on evaluating invariant features in the target domain that closely
resemble those in the source domain, subsequently aligning the target domain
with the source domain. However, these methods are susceptible to hard samples
and influenced by domain bias. In this paper, we propose a Consistent Assistant
Domains Transformer for SFDA, abbreviated as CADTrans, which solves the issue
by constructing invariable feature representations of domain consistency.
Concretely, we develop an assistant domain module for CADTrans to obtain
diversified representations from the intermediate aggregated global attentions,
which addresses the limitation of existing methods in adequately representing
diversity. Based on assistant and target domains, invariable feature
representations are obtained by multiple consistent strategies, which can be
used to distinguish easy and hard samples. Finally, to align the hard samples
to the corresponding easy samples, we construct a conditional multi-kernel max
mean discrepancy (CMK-MMD) strategy to distinguish between samples of the same
category and those of different categories. Extensive experiments are conducted
on various benchmarks such as Office-31, Office-Home, VISDA-C, and
DomainNet-126, proving the significant performance improvements achieved by our
proposed approaches. Code is available at
https://github.com/RoryShao/CADTrans.git.

### 4. [Joint Deblurring and 3D Reconstruction for Macrophotography](http://arxiv.org/pdf/2510.01640v1)

Authors: Yifan Zhao, Liangchen Li, Yuqi Zhou, Kai Wang, Yan Liang, Juyong Zhang

Macro lens has the advantages of high resolution and large magnification, and
3D modeling of small and detailed objects can provide richer information.
However, defocus blur in macrophotography is a long-standing problem that
heavily hinders the clear imaging of the captured objects and high-quality 3D
reconstruction of them. Traditional image deblurring methods require a large
number of images and annotations, and there is currently no multi-view 3D
reconstruction method for macrophotography. In this work, we propose a joint
deblurring and 3D reconstruction method for macrophotography. Starting from
multi-view blurry images captured, we jointly optimize the clear 3D model of
the object and the defocus blur kernel of each pixel. The entire framework
adopts a differentiable rendering method to self-supervise the optimization of
the 3D model and the defocus blur kernel. Extensive experiments show that from
a small number of multi-view images, our proposed method can not only achieve
high-quality image deblurring but also recover high-fidelity 3D appearance.

### 5. [FideDiff: Efficient Diffusion Model for High-Fidelity Image Motion Deblurring](http://arxiv.org/pdf/2510.01641v1)

Authors: Xiaoyang Liu, Zhengyan Zhou, Zihang Xu, Jiezhang Cao, Zheng Chen, Yulun Zhang

Recent advancements in image motion deblurring, driven by CNNs and
transformers, have made significant progress. Large-scale pre-trained diffusion
models, which are rich in true-world modeling, have shown great promise for
high-quality image restoration tasks such as deblurring, demonstrating stronger
generative capabilities than CNN and transformer-based methods. However,
challenges such as unbearable inference time and compromised fidelity still
limit the full potential of the diffusion models. To address this, we introduce
FideDiff, a novel single-step diffusion model designed for high-fidelity
deblurring. We reformulate motion deblurring as a diffusion-like process where
each timestep represents a progressively blurred image, and we train a
consistency model that aligns all timesteps to the same clean image. By
reconstructing training data with matched blur trajectories, the model learns
temporal consistency, enabling accurate one-step deblurring. We further enhance
model performance by integrating Kernel ControlNet for blur kernel estimation
and introducing adaptive timestep prediction. Our model achieves superior
performance on full-reference metrics, surpassing previous diffusion-based
methods and matching the performance of other state-of-the-art models. FideDiff
offers a new direction for applying pre-trained diffusion models to
high-fidelity image restoration tasks, establishing a robust baseline for
further advancing diffusion models in real-world industrial applications. Our
dataset and code will be available at https://github.com/xyLiu339/FideDiff.

### 6. [LadderMoE: Ladder-Side Mixture of Experts Adapters for Bronze Inscription Recognition](http://arxiv.org/pdf/2510.01651v1)

Authors: Rixin Zhou, Peiqiang Qiu, Qian Zhang, Chuntao Li, Xi Yang

Bronze inscriptions (BI), engraved on ritual vessels, constitute a crucial
stage of early Chinese writing and provide indispensable evidence for
archaeological and historical studies. However, automatic BI recognition
remains difficult due to severe visual degradation, multi-domain variability
across photographs, rubbings, and tracings, and an extremely long-tailed
character distribution. To address these challenges, we curate a large-scale BI
dataset comprising 22454 full-page images and 198598 annotated characters
spanning 6658 unique categories, enabling robust cross-domain evaluation.
Building on this resource, we develop a two-stage detection-recognition
pipeline that first localizes inscriptions and then transcribes individual
characters. To handle heterogeneous domains and rare classes, we equip the
pipeline with LadderMoE, which augments a pretrained CLIP encoder with
ladder-style MoE adapters, enabling dynamic expert specialization and stronger
robustness. Comprehensive experiments on single-character and full-page
recognition tasks demonstrate that our method substantially outperforms
state-of-the-art scene text recognition baselines, achieving superior accuracy
across head, mid, and tail categories as well as all acquisition modalities.
These results establish a strong foundation for bronze inscription recognition
and downstream archaeological analysis.

### 7. [VirDA: Reusing Backbone for Unsupervised Domain Adaptation with Visual Reprogramming](http://arxiv.org/pdf/2510.01660v1)

Authors: Duy Nguyen, Dat Nguyen

Existing UDA pipelines fine-tune already well-trained backbone parameters for
every new source-and-target pair, resulting in the number of training
parameters and storage memory growing linearly with each new pair, and also
preventing the reuse of these well-trained backbone parameters.
  Inspired by recent implications that existing backbones have textural biases,
we propose making use of domain-specific textural bias for domain adaptation
via visual reprogramming, namely VirDA.Instead of fine-tuning the full
backbone, VirDA prepends a domain-specific visual reprogramming layer to the
backbone. This layer produces visual prompts that act as an added textural bias
to the input image, adapting its ``style'' to a target domain. To optimize
these visual reprogramming layers, we use multiple objective functions that
optimize the intra- and inter-domain distribution differences when
domain-adapting visual prompts are applied. This process does not require
modifying the backbone parameters, allowing the same backbone to be reused
across different domains.
  We evaluate VirDA on Office-31 and obtain 92.8% mean accuracy with only 1.5M
trainable parameters. VirDA surpasses PDA, the state-of-the-art
parameter-efficient UDA baseline, by +1.6% accuracy while using just 46% of its
parameters. Compared with full-backbone fine-tuning, VirDA outperforms CDTrans
and FixBi by +0.2% and +1.4%, respectively, while requiring only 1.7% and 2.8%
of their trainable parameters. Relative to the strongest current methods
(PMTrans and TVT), VirDA uses ~1.7% of their parameters and trades off only
2.2% and 1.1% accuracy, respectively.

### 8. [Discrete Facial Encoding: : A Framework for Data-driven Facial Display Discovery](http://arxiv.org/pdf/2510.01662v1)

Authors: Minh Tran, Maksim Siniukov, Zhangyu Jin, Mohammad Soleymani

Facial expression analysis is central to understanding human behavior, yet
existing coding systems such as the Facial Action Coding System (FACS) are
constrained by limited coverage and costly manual annotation. In this work, we
introduce Discrete Facial Encoding (DFE), an unsupervised, data-driven
alternative of compact and interpretable dictionary of facial expressions from
3D mesh sequences learned through a Residual Vector Quantized Variational
Autoencoder (RVQ-VAE). Our approach first extracts identity-invariant
expression features from images using a 3D Morphable Model (3DMM), effectively
disentangling factors such as head pose and facial geometry. We then encode
these features using an RVQ-VAE, producing a sequence of discrete tokens from a
shared codebook, where each token captures a specific, reusable facial
deformation pattern that contributes to the overall expression. Through
extensive experiments, we demonstrate that Discrete Facial Encoding captures
more precise facial behaviors than FACS and other facial encoding alternatives.
We evaluate the utility of our representation across three high-level
psychological tasks: stress detection, personality prediction, and depression
detection. Using a simple Bag-of-Words model built on top of the learned
tokens, our system consistently outperforms both FACS-based pipelines and
strong image and video representation learning models such as Masked
Autoencoders. Further analysis reveals that our representation covers a wider
variety of facial displays, highlighting its potential as a scalable and
effective alternative to FACS for psychological and affective computing
applications.

### 9. [UniVerse: Unleashing the Scene Prior of Video Diffusion Models for Robust Radiance Field Reconstruction](http://arxiv.org/pdf/2510.01669v1)

Authors: Jin Cao, Hongrui Wu, Ziyong Feng, Hujun Bao, Xiaowei Zhou, Sida Peng

This paper tackles the challenge of robust reconstruction, i.e., the task of
reconstructing a 3D scene from a set of inconsistent multi-view images. Some
recent works have attempted to simultaneously remove image inconsistencies and
perform reconstruction by integrating image degradation modeling into neural 3D
scene representations.However, these methods rely heavily on dense observations
for robustly optimizing model parameters.To address this issue, we propose to
decouple robust reconstruction into two subtasks: restoration and
reconstruction, which naturally simplifies the optimization process.To this
end, we introduce UniVerse, a unified framework for robust reconstruction based
on a video diffusion model. Specifically, UniVerse first converts inconsistent
images into initial videos, then uses a specially designed video diffusion
model to restore them into consistent images, and finally reconstructs the 3D
scenes from these restored images.Compared with case-by-case per-view
degradation modeling, the diffusion model learns a general scene prior from
large-scale data, making it applicable to diverse image
inconsistencies.Extensive experiments on both synthetic and real-world datasets
demonstrate the strong generalization capability and superior performance of
our method in robust reconstruction. Moreover, UniVerse can control the style
of the reconstructed 3D scene. Project page:
https://jin-cao-tma.github.io/UniVerse.github.io/

### 10. [An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution](http://arxiv.org/pdf/2510.01678v1)

Authors: Ke Jia, Ji Zhou, Hanxin Li, Zhigan Zhou, Haojie Chu, Xiaojie Li

In industrial inspection and component alignment tasks, template matching
requires efficient estimation of a target's position and geometric state
(rotation and scaling) under complex backgrounds to support precise downstream
operations. Traditional methods rely on exhaustive enumeration of angles and
scales, leading to low efficiency under compound transformations. Meanwhile,
most deep learning-based approaches only estimate similarity scores without
explicitly modeling geometric pose, making them inadequate for real-world
deployment. To overcome these limitations, we propose a lightweight end-to-end
framework that reformulates template matching as joint localization and
geometric regression, outputting the center coordinates, rotation angle, and
independent horizontal and vertical scales. A Template-Aware Dynamic
Convolution Module (TDCM) dynamically injects template features at inference to
guide generalizable matching. The compact network integrates depthwise
separable convolutions and pixel shuffle for efficient matching. To enable
geometric-annotation-free training, we introduce a rotation-shear-based
augmentation strategy with structure-aware pseudo labels. A lightweight
refinement module further improves angle and scale precision via local
optimization. Experiments show our 3.07M model achieves high precision and 14ms
inference under compound transformations. It also demonstrates strong
robustness in small-template and multi-object scenarios, making it highly
suitable for deployment in real-time industrial applications. The code is
available at:https://github.com/ZhouJ6610/PoseMatch-TDCM.

### Computers and Society

### 1. [Small is Sufficient: Reducing the World AI Energy Consumption Through Model Selection](http://arxiv.org/pdf/2510.01889v1)

Authors: Tiago da Silva Barros, Frédéric Giroire, Ramon Aparicio-Pardo, Joanna Moulierac

The energy consumption and carbon footprint of Artificial Intelligence (AI)
have become critical concerns due to rising costs and environmental impacts. In
response, a new trend in green AI is emerging, shifting from the "bigger is
better" paradigm, which prioritizes large models, to "small is sufficient",
emphasizing energy sobriety through smaller, more efficient models.
  We explore how the AI community can adopt energy sobriety today by focusing
on model selection during inference. Model selection consists of choosing the
most appropriate model for a given task, a simple and readily applicable
method, unlike approaches requiring new hardware or architectures. Our
hypothesis is that, as in many industrial activities, marginal utility gains
decrease with increasing model size. Thus, applying model selection can
significantly reduce energy consumption while maintaining good utility for AI
inference.
  We conduct a systematic study of AI tasks, analyzing their popularity, model
size, and efficiency. We examine how the maturity of different tasks and model
adoption patterns impact the achievable energy savings, ranging from 1% to 98%
for different tasks. Our estimates indicate that applying model selection could
reduce AI energy consumption by 27.8%, saving 31.9 TWh worldwide in 2025 -
equivalent to the annual output of five nuclear power reactors.

### 2. [TriAlignXA: An Explainable Trilemma Alignment Framework for Trustworthy Agri-product Grading](http://arxiv.org/pdf/2510.01990v1)

Authors: Jianfei Xie, Ziyang Li

The 'trust deficit' in online fruit and vegetable e-commerce stems from the
inability of digital transactions to provide direct sensory perception of
product quality. This paper constructs a 'Trust Pyramid' model through
'dual-source verification' of consumer trust. Experiments confirm that quality
is the cornerstone of trust. The study reveals an 'impossible triangle' in
agricultural product grading, comprising biological characteristics,
timeliness, and economic viability, highlighting the limitations of traditional
absolute grading standards. To quantitatively assess this trade-off, we propose
the 'Triangular Trust Index' (TTI). We redefine the role of algorithms from
'decision-makers' to 'providers of transparent decision-making bases',
designing the explainable AI framework--TriAlignXA. This framework supports
trustworthy online transactions within agricultural constraints through
multi-objective optimization. Its core relies on three engines: the
Bio-Adaptive Engine for granular quality description; the Timeliness
Optimization Engine for processing efficiency; and the Economic Optimization
Engine for cost control. Additionally, the "Pre-Mapping Mechanism" encodes
process data into QR codes, transparently conveying quality information.
Experiments on grading tasks demonstrate significantly higher accuracy than
baseline models. Empirical evidence and theoretical analysis verify the
framework's balancing capability in addressing the "impossible triangle". This
research provides comprehensive support--from theory to practice--for building
a trustworthy online produce ecosystem, establishing a critical pathway from
algorithmic decision-making to consumer trust.

### 3. [The Current State of AI Bias Bounties: An Overview of Existing Programmes and Research](http://arxiv.org/pdf/2510.02036v1)

Authors: Sergej Kucenko, Nathaniel Dennler, Fengxiang He

Current bias evaluation methods rarely engage with communities impacted by AI
systems. Inspired by bug bounties, bias bounties have been proposed as a
reward-based method that involves communities in AI bias detection by asking
users of AI systems to report biases they encounter when interacting with such
systems. In the absence of a state-of-the-art review, this survey aimed to
identify and analyse existing AI bias bounty programmes and to present academic
literature on bias bounties. Google, Google Scholar, PhilPapers, and IEEE
Xplore were searched, and five bias bounty programmes, as well as five research
publications, were identified. All bias bounties were organised by U.S.-based
organisations as time-limited contests, with public participation in four
programmes and prize pools ranging from 7,000 to 24,000 USD. The five research
publications included a report on the application of bug bounties to
algorithmic harms, an article addressing Twitter's bias bounty, a proposal for
bias bounties as an institutional mechanism to increase AI scrutiny, a workshop
discussing bias bounties from queer perspectives, and an algorithmic framework
for bias bounties. We argue that reducing the technical requirements to enter
bounty programmes is important to include those without coding experience.
Given the limited adoption of bias bounties, future efforts should explore the
transferability of the best practices from bug bounties and examine how such
programmes can be designed to be sensitive to underrepresented groups while
lowering adoption barriers for organisations.

### 4. [Komitee Equal Shares: Choosing Together as Voters and as Groups with a Co-designed Virtual Budget Algorithm](http://arxiv.org/pdf/2510.02040v1)

Authors: Joshua C. Yang, Noemi Scheurer

Public funding processes demand fairness, learning, and outcomes that
participants can understand. We introduce Komitee Equal Shares, a priceable
virtual-budget allocation framework that integrates two signals: in voter mode,
participants cast point votes; in evaluator mode, small groups assess proposals
against collectively defined impact fields. The framework extends the Method of
Equal Shares by translating both signals into virtual spending power and
producing voting receipts. We deployed the framework in the 2025 Kultur Komitee
in Winterthur, Switzerland. Our contributions are: (1) a clear separation of
decision modes, addressing a gap in social choice that typically treats
participatory budgeting as preference aggregation while citizens also see
themselves as evaluators; and (2) the design of voting receipts that
operationalise priceability into participant-facing explanations, making
proportional allocations legible and traceable. The framework generalises to
participatory grant-making and budgeting, offering a model where citizens act
as voters and evaluators within one proportional, explainable allocation.

### 5. [NLP Methods for Detecting Novel LLM Jailbreaks and Keyword Analysis with BERT](http://arxiv.org/pdf/2510.01644v1)

Authors: John Hawkins, Aditya Pramar, Rodney Beard, Rohitash Chandra

Large Language Models (LLMs) suffer from a range of vulnerabilities that
allow malicious users to solicit undesirable responses through manipulation of
the input text. These so-called jailbreak prompts are designed to trick the LLM
into circumventing the safety guardrails put in place to keep responses
acceptable to the developer's policies. In this study, we analyse the ability
of different machine learning models to distinguish jailbreak prompts from
genuine uses, including looking at our ability to identify jailbreaks that use
previously unseen strategies. Our results indicate that using current datasets
the best performance is achieved by fine tuning a Bidirectional Encoder
Representations from Transformers (BERT) model end-to-end for identifying
jailbreaks. We visualise the keywords that distinguish jailbreak from genuine
prompts and conclude that explicit reflexivity in prompt structure could be a
signal of jailbreak intention.

### 6. [Framing Unionization on Facebook: Communication around Representation Elections in the United States](http://arxiv.org/pdf/2510.01757v1)

Authors: Arianna Pera, Veronica Jude, Ceren Budak, Luca Maria Aiello

Digital media have become central to how labor unions communicate, organize,
and sustain collective action. Yet little is known about how unions' online
discourse relates to concrete outcomes such as representation elections. This
study addresses the gap by combining National Labor Relations Board (NLRB)
election data with 158k Facebook posts published by U.S. labor unions between
2015 and 2024. We focused on five discourse frames widely recognized in labor
and social movement communication research: diagnostic (identifying problems),
prognostic (proposing solutions), motivational (mobilizing action), community
(emphasizing solidarity), and engagement (promoting interaction). Using a
fine-tuned RoBERTa classifier, we systematically annotated unions' posts and
analyzed patterns of frame usage around election events. Our findings showed
that diagnostic and community frames dominated union communication overall, but
that frame usage varied substantially across organizations. In election cases
that unions won, communication leading up to the vote showed an increased use
of diagnostic, prognostic, and community frames, followed by a reduction in
prognostic and motivational framing after the event--patterns consistent with
strategic preparation. By contrast, in lost election cases unions showed little
adjustment in their communication, suggesting an absence of tailored
communication strategies. By examining variation in message-level framing, the
study highlights how communication strategies adapt to organizational contexts,
contributing open tools and data and complementing prior research in
understanding digital communication of unions and social movements.

### 7. [Secure Multi-Modal Data Fusion in Federated Digital Health Systems via MCP](http://arxiv.org/pdf/2510.01780v1)

Authors: Aueaphum Aueawatthanaphisut

Secure and interoperable integration of heterogeneous medical data remains a
grand challenge in digital health. Current federated learning (FL) frameworks
offer privacy-preserving model training but lack standardized mechanisms to
orchestrate multi-modal data fusion across distributed and resource-constrained
environments. This study introduces a novel framework that leverages the Model
Context Protocol (MCP) as an interoperability layer for secure, cross-agent
communication in multi-modal federated healthcare systems. The proposed
architecture unifies three pillars: (i) multi-modal feature alignment for
clinical imaging, electronic medical records, and wearable IoT data; (ii)
secure aggregation with differential privacy to protect patient-sensitive
updates; and (iii) energy-aware scheduling to mitigate dropouts in mobile
clients. By employing MCP as a schema-driven interface, the framework enables
adaptive orchestration of AI agents and toolchains while ensuring compliance
with privacy regulations. Experimental evaluation on benchmark datasets and
pilot clinical cohorts demonstrates up to 9.8\% improvement in diagnostic
accuracy compared with baseline FL, a 54\% reduction in client dropout rates,
and clinically acceptable privacy--utility trade-offs. These results highlight
MCP-enabled multi-modal fusion as a scalable and trustworthy pathway toward
equitable, next-generation federated health infrastructures.

### 8. [Just Do It!? Computer-Use Agents Exhibit Blind Goal-Directedness](http://arxiv.org/pdf/2510.01670v1)

Authors: Erfan Shayegani, Keegan Hines, Yue Dong, Nael Abu-Ghazaleh, Roman Lutz, Spencer Whitehead, Vidhisha Balachandran, Besmira Nushi, Vibhav Vineet

Computer-Use Agents (CUAs) are an increasingly deployed class of agents that
take actions on GUIs to accomplish user goals. In this paper, we show that CUAs
consistently exhibit Blind Goal-Directedness (BGD): a bias to pursue goals
regardless of feasibility, safety, reliability, or context. We characterize
three prevalent patterns of BGD: (i) lack of contextual reasoning, (ii)
assumptions and decisions under ambiguity, and (iii) contradictory or
infeasible goals. We develop BLIND-ACT, a benchmark of 90 tasks capturing these
three patterns. Built on OSWorld, BLIND-ACT provides realistic environments and
employs LLM-based judges to evaluate agent behavior, achieving 93.75% agreement
with human annotations. We use BLIND-ACT to evaluate nine frontier models,
including Claude Sonnet and Opus 4, Computer-Use-Preview, and GPT-5, observing
high average BGD rates (80.8%) across them. We show that BGD exposes subtle
risks that arise even when inputs are not directly harmful. While
prompting-based interventions lower BGD levels, substantial risk persists,
highlighting the need for stronger training- or inference-time interventions.
Qualitative analysis reveals observed failure modes: execution-first bias
(focusing on how to act over whether to act), thought-action disconnect
(execution diverging from reasoning), and request-primacy (justifying actions
due to user request). Identifying BGD and introducing BLIND-ACT establishes a
foundation for future research on studying and mitigating this fundamental risk
and ensuring safe CUA deployment.

### Databases

### 1. [Ensemble Threshold Calibration for Stable Sensitivity Control](http://arxiv.org/pdf/2510.02116v1)

Authors: John N. Daras

Precise recall control is critical in large-scale spatial conflation and
entity-matching tasks, where missing even a few true matches can break
downstream analytics, while excessive manual review inflates cost. Classical
confidence-interval cuts such as Clopper-Pearson or Wilson provide lower bounds
on recall, but they routinely overshoot the target by several percentage points
and exhibit high run-to-run variance under skewed score distributions. We
present an end-to-end framework that achieves exact recall with sub-percent
variance over tens of millions of geometry pairs, while remaining TPU-friendly.
Our pipeline starts with an equigrid bounding-box filter and compressed sparse
row (CSR) candidate representation, reducing pair enumeration by two orders of
magnitude. A deterministic xxHash bootstrap sample trains a lightweight neural
ranker; its scores are propagated to all remaining pairs via a single forward
pass and used to construct a reproducible, score-decile-stratified calibration
set. Four complementary threshold estimators - Clopper-Pearson, Jeffreys,
Wilson, and an exact quantile - are aggregated via inverse-variance weighting,
then fused across nine independent subsamples. This ensemble reduces threshold
variance compared to any single method. Evaluated on two real cadastral
datasets (approximately 6.31M and 67.34M pairs), our approach consistently hits
a recall target within a small error, decreases redundant verifications
relative to other calibrations, and runs end-to-end on a single TPU v3 core.

### Distributed, Parallel, and Cluster Computing

### 1. [QScale: Probabilistic Chained Consensus for Moderate-Scale Systems](http://arxiv.org/pdf/2510.01536v1)

Authors: Hasan Heydari, Alysson Bessani, Kartik Nayak

Existing distributed ledger protocols either incur a high communication
complexity and are thus suited to systems with a small number of processes
(e.g., PBFT), or rely on committee-sampling-based approaches that only work for
a very large number of processes (e.g., Algorand). Neither of these lines of
work is well-suited for moderate-scale distributed ledgers ranging from a few
hundred to a thousand processes, which are common in production (e.g, Redbelly,
Sui). The goal of this work is to design a distributed ledger with sub-linear
communication complexity per process, sub-quadratic total communication
complexity, and low latency for finalizing a block into the ledger, such that
it can be used for moderate-scale systems. We propose QScale, a protocol in
which every process incurs only $\widetilde{O}(\kappa \sqrt{n})$ communication
complexity per-block in expectation, $\widetilde{O}(n\kappa)$ total
communication complexity per-block in expectation, and a best-case latency of
$O(\kappa)$ rounds while ensuring safety and liveness with overwhelming
probability, with $\kappa$ being a small security parameter.

### 2. [TetriServe: Efficient DiT Serving for Heterogeneous Image Generation](http://arxiv.org/pdf/2510.01565v1)

Authors: Runyu Lu, Shiqi He, Wenxuan Tan, Shenggui Li, Ruofan Wu, Jeff J. Ma, Ang Chen, Mosharaf Chowdhury

Diffusion Transformer (DiT) models excel at generating highquality images
through iterative denoising steps, but serving them under strict Service Level
Objectives (SLOs) is challenging due to their high computational cost,
particularly at large resolutions. Existing serving systems use fixed degree
sequence parallelism, which is inefficient for heterogeneous workloads with
mixed resolutions and deadlines, leading to poor GPU utilization and low SLO
attainment.
  In this paper, we propose step-level sequence parallelism to dynamically
adjust the parallel degree of individual requests according to their deadlines.
We present TetriServe, a DiT serving system that implements this strategy for
highly efficient image generation. Specifically, TetriServe introduces a novel
round-based scheduling mechanism that improves SLO attainment: (1) discretizing
time into fixed rounds to make deadline-aware scheduling tractable, (2)
adapting parallelism at the step level and minimize GPU hour consumption, and
(3) jointly packing requests to minimize late completions. Extensive evaluation
on state-of-the-art DiT models shows that TetriServe achieves up to 32% higher
SLO attainment compared to existing solutions without degrading image quality.

### 3. [Exponential Quantum Advantage for Message Complexity in Distributed Algorithms](http://arxiv.org/pdf/2510.01657v1)

Authors: François Le Gall, Maël Luce, Joseph Marchand, Mathieu Roget

We investigate how much quantum distributed algorithms can outperform
classical distributed algorithms with respect to the message complexity (the
overall amount of communication used by the algorithm). Recently, Dufoulon,
Magniez and Pandurangan (PODC 2025) have shown a polynomial quantum advantage
for several tasks such as leader election and agreement. In this paper, we show
an exponential quantum advantage for a fundamental task: routing information
between two specified nodes of a network. We prove that for the family of
``welded trees" introduced in the seminal work by Childs, Cleve, Deotto, Farhi,
Gutmann and Spielman (STOC 2003), there exists a quantum distributed algorithm
that transfers messages from the entrance of the graph to the exit with message
complexity exponentially smaller than any classical algorithm. Our quantum
algorithm is based on the recent "succinct" implementation of quantum walks
over the welded trees by Li, Li and Luo (SODA 2024). Our classical lower bound
is obtained by ``lifting'' the lower bound from Childs, Cleve, Deotto, Farhi,
Gutmann and Spielman (STOC 2003) from query complexity to message complexity.

### 4. [Accuracy vs Performance: An abstraction model for deadline constrained offloading at the mobile-edge](http://arxiv.org/pdf/2510.01885v1)

Authors: Jamie Cotter, Ignacio Castineiras, Victor Cionca

In this paper, we present a solution for low-latency deadline-constrained DNN
offloading on mobile edge devices. We design a scheduling algorithm with
lightweight network state representation, considering device availability,
communication on the network link, priority-aware pre-emption, and task
deadlines. The scheduling algorithm aims to reduce latency by designing a
resource availability representation, as well as a network discretisation and a
dynamic bandwidth estimation mechanism. We implement the scheduling algorithm
into a system composed of four Raspberry Pi 2 (model Bs) mobile edge devices,
sampling a waste classification conveyor belt at a set frame rate. The system
is evaluated and compared to a previous approach of ours, which was proven to
outcompete work-stealers and a non-pre-emption based scheduling heuristic under
the aforementioned waste classification scenario. Our findings show the novel
lower latency abstraction models yield better performance under high-volume
workloads, with the dynamic bandwidth estimation assisting the task placement
while, ultimately, increasing task throughput in times of resource scarcity.

### Digital Libraries

### 1. [Investigating Industry--Academia Collaboration in Artificial Intelligence: PDF-Based Bibliometric Analysis from Leading Conferences](http://arxiv.org/pdf/2510.01593v1)

Authors: Kazuhiro Yamauchi, Marie Katsurai

This study presents a bibliometric analysis of industry--academia
collaboration in artificial intelligence (AI) research, focusing on papers from
two major international conferences, AAAI and IJCAI, from 2010 to 2023. Most
previous studies have relied on publishers and other databases to analyze
bibliographic information. However, these databases have problems, such as
missing articles and omitted metadata. Therefore, we adopted a novel approach
to extract bibliographic information directly from the article PDFs: we
examined 20,549 articles and identified the collaborative papers through a
classification process of author affiliation. The analysis explores the
temporal evolution of collaboration in AI, highlighting significant changes in
collaboration patterns over the past decade. In particular, this study examines
the role of key academic and industrial institutions in facilitating these
collaborations, focusing on emerging global trends. Additionally, a content
analysis using document classification was conducted to examine the type of
first author in collaborative research articles and explore the potential
differences between collaborative and noncollaborative research articles. The
results showed that, in terms of publication, collaborations are mainly led by
academia, but their content is not significantly different from that of others.
The affiliation metadata are available at
https://github.com/mm-doshisha/ICADL2024.

### 2. [PreprintToPaper dataset: connecting bioRxiv preprints with journal publications](http://arxiv.org/pdf/2510.01783v1)

Authors: Fidan Badalova, Julian Sienkiewicz, Philipp Mayr

The PreprintToPaper dataset connects bioRxiv preprints with their
corresponding journal publications, enabling large-scale analysis of the
preprint-to-publication process. It comprises metadata for 145,517 preprints
from two periods, 2016-2018 (pre-pandemic) and 2020-2022 (pandemic), retrieved
via the bioRxiv and Crossref APIs. Each record includes bibliographic
information such as titles, abstracts, authors, institutions, submission dates,
licenses, and subject categories, alongside enriched publication metadata
including journal names, publication dates, author lists, and further
information. Preprints are categorized into three groups: Published (formally
linked to a journal article), Preprint Only (unpublished), and Gray Zone
(potentially published but unlinked). To enhance reliability, title and author
similarity scores were calculated, and a human-annotated subset of 299 records
was created for evaluation of Gray Zone cases. The dataset supports diverse
applications, including studies of scholarly communication, open science
policies, bibliometric tool development, and natural language processing
research on textual changes between preprints and their published versions. The
dataset is publicly available in CSV format via Zenodo.

### 3. [KTBox: A Modular LaTeX Framework for Semantic Color, Structured Highlighting, and Scholarly Communication](http://arxiv.org/pdf/2510.01961v1)

Authors: Bhaskar Mangal, Ashutosh Bhatia, Yashvardhan Sharma, Kamlesh Tiwari, Rashmi Verma

The communication of technical insight in scientific manuscripts often relies
on ad-hoc formatting choices, resulting in inconsistent visual emphasis and
limited portability across document classes. This paper introduces ktbox, a
modular LaTeX framework that unifies semantic color palettes, structured
highlight boxes, taxonomy trees, and author metadata utilities into a coherent
system for scholarly writing. The framework is distributed as a set of
lightweight, namespaced components: ktcolor.sty for semantic palettes,
ktbox.sty for structured highlight and takeaway environments, ktlrtree.sty for
taxonomy trees with fusion and auxiliary annotations, and ktorcid.sty for
ORCID-linked author metadata. Each component is independently usable yet
interoperable, ensuring compatibility with major templates such as IEEEtran,
acmart, iclr conference, and beamer. Key features include auto-numbered
takeaway boxes, wide-format highlights, flexible taxonomy tree visualizations,
and multi-column layouts supporting embedded tables, enumerations, and code
blocks. By adopting a clear separation of concerns and enforcing a consistent
naming convention under the kt namespace, the framework transforms visual
styling from cosmetic add-ons into reproducible, extensible building blocks of
scientific communication, improving clarity, portability, and authoring
efficiency across articles, posters, and presentations.

### 4. [How to Find Fantastic Papers: Self-Rankings as a Powerful Predictor of Scientific Impact Beyond Peer Review](http://arxiv.org/pdf/2510.02143v1)

Authors: Buxin Su, Natalie Collina, Garrett Wen, Didong Li, Kyunghyun Cho, Jianqing Fan, Bingxin Zhao, Weijie Su

Peer review in academic research aims not only to ensure factual correctness
but also to identify work of high scientific potential that can shape future
research directions. This task is especially critical in fast-moving fields
such as artificial intelligence (AI), yet it has become increasingly difficult
given the rapid growth of submissions. In this paper, we investigate an
underexplored measure for identifying high-impact research: authors' own
rankings of their multiple submissions to the same AI conference. Grounded in
game-theoretic reasoning, we hypothesize that self-rankings are informative
because authors possess unique understanding of their work's conceptual depth
and long-term promise. To test this hypothesis, we conducted a large-scale
experiment at a leading AI conference, where 1,342 researchers self-ranked
their 2,592 submissions by perceived quality. Tracking outcomes over more than
a year, we found that papers ranked highest by their authors received twice as
many citations as their lowest-ranked counterparts; self-rankings were
especially effective at identifying highly cited papers (those with over 150
citations). Moreover, we showed that self-rankings outperformed peer review
scores in predicting future citation counts. Our results remained robust after
accounting for confounders such as preprint posting time and self-citations.
Together, these findings demonstrate that authors' self-rankings provide a
reliable and valuable complement to peer review for identifying and elevating
high-impact research in AI.

### Discrete Mathematics

### 1. [Computing Phylogenetic Diversity](http://arxiv.org/pdf/2510.01849v1)

Authors: Jannik Schestag

Phylogenetic Diversity(PD)is a well-regarded measure of the overall
biodiversity of a set of present-day species(taxa)that indicates its ecological
significance.In the Maximize Phylogenetic Diversity(Max-PD)problem one is asked
to find a small set of taxa in a phylogenetic tree for which this measure is
maximized.Max-PD is particularly relevant in conservation planning,where
limited resources necessitate prioritizing certain taxa to minimize
biodiversity loss.Although Max-PD can be solved in polynomial time
[Steel,SB,2005;Pardi&Goldman,PLoS,2005],its generalizations-which aim to model
biological processes and other aspects in conservation planning with greater
accuracy-often exhibit NP-hardness,making them computationally challenging.This
thesis explores a selection of these generalized problems within the framework
of parameterized complexity. In Generalized Noah's Ark Problem(GNAP),each taxon
only survives at a certain survival probability,which can be increased by
investing more money in the taxon.We show that GNAP is W[1]-hard with respect
to the number of taxa but is XP with respect to the number of different costs
and different survival probabilities. Additionally,we show that unit-cost-NAP,a
special case of GNAP,is NP-hard. In Time Sensitive Maximization of Phylogenetic
Diversity(Time-PD),different extinction times of taxa are considered after
which they can no longer be saved.For Time-PD,we present color-coding
algorithms that prove that Time-PD is fixed-parameter tractable(FPT)with
respect to the threshold of diversity and the acceptable loss of diversity. In
Optimizing PD with Dependencies(PDD),each saved taxon must be a source in the
ecological system or a predator of another saved species.These dependencies are
given in a food-web.We show that PDD is FPT when parameterized with the size of
the solution plus the height of the phylogenetic tree. Further,we consider
pa...

### 2. [On cuts of small chromatic number in sparse graphs](http://arxiv.org/pdf/2510.01791v1)

Authors: Guillaume Aubian, Marthe Bonamy, Romain Bourneuf, Oscar Fontaine, Lucas Picasarri-Arrieta

For a given integer $k$, let $\ell_k$ denote the supremum $\ell$ such that
every sufficiently large graph $G$ with average degree less than $2\ell$ admits
a separator $X \subseteq V(G)$ for which $\chi(G[X]) < k$. Motivated by the
values of $\ell_1$, $\ell_2$ and $\ell_3$, a natural conjecture suggests that
$\ell_k = k$ for all $k$. We prove that this conjecture fails dramatically:
asymptotically, the trivial lower bound $\ell_k \geq \tfrac{k}{2}$ is tight.
More precisely, we prove that for every $\varepsilon>0$ and all sufficiently
large $k$, we have $\ell_k \leq (1+\varepsilon)\tfrac{k}{2}$.

### Data Structures and Algorithms

### 1. [Foremost, Fastest, Shortest: Temporal Graph Realization under Various Path Metrics](http://arxiv.org/pdf/2510.01702v1)

Authors: Justine Cauvi, Nils Morawietz, Laurent Viennot

In this work, we follow the current trend on temporal graph realization,
where one is given a property P and the goal is to determine whether there is a
temporal graph, that is, a graph where the edge set changes over time, with
property P . We consider the problems where as property P , we are given a
prescribed matrix for the duration, length, or earliest arrival time of
pairwise temporal paths. That is, we are given a matrix D and ask whether there
is a temporal graph such that for any ordered pair of vertices (s, t), Ds,t
equals the duration (length, or earliest arrival time, respectively) of any
temporal path from s to t minimizing that specific temporal path metric. For
shortest and earliest arrival temporal paths, we are the first to consider
these problems as far as we know. We analyze these problems for many settings
like: strict and non-strict paths, periodic and non-periodic temporal graphs,
and limited number of labels per edge (that is, limited occurrence number per
edge over time). In contrast to all other path metrics, we show that for the
earliest arrival times, we can achieve polynomial-time algorithms in periodic
and non-periodic temporal graphs and for strict and and non-strict paths.
However, the problem becomes NP-hard when the matrix does not contain a single
integer but a set or range of possible allowed values. As we show, the problem
can still be solved efficiently in this scenario, when the number of entries
with more than one value is small, that is, we develop an FPT-algorithm for the
number of such entries. For the setting of fastest paths, we achieve new
hardness results that answers an open question by Klobas, Mertzios, Molter, and
Spirakis [Theor. Comput. Sci. '25] about the parameterized complexity of the
problem with respect to the vertex cover number and significantly improves over
a previous hardness result for the feedback vertex set number. When considering
shortest paths, we show that the periodic versions are polynomial-time solvable
whereas the non-periodic versions become NP-hard.

### 2. [Improved $\ell_{p}$ Regression via Iteratively Reweighted Least Squares](http://arxiv.org/pdf/2510.01729v1)

Authors: Alina Ene, Ta Duy Nguyen, Adrian Vladu

We introduce fast algorithms for solving $\ell_{p}$ regression problems using
the iteratively reweighted least squares (IRLS) method. Our approach achieves
state-of-the-art iteration complexity, outperforming the IRLS algorithm by
Adil-Peng-Sachdeva (NeurIPS 2019) and matching the theoretical bounds
established by the complex algorithm of Adil-Kyng-Peng-Sachdeva (SODA 2019, J.
ACM 2024) via a simpler lightweight iterative scheme. This bridges the existing
gap between theoretical and practical algorithms for $\ell_{p}$ regression. Our
algorithms depart from prior approaches, using a primal-dual framework, in
which the update rule can be naturally derived from an invariant maintained for
the dual objective. Empirically, we show that our algorithms significantly
outperform both the IRLS algorithm by Adil-Peng-Sachdeva and MATLAB/CVX
implementations.

### 3. [Short circuit walks in fixed dimension](http://arxiv.org/pdf/2510.01916v1)

Authors: Alexander E. Black, Christian Nöbel, Raphael Steiner

Circuit augmentation schemes are a family of combinatorial algorithms for
linear programming that generalize the simplex method. To solve the linear
program, they construct a so-called monotone circuit walk: They start at an
initial vertex of the feasible region and traverse a discrete sequence of
points on the boundary, while moving along certain allowed directions
(circuits) and improving the objective function at each step until reaching an
optimum. Since the existence of short circuit walks has been conjectured
(Circuit Diameter Conjecture), several works have investigated how well one can
efficiently approximate shortest monotone circuit walks towards an optimum. A
first result addressing this question was given by De Loera, Kafer, and
Sanit\`a [SIAM J. Opt., 2022], who showed that given as input an LP and the
starting vertex, finding a $2$-approximation for this problem is NP-hard.
Cardinal and the third author [Math. Prog. 2023] gave a stronger lower bound
assuming the exponential time hypothesis, showing that even an approximation
factor of $O(\frac{\log m}{\log \log m})$ is intractable for LPs defined by $m$
inequalities. Both of these results were based on reductions from highly
degenerate polytopes in combinatorial optimization with high dimension.
  In this paper, we significantly strengthen the aforementioned hardness
results by showing that for every fixed $\varepsilon>0$ approximating the
problem on polygons with $m$ edges to within a factor of $O(m^{1-\varepsilon})$
is NP-hard. This result is essentially best-possible, as it cannot be improved
beyond $o(m)$. In particular, this implies hardness for simple polytopes and in
fixed dimension.

### 4. [Bifurcation: How to Explore a Tree](http://arxiv.org/pdf/2510.01939v1)

Authors: Sariel Har-Peled

Avraham et al. [AFK+15] presented an alternative approach to parametric
search, called \emph{bifurcation}, that performs faster under certain
circumstances. Intuitively, when the underlying decider execution can be rolled
back cheaply and the decider has a near-linear running time. For some problems,
this leads to fast algorithms that beat the seemingly natural lower bound
arising from distance selection.
  Bifurcation boils down to a tree exploration problem. You are given a binary
(unfortunately implicit) tree of height $n$ and $k$ internal nodes with two
children (all other internal nodes have a single child), and assume each node
has an associated parameter value. These values are sorted in the inorder
traversal of the tree. Assume there is (say) a node (not necessarily a leaf)
that is the target node that the exploration needs to discover.
  The player starts from the root. At each step, the player can move to
adjacent nodes to the current location (i.e., one of the children or the
parent). Alternatively, the player can call an oracle on the current node,
which returns either that it is the target (thus, mission accomplished!) or
whether the target value is strictly smaller or larger than the current one.
  A naive algorithm explores the whole tree, in $O(n k)$ time, then performs
$O(\log k n)$ calls to the oracle to find the desired leaf. Avraham \etal
showed that this can be improved to $O(n \sqrt{k} )$ time, and $O( \sqrt{k}
\log n)$ oracle calls.
  Here, we improve this to $O(n \sqrt{k} )$ time, with only $ O( \sqrt{k} +
\log n)$ oracle calls. We also show matching lower bounds, under certain
assumptions. We believe our interpretation of bifurcation as a tree exploration
problem, and the associated algorithm, are of independent interest.

### Emerging Technologies

### 1. [ENLighten: Lighten the Transformer, Enable Efficient Optical Acceleration](http://arxiv.org/pdf/2510.01673v1)

Authors: Hanqing Zhu, Zhican Zhou, Shupeng Ning, Xuhao Wu, Ray Chen, Yating Wan, David Pan

Photonic computing has emerged as a promising substrate for accelerating the
dense linear-algebra operations at the heart of AI, yet adoption for large
Transformer models remains in its infancy. We identify two bottlenecks: (1)
costly electro--optic conversions and data-movement overheads that erode energy
efficiency as model sizes scale; (2) a mismatch between limited on-chip
photonic resources and Transformer scale, which forces frequent reuse of
photonic tensor cores and dilutes throughput gains. To address these
challenges, we introduce a hardware--software co-design framework. First, we
propose \texttt{Lighten}, a PTC-aware compression flow that post-hoc decomposes
each Transformer weight matrix into a low-rank component plus a
structured-sparse component aligned to photonic tensor-core granularity,
without lengthy retraining. Second, we present \texttt{ENLighten}, a
reconfigurable photonic accelerator with dynamically adaptive tensor cores,
driven by broadband light redistribution, enabling fine-grained sparsity
support and full power gating of inactive parts. On ImageNet, \texttt{Lighten}
prunes a Base-scale Vision Transformer by 50\% with $\approx$1\% accuracy drop
after only 3 epochs (about 1 hour) of fine-tuning. Deployed on
\texttt{ENLighten}, it achieves a $2.5\times$ improvement in energy--delay
product over the state-of-the-art photonic Transformer accelerator.

### 2. [Zero-shot reasoning for simulating scholarly peer-review](http://arxiv.org/pdf/2510.02027v1)

Authors: Khalid M. Saqr

The scholarly publishing ecosystem faces a dual crisis of unmanageable
submission volumes and unregulated AI, creating an urgent need for new
governance models to safeguard scientific integrity. The traditional human-only
peer review regime lacks a scalable, objective benchmark, making editorial
processes opaque and difficult to audit. Here we investigate a deterministic
simulation framework that provides the first stable, evidence-based standard
for evaluating AI-generated peer review reports. Analyzing 352 peer-review
simulation reports, we identify consistent system state indicators that
demonstrate its reliability. First, the system is able to simulate calibrated
editorial judgment, with 'Revise' decisions consistently forming the majority
outcome (>50%) across all disciplines, while 'Reject' rates dynamically adapt
to field-specific norms, rising to 45% in Health Sciences. Second, it maintains
unwavering procedural integrity, enforcing a stable 29% evidence-anchoring
compliance rate that remains invariant across diverse review tasks and
scientific domains. These findings demonstrate a system that is predictably
rule-bound, mitigating the stochasticity of generative AI. For the scientific
community, this provides a transparent tool to ensure fairness; for publishing
strategists, it offers a scalable instrument for auditing workflows, managing
integrity risks, and implementing evidence-based governance. The framework
repositions AI as an essential component of institutional accountability,
providing the critical infrastructure to maintain trust in scholarly
communication.

### Graphics

### 1. [MIRAGE: Patient-Specific Mixed Reality Coaching for MRI via Depth-Only Markerless Registration and Immersive VR](http://arxiv.org/pdf/2510.01743v1)

Authors: Daniel Brooks, Emily Carter, Hu Guo, Rajesh Nair

Magnetic resonance imaging (MRI) is an indispensable diagnostic tool, yet the
confined bore and acoustic noise can evoke considerable anxiety and
claustrophobic reactions. High anxiety leads to motion artifacts, incomplete
scans and reliance on pharmacological sedation. MIRAGE (Mixed Reality Anxiety
Guidance Environment) harnesses the latest mixed reality (MR) hardware to
prepare patients for MRI through immersive virtual reality (VR) and markerless
augmented reality (AR) registration. In this paper, we extend our previous work
by providing a comprehensive review of related research, detailing the system
architecture, and exploring metrics for patient and clinician experience. We
also present considerations for clinical deployment of MR systems within
hospital workflows. Our results indicate that depth-based registration achieves
sub-centimeter accuracy with minimal setup, while the immersive coaching
environment reduces patient anxiety and yields favourable usability scores.

### 2. [MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](http://arxiv.org/pdf/2510.01619v1)

Authors: Changmin Lee, Jihyun Lee, Tae-Kyun Kim

While there has been significant progress in the field of 3D avatar creation
from visual observations, modeling physically plausible dynamics of humans with
loose garments remains a challenging problem. Although a few existing works
address this problem by leveraging physical simulation, they suffer from
limited accuracy or robustness to novel animation inputs. In this work, we
present MPMAvatar, a framework for creating 3D human avatars from multi-view
videos that supports highly realistic, robust animation, as well as
photorealistic rendering from free viewpoints. For accurate and robust dynamics
modeling, our key idea is to use a Material Point Method-based simulator, which
we carefully tailor to model garments with complex deformations and contact
with the underlying body by incorporating an anisotropic constitutive model and
a novel collision handling algorithm. We combine this dynamics modeling scheme
with our canonical avatar that can be rendered using 3D Gaussian Splatting with
quasi-shadowing, enabling high-fidelity rendering for physically realistic
animations. In our experiments, we demonstrate that MPMAvatar significantly
outperforms the existing state-of-the-art physics-based avatar in terms of (1)
dynamics modeling accuracy, (2) rendering accuracy, and (3) robustness and
efficiency. Additionally, we present a novel application in which our avatar
generalizes to unseen interactions in a zero-shot manner-which was not
achievable with previous learning-based methods due to their limited simulation
generalizability. Our project page is at:
https://KAISTChangmin.github.io/MPMAvatar/

### 3. [Multimodal Feedback for Task Guidance in Augmented Reality](http://arxiv.org/pdf/2510.01690v1)

Authors: Hu Guo, Lily Patel, Rohan Gupt

Optical see-through augmented reality (OST-AR) overlays digital targets and
annotations on the physical world, offering promising guidance for hands-on
tasks such as medical needle insertion or assembly. Recent work on OST-AR depth
perception shows that target opacity and tool visualization significantly
affect accuracy and usability; opaque targets and rendering the real instrument
reduce depth errors, whereas transparent targets and absent tools impair
performance. However, reliance on visual overlays may overload attention and
leaves little room for depth cues when occlusion or lighting hampers
perception. To address these limitations, we explore multimodal feedback that
combines OST-AR with wrist-based vibrotactile haptics. The past two years have
seen rapid advances in haptic technology. Researchers have investigated
skin-stretch and vibrotactile cues for conveying spatial information to blind
users, wearable ring actuators that support precise pinching in AR, cross-modal
audio-haptic cursors that enable eyes-free object selection, and wrist-worn
feedback for teleoperated surgery that improves force awareness at the cost of
longer task times. Studies comparing pull versus push vibrotactile metaphors
found that pull cues yield faster gesture completion and lower cognitive load.
These findings motivate revisiting OST-AR guidance with a fresh perspective on
wrist-based haptics. We design a custom wristband with six vibromotors
delivering directional and state cues, integrate it with a handheld tool and
OST-AR, and assess its impact on cue recognition and depth guidance. Through a
formative study and two experiments (N=21 and N=27), we show that participants
accurately identify haptic patterns under cognitive load and that multimodal
feedback improves spatial precision and usability compared with visual-only or
haptic-only conditions.

### 4. [ROI-GS: Interest-based Local Quality 3D Gaussian Splatting](http://arxiv.org/pdf/2510.01978v1)

Authors: Quoc-Anh Bui, Gilles Rougeron, Géraldine Morin, Simone Gasparini

We tackle the challenge of efficiently reconstructing 3D scenes with high
detail on objects of interest. Existing 3D Gaussian Splatting (3DGS) methods
allocate resources uniformly across the scene, limiting fine detail to Regions
Of Interest (ROIs) and leading to inflated model size. We propose ROI-GS, an
object-aware framework that enhances local details through object-guided camera
selection, targeted Object training, and seamless integration of high-fidelity
object of interest reconstructions into the global scene. Our method
prioritizes higher resolution details on chosen objects while maintaining
real-time performance. Experiments show that ROI-GS significantly improves
local quality (up to 2.96 dB PSNR), while reducing overall model size by
$\approx 17\%$ of baseline and achieving faster training for a scene with a
single object of interest, outperforming existing methods.

### 5. [Spec-Gloss Surfels and Normal-Diffuse Priors for Relightable Glossy Objects](http://arxiv.org/pdf/2510.02069v1)

Authors: Georgios Kouros, Minye Wu, Tinne Tuytelaars

Accurate reconstruction and relighting of glossy objects remain a
longstanding challenge, as object shape, material properties, and illumination
are inherently difficult to disentangle. Existing neural rendering approaches
often rely on simplified BRDF models or parameterizations that couple diffuse
and specular components, which restricts faithful material recovery and limits
relighting fidelity. We propose a relightable framework that integrates a
microfacet BRDF with the specular-glossiness parameterization into 2D Gaussian
Splatting with deferred shading. This formulation enables more physically
consistent material decomposition, while diffusion-based priors for surface
normals and diffuse color guide early-stage optimization and mitigate
ambiguity. A coarse-to-fine optimization of the environment map accelerates
convergence and preserves high-dynamic-range specular reflections. Extensive
experiments on complex, glossy scenes demonstrate that our method achieves
high-quality geometry and material reconstruction, delivering substantially
more realistic and consistent relighting under novel illumination compared to
existing Gaussian splatting methods.

### Computer Science and Game Theory

### 1. [Incentive Analysis of Collusion in Fair Division](http://arxiv.org/pdf/2510.01689v1)

Authors: Haoqiang Huang, Biaoshuai Tao, Mingwei Yang, Shengwei Zhou

We study fair division problems with strategic agents capable of gaining
advantages by manipulating their reported preferences. Although several
impossibility results have revealed the incompatibility of truthfulness with
standard fairness criteria, subsequent works have circumvented this limitation
through the incentive ratio framework. Previous studies demonstrate that
fundamental mechanisms like Maximum Nash Welfare (MNW) and Probabilistic Serial
(PS) for divisible goods, and Round-Robin (RR) for indivisible goods achieve an
incentive ratio of $2$, implying that no individual agent can gain more than
double his truthful utility through manipulation. However, collusive
manipulation by agent groups remains unexplored.
  In this work, we define strong group incentive ratio (SGIR) and group
incentive ratio (GIR) to measure the gain of collusive manipulation, where SGIR
and GIR are respectively the maximum and minimum of the incentive ratios of
corrupted agents. Then, we tightly characterize the SGIRs and GIRs of MNW, PS,
and RR. In particular, the GIR of MNW is $2$ regardless of the coalition size.
Moreover, for coalition size $c \geq 1$, the SGIRs of MNW and PS, and the GIRs
of PS and RR are $c + 1$. Finally, the SGIR of RR is unbounded for coalition
size $c \geq 2$. Our results reveal fundamental differences of these three
mechanisms in their vulnerability to collusion.

### 2. [Multi-group Bayesian Games](http://arxiv.org/pdf/2510.02078v1)

Authors: Hongxing Yuan, Xuan Zhang, Chunyu Wei, Yushun Fan

This paper presents a model of multi-group Bayesian games (MBGs) to describe
the group behavior in Bayesian games, and gives methods to find (strongly)
multi-group Bayesian Nash equilibria (MBNE) of this model with a proposed
transformation. MBNE represent the optimal strategy \textit{profiles} under the
situation where players within a group play a cooperative game, while strongly
MBNE characterize the optimal strategy \textit{profiles} under the situation
where players within a group play a noncooperative game. Firstly, we propose a
model of MBGs and give a transformation to convert any MBG into a multi-group
ex-ante agent game (MEAG) which is a normal-form game. Secondly, we give a
sufficient and necessary condition for a MBG's MEAG to be (strongly) potential.
If it is (strongly) potential, all its (strongly) Nash equilibria can be found,
and then all (strongly) MBNE of the MBG can be obtained by leveraging the
transformation's good properties. Finally, we provide algorithms for finding
(strongly) MBNE of a MBG whose MEAG is (strongly) potential and use an
illustrative example to verify the correctness of our results.

### 3. [A Linear Programming Approach to Estimate the Core in Cooperative Games](http://arxiv.org/pdf/2510.01766v1)

Authors: J Camacho, JC Gonçalves-Dosantos, J Sánchez-Soriano

This paper proposes a novel algorithm to approximate the core of transferable
utility (TU) cooperative games via linear programming. Given the computational
hardness of determining the full core, our approach provides a tractable
approximation by sampling extreme points through randomized linear problems
(LPs). We analyze its convergence and computational complexity, and validate
its effectiveness through extensive simulations on various game models. Our
results show that the method is scalable and achieves high accuracy in terms of
core reconstruction.

### Human-Computer Interaction

### 1. [Dialogues with AI Reduce Beliefs in Misinformation but Build No Lasting Discernment Skills](http://arxiv.org/pdf/2510.01537v1)

Authors: Anku Rani, Valdemar Danry, Paul Pu Liang, Andrew B. Lippman, Pattie Maes

Given the growing prevalence of fake information, including increasingly
realistic AI-generated news, there is an urgent need to train people to better
evaluate and detect misinformation. While interactions with AI have been shown
to durably reduce people's beliefs in false information, it is unclear whether
these interactions also teach people the skills to discern false information
themselves. We conducted a month-long study where 67 participants classified
news headline-image pairs as real or fake, discussed their assessments with an
AI system, followed by an unassisted evaluation of unseen news items to measure
accuracy before, during, and after AI assistance. While AI assistance produced
immediate improvements during AI-assisted sessions (+21\% average),
participants' unassisted performance on new items declined significantly by
week 4 (-15.3\%). These results indicate that while AI may help immediately, it
ultimately degrades long-term misinformation detection abilities.

### 2. [TimeGazer: Temporal Modeling of Predictive Gaze Stabilization for AR Interaction](http://arxiv.org/pdf/2510.01561v1)

Authors: Yaozheng Xia, Zaiping Zhu, Bo Pang, Shaorong Wang, Sheng Li

Gaze stabilization is critical for enabling fluid, accurate, and efficient
interaction in immersive augmented reality (AR) environments, particularly
during task-oriented visual behaviors. However, fixation sequences captured in
active gaze tasks often exhibit irregular dispersion and systematic deviations
from target locations, a variability primarily caused by the combined effects
of human oculomotor physiology, insufficient AR headset tracking and
calibration accuracy, and environmental disturbances, undermining interaction
performance and visual engagement. To address this issue, we propose TimeGazer,
which reformulates gaze stabilization as a sequence-to-sequence temporal
regression problem, predicting idealized fixation trajectories for the
target-fixation phase from historical gaze dynamics in the search phase. We
present a synthetic data generation and blending strategy that produces
spatially concentrated, target-centered fixation references aligned with task
objectives, substantially enriching the training space and enhancing model
generalization. We train and evaluate TimeGazer on a hybrid dataset of real and
augmented gaze sequences collected via Microsoft HoloLens 2 from 54
participants across multiple prediction horizons. Through the user study,
statistical results demonstrate that TimeGazer significantly improves
interaction accuracy and reduces completion time, confirming that temporal
modeling of predictive gaze stabilization can strengthen attentional
consistency and responsiveness in task-driven AR interaction. These findings
highlight the broader potential of TimeGazer for advancing adaptive gaze-based
interfaces and temporal modeling research in immersive systems.

### 3. [Who is responsible? Social Identity, Robot Errors and Blame Attribution](http://arxiv.org/pdf/2510.01862v1)

Authors: Samantha Stedtler, Marianna Leventi

This paper argues that conventional blame practices fall short of capturing
the complexity of moral experiences, neglecting power dynamics and
discriminatory social practices. It is evident that robots, embodying roles
linked to specific social groups, pose a risk of reinforcing stereotypes of how
these groups behave or should behave, so they set a normative and descriptive
standard. In addition, we argue that faulty robots might create expectations of
who is supposed to compensate and repair after their errors, where social
groups that are already disadvantaged might be blamed disproportionately if
they do not act according to their ascribed roles. This theoretical and
empirical gap becomes even more urgent to address as there have been
indications of potential carryover effects from Human-Robot Interactions (HRI)
to Human-Human Interactions (HHI). We therefore urge roboticists and designers
to stay in an ongoing conversation about how social traits are conceptualised
and implemented in this technology. We also argue that one solution could be to
'embrace the glitch' and to focus on constructively disrupting practices
instead of prioritizing efficiency and smoothness of interaction above
everything else. Apart from considering ethical aspects in the design phase of
social robots, we see our analysis as a call for more research on the
consequences of robot stereotyping and blame attribution.

### 4. [Agentic Reasoning and Refinement through Semantic Interaction](http://arxiv.org/pdf/2510.02157v1)

Authors: Xuxin Tang, Rehema Abulikemu, Eric Krokos, Kirsten Whitley, Xuan Wang, Chris North

Sensemaking report writing often requires multiple refinements in the
iterative process. While Large Language Models (LLMs) have shown promise in
generating initial reports based on human visual workspace representations,
they struggle to precisely incorporate sequential semantic interactions during
the refinement process. We introduce VIS-ReAct, a framework that reasons about
newly-added semantic interactions in visual workspaces to steer the LLM for
report refinement.
  VIS-ReAct is a two-agent framework: a primary LLM analysis agent interprets
new semantic interactions to infer user intentions and generate refinement
planning, followed by an LLM refinement agent that updates reports accordingly.
Through case study, VIS-ReAct outperforms baseline and VIS-ReAct (without LLM
analysis) on targeted refinement, semantic fidelity, and transparent inference.
Results demonstrate that VIS-ReAct better handles various interaction types and
granularities while enhancing the transparency of human-LLM collaboration.

### 5. [Towards Human-Centered RegTech: Unpacking Professionals' Strategies and Needs for Using LLMs Safely](http://arxiv.org/pdf/2510.01638v1)

Authors: Siying Hu, Yaxing Yao, Zhicong Lu

Large Language Models are profoundly changing work patterns in high-risk
professional domains, yet their application also introduces severe and
underexplored compliance risks. To investigate this issue, we conducted
semi-structured interviews with 24 highly-skilled knowledge workers from
industries such as law, healthcare, and finance. The study found that these
experts are commonly concerned about sensitive information leakage,
intellectual property infringement, and uncertainty regarding the quality of
model outputs. In response, they spontaneously adopt various mitigation
strategies, such as actively distorting input data and limiting the details in
their prompts. However, the effectiveness of these spontaneous efforts is
limited due to a lack of specific compliance guidance and training for Large
Language Models. Our research reveals a significant gap between current NLP
tools and the actual compliance needs of experts. This paper positions these
valuable empirical findings as foundational work for building the next
generation of Human-Centered, Compliance-Driven Natural Language Processing for
Regulatory Technology (RegTech), providing a critical human-centered
perspective and design requirements for engineering NLP systems that can
proactively support expert compliance workflows.

### 6. [A Locally Executable AI System for Improving Preoperative Patient Communication: A Multi-Domain Clinical Evaluation](http://arxiv.org/pdf/2510.01671v1)

Authors: Motoki Sato, Yuki Matsushita, Hidekazu Takahashi, Tomoaki Kakazu, Sou Nagata, Mizuho Ohnuma, Atsushi Yoshikawa, Masayuki Yamamura

Patients awaiting invasive procedures often have unanswered pre-procedural
questions; however, time-pressured workflows and privacy constraints limit
personalized counseling. We present LENOHA (Low Energy, No Hallucination, Leave
No One Behind Architecture), a safety-first, local-first system that routes
inputs with a high-precision sentence-transformer classifier and returns
verbatim answers from a clinician-curated FAQ for clinical queries, eliminating
free-text generation in the clinical path. We evaluated two domains (tooth
extraction and gastroscopy) using expert-reviewed validation sets
(n=400/domain) for thresholding and independent test sets (n=200/domain). Among
the four encoders, E5-large-instruct (560M) achieved an overall accuracy of
0.983 (95% CI 0.964-0.991), AUC 0.996, and seven total errors, which were
statistically indistinguishable from GPT-4o on this task; Gemini made no errors
on this test set. Energy logging shows that the non-generative clinical path
consumes ~1.0 mWh per input versus ~168 mWh per small-talk reply from a local
8B SLM, a ~170x difference, while maintaining ~0.10 s latency on a single
on-prem GPU. These results indicate that near-frontier discrimination and
generation-induced errors are structurally avoided in the clinical path by
returning vetted FAQ answers verbatim, supporting privacy, sustainability, and
equitable deployment in bandwidth-limited environments.

### 7. [Multimodal Feedback for Task Guidance in Augmented Reality](http://arxiv.org/pdf/2510.01690v1)

Authors: Hu Guo, Lily Patel, Rohan Gupt

Optical see-through augmented reality (OST-AR) overlays digital targets and
annotations on the physical world, offering promising guidance for hands-on
tasks such as medical needle insertion or assembly. Recent work on OST-AR depth
perception shows that target opacity and tool visualization significantly
affect accuracy and usability; opaque targets and rendering the real instrument
reduce depth errors, whereas transparent targets and absent tools impair
performance. However, reliance on visual overlays may overload attention and
leaves little room for depth cues when occlusion or lighting hampers
perception. To address these limitations, we explore multimodal feedback that
combines OST-AR with wrist-based vibrotactile haptics. The past two years have
seen rapid advances in haptic technology. Researchers have investigated
skin-stretch and vibrotactile cues for conveying spatial information to blind
users, wearable ring actuators that support precise pinching in AR, cross-modal
audio-haptic cursors that enable eyes-free object selection, and wrist-worn
feedback for teleoperated surgery that improves force awareness at the cost of
longer task times. Studies comparing pull versus push vibrotactile metaphors
found that pull cues yield faster gesture completion and lower cognitive load.
These findings motivate revisiting OST-AR guidance with a fresh perspective on
wrist-based haptics. We design a custom wristband with six vibromotors
delivering directional and state cues, integrate it with a handheld tool and
OST-AR, and assess its impact on cue recognition and depth guidance. Through a
formative study and two experiments (N=21 and N=27), we show that participants
accurately identify haptic patterns under cognitive load and that multimodal
feedback improves spatial precision and usability compared with visual-only or
haptic-only conditions.

### 8. [Komitee Equal Shares: Choosing Together as Voters and as Groups with a Co-designed Virtual Budget Algorithm](http://arxiv.org/pdf/2510.02040v1)

Authors: Joshua C. Yang, Noemi Scheurer

Public funding processes demand fairness, learning, and outcomes that
participants can understand. We introduce Komitee Equal Shares, a priceable
virtual-budget allocation framework that integrates two signals: in voter mode,
participants cast point votes; in evaluator mode, small groups assess proposals
against collectively defined impact fields. The framework extends the Method of
Equal Shares by translating both signals into virtual spending power and
producing voting receipts. We deployed the framework in the 2025 Kultur Komitee
in Winterthur, Switzerland. Our contributions are: (1) a clear separation of
decision modes, addressing a gap in social choice that typically treats
participatory budgeting as preference aggregation while citizens also see
themselves as evaluators; and (2) the design of voting receipts that
operationalise priceability into participant-facing explanations, making
proportional allocations legible and traceable. The framework generalises to
participatory grant-making and budgeting, offering a model where citizens act
as voters and evaluators within one proportional, explainable allocation.

### 9. [Human-Robo-advisor collaboration in decision-making: Evidence from a multiphase mixed methods experimental study](http://arxiv.org/pdf/2510.02153v1)

Authors: Hasan Mahmud, Najmul Islam, Satish Krishnan

Robo-advisors (RAs) are cost-effective, bias-resistant alternatives to human
financial advisors, yet adoption remains limited. While prior research has
examined user interactions with RAs, less is known about how individuals
interpret RA roles and integrate their advice into decision-making. To address
this gap, this study employs a multiphase mixed methods design integrating a
behavioral experiment (N = 334), thematic analysis, and follow-up quantitative
testing. Findings suggest that people tend to rely on RAs, with reliance shaped
by information about RA performance and the framing of advice as gains or
losses. Thematic analysis reveals three RA roles in decision-making and four
user types, each reflecting distinct patterns of advice integration. In
addition, a 2 x 2 typology categorizes antecedents of acceptance into enablers
and inhibitors at both the individual and algorithmic levels. By combining
behavioral, interpretive, and confirmatory evidence, this study advances
understanding of human-RA collaboration and provides actionable insights for
designing more trustworthy and adaptive RA systems.

### 10. [NeuroSwift: A Lightweight Cross-Subject Framework for fMRI Visual Reconstruction of Complex Scenes](http://arxiv.org/pdf/2510.02266v1)

Authors: Shiyi Zhang, Dong Liang, Yihang Zhou

Reconstructing visual information from brain activity via computer vision
technology provides an intuitive understanding of visual neural mechanisms.
Despite progress in decoding fMRI data with generative models, achieving
accurate cross-subject reconstruction of visual stimuli remains challenging and
computationally demanding. This difficulty arises from inter-subject
variability in neural representations and the brain's abstract encoding of core
semantic features in complex visual inputs. To address these challenges, we
propose NeuroSwift, which integrates complementary adapters via diffusion:
AutoKL for low-level features and CLIP for semantics. NeuroSwift's CLIP Adapter
is trained on Stable Diffusion generated images paired with COCO captions to
emulate higher visual cortex encoding. For cross-subject generalization, we
pretrain on one subject and then fine-tune only 17 percent of parameters (fully
connected layers) for new subjects, while freezing other components. This
enables state-of-the-art performance with only one hour of training per subject
on lightweight GPUs (three RTX 4090), and it outperforms existing methods.

### Information Retrieval

### 1. [IoDResearch: Deep Research on Private Heterogeneous Data via the Internet of Data](http://arxiv.org/pdf/2510.01553v1)

Authors: Zhuofan Shi, Zijie Guo, Xinjian Ma, Gang Huang, Yun Ma, Xiang Jing

The rapid growth of multi-source, heterogeneous, and multimodal scientific
data has increasingly exposed the limitations of traditional data management.
Most existing DeepResearch (DR) efforts focus primarily on web search while
overlooking local private data. Consequently, these frameworks exhibit low
retrieval efficiency for private data and fail to comply with the FAIR
principles, ultimately resulting in inefficiency and limited reusability. To
this end, we propose IoDResearch (Internet of Data Research), a private
data-centric Deep Research framework that operationalizes the Internet of Data
paradigm. IoDResearch encapsulates heterogeneous resources as FAIR-compliant
digital objects, and further refines them into atomic knowledge units and
knowledge graphs, forming a heterogeneous graph index for multi-granularity
retrieval. On top of this representation, a multi-agent system supports both
reliable question answering and structured scientific report generation.
Furthermore, we establish the IoD DeepResearch Benchmark to systematically
evaluate both data representation and Deep Research capabilities in IoD
scenarios. Experimental results on retrieval, QA, and report-writing tasks show
that IoDResearch consistently surpasses representative RAG and Deep Research
baselines. Overall, IoDResearch demonstrates the feasibility of
private-data-centric Deep Research under the IoD paradigm, paving the way
toward more trustworthy, reusable, and automated scientific discovery.

### 2. [Contrastive Retrieval Heads Improve Attention-Based Re-Ranking](http://arxiv.org/pdf/2510.02219v1)

Authors: Linh Tran, Yulong Li, Radu Florian, Wei Sun

The strong zero-shot and long-context capabilities of recent Large Language
Models (LLMs) have paved the way for highly effective re-ranking systems.
Attention-based re-rankers leverage attention weights from transformer heads to
produce relevance scores, but not all heads are created equally: many
contribute noise and redundancy, thus limiting performance. To address this, we
introduce CoRe heads, a small set of retrieval heads identified via a
contrastive scoring metric that explicitly rewards high attention heads that
correlate with relevant documents, while downplaying nodes with higher
attention that correlate with irrelevant documents. This relative ranking
criterion isolates the most discriminative heads for re-ranking and yields a
state-of-the-art list-wise re-ranker. Extensive experiments with three LLMs
show that aggregated signals from CoRe heads, constituting less than 1% of all
heads, substantially improve re-ranking accuracy over strong baselines. We
further find that CoRe heads are concentrated in middle layers, and pruning the
computation of final 50% of model layers preserves accuracy while significantly
reducing inference time and memory usage.

### 3. [Ranking Items from Discrete Ratings: The Cost of Unknown User Thresholds](http://arxiv.org/pdf/2510.01871v1)

Authors: Oscar Villemaud, Suryanarayana Sankagiri, Matthias Grossglauser

Ranking items is a central task in many information retrieval and recommender
systems. User input for the ranking task often comes in the form of ratings on
a coarse discrete scale. We ask whether it is possible to recover a
fine-grained item ranking from such coarse-grained ratings. We model items as
having scores and users as having thresholds; a user rates an item positively
if the item's score exceeds the user's threshold. Although all users agree on
the total item order, estimating that order is challenging when both the scores
and the thresholds are latent. Under our model, any ranking method naturally
partitions the $n$ items into bins; the bins are ordered, but the items inside
each bin are still unordered. Users arrive sequentially, and every new user can
be queried to refine the current ranking. We prove that achieving a
near-perfect ranking, measured by Spearman distance, requires $\Theta(n^2)$
users (and therefore $\Omega(n^2)$ queries). This is significantly worse than
the $O(n\log n)$ queries needed to rank from comparisons; the gap reflects the
additional queries needed to identify the users who have the appropriate
thresholds. Our bound also quantifies the impact of a mismatch between score
and threshold distributions via a quadratic divergence factor. To show the
tightness of our results, we provide a ranking algorithm whose query complexity
matches our bound up to a logarithmic factor. Our work reveals a tension in
online ranking: diversity in thresholds is necessary to merge coarse ratings
from many users into a fine-grained ranking, but this diversity has a cost if
the thresholds are a priori unknown.

### 4. [Study on LLMs for Promptagator-Style Dense Retriever Training](http://arxiv.org/pdf/2510.02241v1)

Authors: Daniel Gwon, Nour Jedidi, Jimmy Lin

Promptagator demonstrated that Large Language Models (LLMs) with few-shot
prompts can be used as task-specific query generators for fine-tuning
domain-specialized dense retrieval models. However, the original Promptagator
approach relied on proprietary and large-scale LLMs which users may not have
access to or may be prohibited from using with sensitive data. In this work, we
study the impact of open-source LLMs at accessible scales ($\leq$14B
parameters) as an alternative. Our results demonstrate that open-source LLMs as
small as 3B parameters can serve as effective Promptagator-style query
generators. We hope our work will inform practitioners with reliable
alternatives for synthetic data generation and give insights to maximize
fine-tuning results for domain-specific applications.

### 5. [Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomplete](http://arxiv.org/pdf/2510.01574v1)

Authors: Adithya Rajan, Xiaoyu Liu, Prateek Verma, Vibhu Arora

We introduce a data-centric approach for mitigating presentation bias in
real-time neural query autocomplete systems through the use of synthetic
prefixes. These prefixes are generated from complete user queries collected
during regular search sessions where autocomplete was not active. This allows
us to enrich the training data for learning to rank models with more diverse
and less biased examples. This method addresses the inherent bias in engagement
signals collected from live query autocomplete interactions, where model
suggestions influence user behavior. Our neural ranker is optimized for
real-time deployment under strict latency constraints and incorporates a rich
set of features, including query popularity, seasonality, fuzzy match scores,
and contextual signals such as department affinity, device type, and vertical
alignment with previous user queries. To support efficient training, we
introduce a task-specific simplification of the listwise loss, reducing
computational complexity from $O(n^2)$ to $O(n)$ by leveraging the query
autocomplete structure of having only one ground-truth selection per prefix.
Deployed in a large-scale e-commerce setting, our system demonstrates
statistically significant improvements in user engagement, as measured by mean
reciprocal rank and related metrics. Our findings show that synthetic prefixes
not only improve generalization but also provide a scalable path toward bias
mitigation in other low-latency ranking tasks, including related searches and
query recommendations.

### 6. [Bridging Collaborative Filtering and Large Language Models with Dynamic Alignment, Multimodal Fusion and Evidence-grounded Explanations](http://arxiv.org/pdf/2510.01606v1)

Authors: Bo Ma, LuYao Liu, Simon Lau, Chandler Yuan, and XueY Cui, Rosie Zhang

Recent research has explored using Large Language Models for recommendation
tasks by transforming user interaction histories and item metadata into text
prompts, then having the LLM produce rankings or recommendations. A promising
approach involves connecting collaborative filtering knowledge to LLM
representations through compact adapter networks, which avoids expensive
fine-tuning while preserving the strengths of both components. Yet several
challenges persist in practice: collaborative filtering models often use static
snapshots that miss rapidly changing user preferences; many real-world items
contain rich visual and audio content beyond textual descriptions; and current
systems struggle to provide trustworthy explanations backed by concrete
evidence. Our work introduces \model{}, a framework that tackles these
limitations through three key innovations. We develop an online adaptation
mechanism that continuously incorporates new user interactions through
lightweight modules, avoiding the need to retrain large models. We create a
unified representation that seamlessly combines collaborative signals with
visual and audio features, handling cases where some modalities may be
unavailable. Finally, we design an explanation system that grounds
recommendations in specific collaborative patterns and item attributes,
producing natural language rationales users can verify. Our approach maintains
the efficiency of frozen base models while adding minimal computational
overhead, making it practical for real-world deployment.

### 7. [LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing](http://arxiv.org/pdf/2510.01622v1)

Authors: Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Lau

Contemporary generative recommendation systems face significant challenges in
handling multimodal data, eliminating algorithmic biases, and providing
transparent decision-making processes. This paper introduces an enhanced
generative recommendation framework that addresses these limitations through
five key innovations: multimodal fusion architecture, retrieval-augmented
generation mechanisms, causal inference-based debiasing, explainable
recommendation generation, and real-time adaptive learning capabilities. Our
framework leverages advanced large language models as the backbone while
incorporating specialized modules for cross-modal understanding, contextual
knowledge integration, bias mitigation, explanation synthesis, and continuous
model adaptation. Extensive experiments on three benchmark datasets
(MovieLens-25M, Amazon-Electronics, Yelp-2023) demonstrate consistent
improvements in recommendation accuracy, fairness, and diversity compared to
existing approaches. The proposed framework achieves up to 2.3% improvement in
NDCG@10 and 1.4% enhancement in diversity metrics while maintaining
computational efficiency through optimized inference strategies.

### 8. [TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling](http://arxiv.org/pdf/2510.01698v1)

Authors: Seungheon Doh, Keunwoo Choi, Juhan Nam

While the recent developments in large language models (LLMs) have
successfully enabled generative recommenders with natural language
interactions, their recommendation behavior is limited, leaving other simpler
yet crucial components such as metadata or attribute filtering underutilized in
the system. We propose an LLM-based music recommendation system with tool
calling to serve as a unified retrieval-reranking pipeline. Our system
positions an LLM as an end-to-end recommendation system that interprets user
intent, plans tool invocations, and orchestrates specialized components:
boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding
similarity), and generative retrieval (semantic IDs). Through tool planning,
the system predicts which types of tools to use, their execution order, and the
arguments needed to find music matching user preferences, supporting diverse
modalities while seamlessly integrating multiple database filtering methods. We
demonstrate that this unified tool-calling framework achieves competitive
performance across diverse recommendation scenarios by selectively employing
appropriate retrieval methods based on user queries, envisioning a new paradigm
for conversational music recommendation systems.

### 9. [Comparison of Unsupervised Metrics for Evaluating Judicial Decision Extraction](http://arxiv.org/pdf/2510.01792v1)

Authors: Ivan Leonidovich Litvak, Anton Kostin, Fedor Lashkin, Tatiana Maksiyan, Sergey Lagutin

The rapid advancement of artificial intelligence in legal natural language
processing demands scalable methods for evaluating text extraction from
judicial decisions. This study evaluates 16 unsupervised metrics, including
novel formulations, to assess the quality of extracting seven semantic blocks
from 1,000 anonymized Russian judicial decisions, validated against 7,168
expert reviews on a 1--5 Likert scale. These metrics, spanning document-based,
semantic, structural, pseudo-ground truth, and legal-specific categories,
operate without pre-annotated ground truth. Bootstrapped correlations, Lin's
concordance correlation coefficient (CCC), and mean absolute error (MAE) reveal
that Term Frequency Coherence (Pearson $r = 0.540$, Lin CCC = 0.512, MAE =
0.127) and Coverage Ratio/Block Completeness (Pearson $r = 0.513$, Lin CCC =
0.443, MAE = 0.139) best align with expert ratings, while Legal Term Density
(Pearson $r = -0.479$, Lin CCC = -0.079, MAE = 0.394) show strong negative
correlations. The LLM Evaluation Score (mean = 0.849, Pearson $r = 0.382$, Lin
CCC = 0.325, MAE = 0.197) showed moderate alignment, but its performance, using
gpt-4.1-mini via g4f, suggests limited specialization for legal textse. These
findings highlight that unsupervised metrics, including LLM-based approaches,
enable scalable screening but, with moderate correlations and low CCC values,
cannot fully replace human judgment in high-stakes legal contexts. This work
advances legal NLP by providing annotation-free evaluation tools, with
implications for judicial analytics and ethical AI deployment.

### Machine Learning

### 1. [NVIDIA AI Aerial: AI-Native Wireless Communications](http://arxiv.org/pdf/2510.01533v1)

Authors: Kobi Cohen-Arazi, Michael Roe, Zhen Hu, Rohan Chavan, Anna Ptasznik, Joanna Lin, Joao Morais, Joseph Boccuzzi, Tommaso Balercia

6G brings a paradigm shift towards AI-native wireless systems, necessitating
the seamless integration of digital signal processing (DSP) and machine
learning (ML) within the software stacks of cellular networks. This
transformation brings the life cycle of modern networks closer to AI systems,
where models and algorithms are iteratively trained, simulated, and deployed
across adjacent environments. In this work, we propose a robust framework that
compiles Python-based algorithms into GPU-runnable blobs. The result is a
unified approach that ensures efficiency, flexibility, and the highest possible
performance on NVIDIA GPUs. As an example of the capabilities of the framework,
we demonstrate the efficacy of performing the channel estimation function in
the PUSCH receiver through a convolutional neural network (CNN) trained in
Python. This is done in a digital twin first, and subsequently in a real-time
testbed. Our proposed methodology, realized in the NVIDIA AI Aerial platform,
lays the foundation for scalable integration of AI/ML models into
next-generation cellular systems, and is essential for realizing the vision of
natively intelligent 6G networks.

### 2. [TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis](http://arxiv.org/pdf/2510.01538v1)

Authors: Haokun Zhao, Xiang Zhang, Jiaqi Wei, Yiwei Xu, Yuting He, Siqi Sun, Chenyu You

Time series forecasting is central to decision-making in domains as diverse
as energy, finance, climate, and public health. In practice, forecasters face
thousands of short, noisy series that vary in frequency, quality, and horizon,
where the dominant cost lies not in model fitting, but in the labor-intensive
preprocessing, validation, and ensembling required to obtain reliable
predictions. Prevailing statistical and deep learning models are tailored to
specific datasets or domains and generalize poorly. A general, domain-agnostic
framework that minimizes human intervention is urgently in demand. In this
paper, we introduce TimeSeriesScientist (TSci), the first LLM-driven agentic
framework for general time series forecasting. The framework comprises four
specialized agents: Curator performs LLM-guided diagnostics augmented by
external tools that reason over data statistics to choose targeted
preprocessing; Planner narrows the hypothesis space of model choice by
leveraging multi-modal diagnostics and self-planning over the input; Forecaster
performs model fitting and validation and, based on the results, adaptively
selects the best model configuration as well as ensemble strategy to make final
predictions; and Reporter synthesizes the whole process into a comprehensive,
transparent report. With transparent natural-language rationales and
comprehensive reports, TSci transforms the forecasting workflow into a
white-box system that is both interpretable and extensible across tasks.
Empirical results on eight established benchmarks demonstrate that TSci
consistently outperforms both statistical and LLM-based baselines, reducing
forecast error by an average of 10.4% and 38.2%, respectively. Moreover, TSci
produces a clear and rigorous report that makes the forecasting workflow more
transparent and interpretable.

### 3. [Executable Counterfactuals: Improving LLMs' Causal Reasoning Through Code](http://arxiv.org/pdf/2510.01539v1)

Authors: Aniket Vashishtha, Qirun Dai, Hongyuan Mei, Amit Sharma, Chenhao Tan, Hao Peng

Counterfactual reasoning, a hallmark of intelligence, consists of three
steps: inferring latent variables from observations (abduction), constructing
alternatives (interventions), and predicting their outcomes (prediction). This
skill is essential for advancing LLMs' causal understanding and expanding their
applications in high-stakes domains such as scientific research. However,
existing efforts in assessing LLM's counterfactual reasoning capabilities tend
to skip the abduction step, effectively reducing to interventional reasoning
and leading to overestimation of LLM performance. To address this, we introduce
executable counterfactuals, a novel framework that operationalizes causal
reasoning through code and math problems. Our framework explicitly requires all
three steps of counterfactual reasoning and enables scalable synthetic data
creation with varying difficulty, creating a frontier for evaluating and
improving LLM's reasoning. Our results reveal substantial drop in accuracy
(25-40%) from interventional to counterfactual reasoning for SOTA models like
o4-mini and Claude-4-Sonnet. To address this gap, we construct a training set
comprising counterfactual code problems having if-else condition and test on
out-of-domain code structures (e.g. having while-loop); we also test whether a
model trained on code would generalize to counterfactual math word problems.
While supervised finetuning on stronger models' reasoning traces improves
in-domain performance of Qwen models, it leads to a decrease in accuracy on OOD
tasks such as counterfactual math problems. In contrast, reinforcement learning
induces the core cognitive behaviors and generalizes to new domains, yielding
gains over the base model on both code (improvement of 1.5x-2x) and math
problems. Analysis of the reasoning traces reinforces these findings and
highlights the promise of RL for improving LLMs' counterfactual reasoning.

### 4. [MIRA: Towards Mitigating Reward Hacking in Inference-Time Alignment of T2I Diffusion Models](http://arxiv.org/pdf/2510.01549v1)

Authors: Kevin Zhai, Utsav Singh, Anirudh Thatipelli, Souradip Chakraborty, Anit Kumar Sahu, Furong Huang, Amrit Singh Bedi, Mubarak Shah

Diffusion models excel at generating images conditioned on text prompts, but
the resulting images often do not satisfy user-specific criteria measured by
scalar rewards such as Aesthetic Scores. This alignment typically requires
fine-tuning, which is computationally demanding. Recently, inference-time
alignment via noise optimization has emerged as an efficient alternative,
modifying initial input noise to steer the diffusion denoising process towards
generating high-reward images. However, this approach suffers from reward
hacking, where the model produces images that score highly, yet deviate
significantly from the original prompt. We show that noise-space regularization
is insufficient and that preventing reward hacking requires an explicit
image-space constraint. To this end, we propose MIRA (MItigating Reward
hAcking), a training-free, inference-time alignment method. MIRA introduces an
image-space, score-based KL surrogate that regularizes the sampling trajectory
with a frozen backbone, constraining the output distribution so reward can
increase without off-distribution drift (reward hacking). We derive a tractable
approximation to KL using diffusion scores. Across SDv1.5 and SDXL, multiple
rewards (Aesthetic, HPSv2, PickScore), and public datasets (e.g.,
Animal-Animal, HPDv2), MIRA achieves >60\% win rate vs. strong baselines while
preserving prompt adherence; mechanism plots show reward gains with near-zero
drift, whereas DNO drifts as compute increases. We further introduce MIRA-DPO,
mapping preference optimization to inference time with a frozen backbone,
extending MIRA to non-differentiable rewards without fine-tuning.

### 5. [Large-Scale Bayesian Causal Discovery with Interventional Data](http://arxiv.org/pdf/2510.01562v1)

Authors: Seong Woo Han, Daniel Duy Vo, Brielin C. Brown

Inferring the causal relationships among a set of variables in the form of a
directed acyclic graph (DAG) is an important but notoriously challenging
problem. Recently, advancements in high-throughput genomic perturbation screens
have inspired development of methods that leverage interventional data to
improve model identification. However, existing methods still suffer poor
performance on large-scale tasks and fail to quantify uncertainty. Here, we
propose Interventional Bayesian Causal Discovery (IBCD), an empirical Bayesian
framework for causal discovery with interventional data. Our approach models
the likelihood of the matrix of total causal effects, which can be approximated
by a matrix normal distribution, rather than the full data matrix. We place a
spike-and-slab horseshoe prior on the edges and separately learn data-driven
weights for scale-free and Erd\H{o}s-R\'enyi structures from observational
data, treating each edge as a latent variable to enable uncertainty-aware
inference. Through extensive simulation, we show that IBCD achieves superior
structure recovery compared to existing baselines. We apply IBCD to CRISPR
perturbation (Perturb-seq) data on 521 genes, demonstrating that edge posterior
inclusion probabilities enable identification of robust graph structures.

### 6. [Gradient Shaping Beyond Clipping: A Functional Perspective on Update Magnitude Control](http://arxiv.org/pdf/2510.01578v1)

Authors: Haochen You, Baojing Liu

Gradient clipping is widely used to stabilize deep network training, but its
formulation as a hard, fixed threshold limits flexibility and ignores gradient
distribution dynamics. We propose SPAMP (Statistical Per-layer Adaptive
Modulation and Projection), a unified framework that generalizes clipping into
smooth, per-layer gradient shaping. SPAMP tracks local gradient statistics,
dynamically estimates thresholds, and applies power-based transformations to
modulate update magnitudes in a differentiable manner. This perspective recasts
clipping and warmup as dual mechanisms for controlling the effective update
scale $\eta_t \|g_t\|$, offering a principled alternative to rigid heuristics.
Extensive experiments across image and language tasks demonstrate that SPAMP
improves stability, convergence, and robustness over existing methods.

### 7. [Posterior Collapse as a Phase Transition in Variational Autoencoders](http://arxiv.org/pdf/2510.01621v1)

Authors: Zhen Li, Fan Zhang, Zheng Zhang, Yu Chen

We investigate the phenomenon of posterior collapse in variational
autoencoders (VAEs) from the perspective of statistical physics, and reveal
that it constitutes a phase transition governed jointly by data structure and
model hyper-parameters. By analyzing the stability of the trivial solution
associated with posterior collapse, we identify a critical hyper-parameter
threshold. This critical boundary, separating meaningful latent inference from
collapse, is characterized by a discontinuity in the KL divergence between the
approximate posterior and the prior distribution. We validate this critical
behavior on both synthetic and real-world datasets, confirming the existence of
a phase transition. Our results demonstrate that posterior collapse is not
merely an optimization failure, but rather an emerging phase transition arising
from the interplay between data structure and variational constraints. This
perspective offers new insights into the trainability and representational
capacity of deep generative models.

### 8. [CAT: Curvature-Adaptive Transformers for Geometry-Aware Learning](http://arxiv.org/pdf/2510.01634v1)

Authors: Ryan Y. Lin, Siddhartha Ojha, Nicholas Bai

Transformers achieve strong performance across diverse domains but implicitly
assume Euclidean geometry in their attention mechanisms, limiting their
effectiveness on data with non-Euclidean structure. While recent extensions to
hyperbolic and spherical spaces show promise for hierarchical and cyclical
patterns, respectively, they require committing to a single geometry a priori,
reducing flexibility when data exhibits mixed geometric properties. We
introduce the Curvature-Adaptive Transformer (CAT), a novel architecture that
dynamically learns per-token routing across three geometric attention branches
through a lightweight, differentiable gating mechanism. Unlike fixed-geometry
approaches, CAT enables adaptive geometric specialization, routing tokens to
the appropriate curvature based on their local relational structure. The
routing network provides interpretable curvature preferences while each branch
employs geometry-specific operations optimized for its respective manifold. On
knowledge graph completion benchmarks (FB15k-237, WN18RR), CAT achieves
approximately 10% improvements in MRR and Hits@10 over fixed-geometry baselines
with minimal overhead (5% parameter increase, comparable inference time). These
results demonstrate that learned geometric adaptation outperforms any single
fixed geometry for complex relational reasoning, establishing CAT as a scalable
and interpretable foundation for mixture-of-geometry architectures across
language, vision, and multimodal domains.

### 9. [Detecting Post-generation Edits to Watermarked LLM Outputs via Combinatorial Watermarking](http://arxiv.org/pdf/2510.01637v1)

Authors: Liyan Xie, Muhammad Siddeek, Mohamed Seif, Andrea J. Goldsmith, Mengdi Wang

Watermarking has become a key technique for proprietary language models,
enabling the distinction between AI-generated and human-written text. However,
in many real-world scenarios, LLM-generated content may undergo post-generation
edits, such as human revisions or even spoofing attacks, making it critical to
detect and localize such modifications. In this work, we introduce a new task:
detecting post-generation edits locally made to watermarked LLM outputs. To
this end, we propose a combinatorial pattern-based watermarking framework,
which partitions the vocabulary into disjoint subsets and embeds the watermark
by enforcing a deterministic combinatorial pattern over these subsets during
generation. We accompany the combinatorial watermark with a global statistic
that can be used to detect the watermark. Furthermore, we design lightweight
local statistics to flag and localize potential edits. We introduce two
task-specific evaluation metrics, Type-I error rate and detection accuracy, and
evaluate our method on open-source LLMs across a variety of editing scenarios,
demonstrating strong empirical performance in edit localization.

### 10. [Support Basis: Fast Attention Beyond Bounded Entries](http://arxiv.org/pdf/2510.01643v1)

Authors: Maryam Aliakbarpour, Vladimir Braverman, Junze Yin, Haochen Zhang

The quadratic complexity of softmax attention remains a central bottleneck in
scaling large language models (LLMs). [Alman and Song, NeurIPS 2023] proposed a
sub-quadratic attention approximation algorithm, but it works only under the
restrictive bounded-entry assumption. Since this assumption rarely holds in
practice, its applicability to modern LLMs is limited.
  In this paper, we introduce support-basis decomposition, a new framework for
efficient attention approximation beyond bounded entries. We empirically
demonstrate that the entries of the query and key matrices exhibit sub-Gaussian
behavior. Our approach uses this property to split large and small entries,
enabling exact computation on sparse components and polynomial approximation on
dense components. We establish rigorous theoretical guarantees, proving a
sub-quadratic runtime, and extend the method to a multi-threshold setting that
eliminates all distributional assumptions. Furthermore, we provide the first
theoretical justification for the empirical success of polynomial attention
[Kacham, Mirrokni, and Zhong, ICML 2024], showing that softmax attention can be
closely approximated by a combination of multiple polynomial attentions with
sketching.

### Neural and Evolutionary Computing

### 1. [Microscaling Floating Point Formats for Large Language Models](http://arxiv.org/pdf/2510.01863v1)

Authors: Marco Cococcioni, Dario Pagani, Federico Rossi

The increasing computational and memory demands of large language models
(LLMs) necessitate innovative approaches to optimize resource usage without
compromising performance. This paper leverages microscaling floating-point
formats, a novel technique designed to address these challenges by reducing the
storage and computational overhead associated with numerical representations in
LLMs. Unlike traditional floating-point representations that allocate a
dedicated scale for each value, microscaling employs a shared scale across a
block of values, enabling compact one-byte floating-point representations while
maintaining an extended dynamic range. We explore the application of
microscaling in the context of 8-bit floating-point formats to significantly
reduce memory footprint and computational costs. We tested several
configurations of microscaling floats within the GPT-2 LLM architecture,
demonstrating that microscaling data formats can achieve competitive accuracy
during training and inference, proving its efficacy as a resource-efficient
alternative for deploying LLMs at scale. The source code is publicly available
at: https://github.com/unipi-dii-compressedarith/llm.c-sve

### 2. [Multiplier-free In-Memory Vector-Matrix Multiplication Using Distributed Arithmetic](http://arxiv.org/pdf/2510.02099v1)

Authors: Felix Zeller, John Reuben, Dietmar Fey

Vector-Matrix Multiplication (VMM) is the fundamental and frequently required
computation in inference of Neural Networks (NN). Due to the large data
movement required during inference, VMM can benefit greatly from in-memory
computing. However, ADC/DACs required for in-memory VMM consume significant
power and area. `Distributed Arithmetic (DA)', a technique in computer
architecture prevalent in 1980s was used to achieve inner product or dot
product of two vectors without using a hard-wired multiplier when one of the
vectors is a constant. In this work, we extend the DA technique to multiply an
input vector with a constant matrix. By storing the sum of the weights in
memory, DA achieves VMM using shift-and-add circuits in the periphery of ReRAM
memory. We verify functional and also estimate non-functional properties
(latency, energy, area) by performing transistor-level simulations. Using
energy-efficient sensing and fine grained pipelining, our approach achieves 4.5
x less latency and 12 x less energy than VMM performed in memory conventionally
by bit slicing. Furthermore, DA completely eliminated the need for power-hungry
ADCs which are the main source of area and energy consumption in the current
VMM implementations in memory.

### 3. [VarCoNet: A variability-aware self-supervised framework for functional connectome extraction from resting-state fMRI](http://arxiv.org/pdf/2510.02120v1)

Authors: Charalampos Lamprou, Aamna Alshehhi, Leontios J. Hadjileontiadis, Mohamed L. Seghier

Accounting for inter-individual variability in brain function is key to
precision medicine. Here, by considering functional inter-individual
variability as meaningful data rather than noise, we introduce VarCoNet, an
enhanced self-supervised framework for robust functional connectome (FC)
extraction from resting-state fMRI (rs-fMRI) data. VarCoNet employs
self-supervised contrastive learning to exploit inherent functional
inter-individual variability, serving as a brain function encoder that
generates FC embeddings readily applicable to downstream tasks even in the
absence of labeled data. Contrastive learning is facilitated by a novel
augmentation strategy based on segmenting rs-fMRI signals. At its core,
VarCoNet integrates a 1D-CNN-Transformer encoder for advanced time-series
processing, enhanced with a robust Bayesian hyperparameter optimization. Our
VarCoNet framework is evaluated on two downstream tasks: (i) subject
fingerprinting, using rs-fMRI data from the Human Connectome Project, and (ii)
autism spectrum disorder (ASD) classification, using rs-fMRI data from the
ABIDE I and ABIDE II datasets. Using different brain parcellations, our
extensive testing against state-of-the-art methods, including 13 deep learning
methods, demonstrates VarCoNet's superiority, robustness, interpretability, and
generalizability. Overall, VarCoNet provides a versatile and robust framework
for FC analysis in rs-fMRI.

### Networking and Internet Architecture

### 1. [MMGaP: Multi-User MIMO Detection and Precoding using GPU-assisted Physics-inspired Computation](http://arxiv.org/pdf/2510.01579v1)

Authors: Abhishek Kumar Singh, Kyle Jamieson

Physics-inspired and quantum compute based methods for processing in the
physical layer of next-generation cellular radio access networks have
demonstrated theoretical advances in spectral efficiency in recent years, but
have stopped short of practical realization on commodity processors, leaving a
gap between the throughput practical systems can achieve and the projected
throughput the state-of-the-art should achieve. To fill this gap, this paper
proposes MMGaP, an uplink multi-user MIMO detector and downlink Vector
perturbation precoder for next-generation cellular networks. MMGaP realizes
these large MIMO processing algorithms for the first time on bare-metal CUDA
kernels that scale to run on large GPU processing platforms, and can be
packaged as TensorFlow modules, allowing easy integration with a variety of
systems. We integrate MMGaP with NVIDIA's software-defined, GPU-accelerated 5G
platform and evaluate its performance against the state-of-the-art. In a 5G
cellular network using 100 MHz of radio bandwidth, eight antennas at the base
station and eight concurrent users, we show that MMGaP improves uplink
throughput by approximately 50 Mbps per user and downlink throughput by 100
Mbps per user over a wide range of SNR. We further show that MMGaP can also
support larger MIMO sizes: for 16 antennas at the base station and 16
concurrent users, MMGaP provides more than 50 Mbps higher uplink throughput per
user. We measure the execution time of MMGaP on different NVIDIA GPUs and show
that it can operate at line-rate and meet the timing requirements of
state-of-the-art 5G systems.

### 2. [ReSSFormer: A Recursive Sparse Structured Transformer for Scalable and Long-Context Reasoning](http://arxiv.org/pdf/2510.01585v1)

Authors: Haochen You, Baojing Liu

While Transformer architectures have demonstrated impressive scalability
across domains, they continue to face challenges in long-context reasoning,
computational efficiency, and structural generalization - largely due to rigid
layer stacking, dense attention, and reliance on positional encodings. We
present ReSSFormer, a Recursive Sparse Structured Transformer that integrates
three complementary innovations: Recurrent Reasoning & Memory Unit (R2MU) for
iterative reasoning with bounded depth, Adaptive Sparse Attention Module (ASAM)
for efficient and focused context selection, and Self-Organizing Encoder
Structure (SOES) for position-free structure induction. ReSSFormer replaces
conventional depth stacking with recurrent inference, substitutes full
attention with token- and expert-level sparsity, and models latent token
topology directly from content. Across language modeling, multi-hop QA, and
structure-sensitive tasks, ReSSFormer consistently outperforms strong baselines
under comparable FLOPs and parameter budgets, highlighting its scalability,
efficiency, and structural flexibility.

### 3. [Accuracy vs Performance: An abstraction model for deadline constrained offloading at the mobile-edge](http://arxiv.org/pdf/2510.01885v1)

Authors: Jamie Cotter, Ignacio Castineiras, Victor Cionca

In this paper, we present a solution for low-latency deadline-constrained DNN
offloading on mobile edge devices. We design a scheduling algorithm with
lightweight network state representation, considering device availability,
communication on the network link, priority-aware pre-emption, and task
deadlines. The scheduling algorithm aims to reduce latency by designing a
resource availability representation, as well as a network discretisation and a
dynamic bandwidth estimation mechanism. We implement the scheduling algorithm
into a system composed of four Raspberry Pi 2 (model Bs) mobile edge devices,
sampling a waste classification conveyor belt at a set frame rate. The system
is evaluated and compared to a previous approach of ours, which was proven to
outcompete work-stealers and a non-pre-emption based scheduling heuristic under
the aforementioned waste classification scenario. Our findings show the novel
lower latency abstraction models yield better performance under high-volume
workloads, with the dynamic bandwidth estimation assisting the task placement
while, ultimately, increasing task throughput in times of resource scarcity.

### 4. [How to Combat Reactive and Dynamic Jamming Attacks with Reinforcement Learning](http://arxiv.org/pdf/2510.02265v1)

Authors: Yalin E. Sagduyu, Tugba Erpek, Kemal Davaslioglu, Sastry Kompella

This paper studies the problem of mitigating reactive jamming, where a jammer
adopts a dynamic policy of selecting channels and sensing thresholds to detect
and jam ongoing transmissions. The transmitter-receiver pair learns to avoid
jamming and optimize throughput over time (without prior knowledge of channel
conditions or jamming strategies) by using reinforcement learning (RL) to adapt
transmit power, modulation, and channel selection. Q-learning is employed for
discrete jamming-event states, while Deep Q-Networks (DQN) are employed for
continuous states based on received power. Through different reward functions
and action sets, the results show that RL can adapt rapidly to spectrum
dynamics and sustain high rates as channels and jamming policies change over
time.

### Robotics

### 1. [Real-time Multi-Plane Segmentation Based on GPU Accelerated High-Resolution 3D Voxel Mapping for Legged Robot Locomotion](http://arxiv.org/pdf/2510.01592v1)

Authors: Shun Niijima, Ryoichi Tsuzaki, Noriaki Takasugi, Masaya Kinoshita

This paper proposes a real-time multi-plane segmentation method based on
GPU-accelerated high-resolution 3D voxel mapping for legged robot locomotion.
Existing online planar mapping approaches struggle to balance accuracy and
computational efficiency: direct depth image segmentation from specific sensors
suffers from poor temporal integration, height map-based methods cannot
represent complex 3D structures like overhangs, and voxel-based plane
segmentation remains unexplored for real-time applications. To address these
limitations, we develop a novel framework that integrates vertex-based
connected component labeling with random sample consensus based plane detection
and convex hull, leveraging GPU parallel computing to rapidly extract planar
regions from point clouds accumulated in high-resolution 3D voxel maps.
Experimental results demonstrate that the proposed method achieves fast and
accurate 3D multi-plane segmentation at over 30 Hz update rate even at a
resolution of 0.01 m, enabling the detected planes to be utilized in real time
for locomotion tasks. Furthermore, we validate the effectiveness of our
approach through experiments in both simulated environments and physical legged
robot platforms, confirming robust locomotion performance when considering 3D
planar structures.

### 2. [MiniBEE: A New Form Factor for Compact Bimanual Dexterity](http://arxiv.org/pdf/2510.01603v1)

Authors: Sharfin Islam, Zewen Chen, Zhanpeng He, Swapneel Bhatt, Andres Permuy, Brock Taylor, James Vickery, Pedro Piacenza, Cheng Zhang, Matei Ciocarlie

Bimanual robot manipulators can achieve impressive dexterity, but typically
rely on two full six- or seven- degree-of-freedom arms so that paired grippers
can coordinate effectively. This traditional framework increases system
complexity while only exploiting a fraction of the overall workspace for
dexterous interaction. We introduce the MiniBEE (Miniature Bimanual
End-effector), a compact system in which two reduced-mobility arms (3+ DOF
each) are coupled into a kinematic chain that preserves full relative
positioning between grippers. To guide our design, we formulate a kinematic
dexterity metric that enlarges the dexterous workspace while keeping the
mechanism lightweight and wearable. The resulting system supports two
complementary modes: (i) wearable kinesthetic data collection with self-tracked
gripper poses, and (ii) deployment on a standard robot arm, extending dexterity
across its entire workspace. We present kinematic analysis and design
optimization methods for maximizing dexterous range, and demonstrate an
end-to-end pipeline in which wearable demonstrations train imitation learning
policies that perform robust, real-world bimanual manipulation.

### 3. [FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action Models](http://arxiv.org/pdf/2510.01642v1)

Authors: Zijun Lin, Jiafei Duan, Haoquan Fang, Dieter Fox, Ranjay Krishna, Cheston Tan, Bihan Wen

Recent advances in robotic manipulation have integrated low-level robotic
control into Vision-Language Models (VLMs), extending them into
Vision-Language-Action (VLA) models. Although state-of-the-art VLAs achieve
strong performance in downstream robotic applications, supported by large-scale
crowd-sourced robot training data, they still inevitably encounter failures
during execution. Enabling robots to reason about and recover from
unpredictable and abrupt failures remains a critical challenge. Existing
robotic manipulation datasets, collected in either simulation or the real
world, primarily provide only ground-truth trajectories, leaving robots unable
to recover once failures occur. Moreover, the few datasets that address failure
detection typically offer only textual explanations, which are difficult to
utilize directly in VLA models. To address this gap, we introduce FailSafe, a
novel failure generation and recovery system that automatically produces
diverse failure cases paired with executable recovery actions. FailSafe can be
seamlessly applied to any manipulation task in any simulator, enabling scalable
creation of failure-action data. To demonstrate its effectiveness, we fine-tune
LLaVa-OneVision-7B (LLaVa-OV-7B) to build FailSafe-VLM. Experimental results
show that FailSafe-VLM successfully helps robotic arm detect and recover from
potential failures, improving the performance of three state-of-the-art VLA
models pi0-FAST, OpenVLA, OpenVLA-OFT) by up to 22.6% on average across several
tasks in Maniskill. Furthermore, FailSafe-VLM could generalize across different
spatial configurations, camera viewpoints, and robotic embodiments. We plan to
release the FailSafe code to the community.

### 4. [Statistical Uncertainty Learning for Robust Visual-Inertial State Estimation](http://arxiv.org/pdf/2510.01648v1)

Authors: Seungwon Choi, Donggyu Park, Seo-Yeon Hwang, Tae-Wan Kim

A fundamental challenge in robust visual-inertial odometry (VIO) is to
dynamically assess the reliability of sensor measurements. This assessment is
crucial for properly weighting the contribution of each measurement to the
state estimate. Conventional methods often simplify this by assuming a static,
uniform uncertainty for all measurements. This heuristic, however, may be
limited in its ability to capture the dynamic error characteristics inherent in
real-world data. To improve this limitation, we present a statistical framework
that learns measurement reliability assessment online, directly from sensor
data and optimization results. Our approach leverages multi-view geometric
consistency as a form of self-supervision. This enables the system to infer
landmark uncertainty and adaptively weight visual measurements during
optimization. We evaluated our method on the public EuRoC dataset,
demonstrating improvements in tracking accuracy with average reductions of
approximately 24\% in translation error and 42\% in rotation error compared to
baseline methods with fixed uncertainty parameters. The resulting framework
operates in real time while showing enhanced accuracy and robustness. To
facilitate reproducibility and encourage further research, the source code will
be made publicly available.

### 5. [Symskill: Symbol and Skill Co-Invention for Data-Efficient and Real-Time Long-Horizon Manipulation](http://arxiv.org/pdf/2510.01661v1)

Authors: Yifei Simon Shao, Yuchen Zheng, Sunan Sun, Pratik Chaudhari, Vijay Kumar, Nadia Figueroa

Multi-step manipulation in dynamic environments remains challenging. Two
major families of methods fail in distinct ways: (i) imitation learning (IL) is
reactive but lacks compositional generalization, as monolithic policies do not
decide which skill to reuse when scenes change; (ii) classical task-and-motion
planning (TAMP) offers compositionality but has prohibitive planning latency,
preventing real-time failure recovery. We introduce SymSkill, a unified
learning framework that combines the benefits of IL and TAMP, allowing
compositional generalization and failure recovery in real-time. Offline,
SymSkill jointly learns predicates, operators, and skills directly from
unlabeled and unsegmented demonstrations. At execution time, upon specifying a
conjunction of one or more learned predicates, SymSkill uses a symbolic planner
to compose and reorder learned skills to achieve the symbolic goals, while
performing recovery at both the motion and symbolic levels in real time.
Coupled with a compliant controller, SymSkill enables safe and uninterrupted
execution under human and environmental disturbances. In RoboCasa simulation,
SymSkill can execute 12 single-step tasks with 85% success rate. Without
additional data, it composes these skills into multi-step plans requiring up to
6 skill recompositions, recovering robustly from execution failures. On a real
Franka robot, we demonstrate SymSkill, learning from 5 minutes of unsegmented
and unlabeled play data, is capable of performing multiple tasks simply by goal
specifications. The source code and additional analysis can be found on
https://sites.google.com/view/symskill.

### 6. [An Anytime, Scalable and Complete Algorithm for Embedding a Manufacturing Procedure in a Smart Factory](http://arxiv.org/pdf/2510.01770v1)

Authors: Christopher Leet, Aidan Sciortino, Sven Koenig

Modern automated factories increasingly run manufacturing procedures using a
matrix of programmable machines, such as 3D printers, interconnected by a
programmable transport system, such as a fleet of tabletop robots. To embed a
manufacturing procedure into a smart factory, an operator must: (a) assign each
of its processes to a machine and (b) specify how agents should transport parts
between machines. The problem of embedding a manufacturing process into a smart
factory is termed the Smart Factory Embedding (SFE) problem. State-of-the-art
SFE solvers can only scale to factories containing a couple dozen machines.
Modern smart factories, however, may contain hundreds of machines. We fill this
hole by introducing the first highly scalable solution to the SFE, TS-ACES, the
Traffic System based Anytime Cyclic Embedding Solver. We show that TS-ACES is
complete and can scale to SFE instances based on real industrial scenarios with
more than a hundred machines.

### 7. [What Matters in RL-Based Methods for Object-Goal Navigation? An Empirical Study and A Unified Framework](http://arxiv.org/pdf/2510.01830v1)

Authors: Hongze Wang, Boyang Sun, Jiaxu Xing, Fan Yang, Marco Hutter, Dhruv Shah, Davide Scaramuzza, Marc Pollefeys

Object-Goal Navigation (ObjectNav) is a critical component toward deploying
mobile robots in everyday, uncontrolled environments such as homes, schools,
and workplaces. In this context, a robot must locate target objects in
previously unseen environments using only its onboard perception. Success
requires the integration of semantic understanding, spatial reasoning, and
long-horizon planning, which is a combination that remains extremely
challenging. While reinforcement learning (RL) has become the dominant
paradigm, progress has spanned a wide range of design choices, yet the field
still lacks a unifying analysis to determine which components truly drive
performance. In this work, we conduct a large-scale empirical study of modular
RL-based ObjectNav systems, decomposing them into three key components:
perception, policy, and test-time enhancement. Through extensive controlled
experiments, we isolate the contribution of each and uncover clear trends:
perception quality and test-time strategies are decisive drivers of
performance, whereas policy improvements with current methods yield only
marginal gains. Building on these insights, we propose practical design
guidelines and demonstrate an enhanced modular system that surpasses
State-of-the-Art (SotA) methods by 6.6% on SPL and by a 2.7% success rate. We
also introduce a human baseline under identical conditions, where experts
achieve an average 98% success, underscoring the gap between RL agents and
human-level navigation. Our study not only sets the SotA performance but also
provides principled guidance for future ObjectNav development and evaluation.

### 8. [Like Playing a Video Game: Spatial-Temporal Optimization of Foot Trajectories for Controlled Football Kicking in Bipedal Robots](http://arxiv.org/pdf/2510.01843v1)

Authors: Wanyue Li, Ji Ma, Minghao Lu, Peng Lu

Humanoid robot soccer presents several challenges, particularly in
maintaining system stability during aggressive kicking motions while achieving
precise ball trajectory control. Current solutions, whether traditional
position-based control methods or reinforcement learning (RL) approaches,
exhibit significant limitations. Model predictive control (MPC) is a prevalent
approach for ordinary quadruped and biped robots. While MPC has demonstrated
advantages in legged robots, existing studies often oversimplify the leg swing
progress, relying merely on simple trajectory interpolation methods. This
severely constrains the foot's environmental interaction capability, hindering
tasks such as ball kicking. This study innovatively adapts the spatial-temporal
trajectory planning method, which has been successful in drone applications, to
bipedal robotic systems. The proposed approach autonomously generates foot
trajectories that satisfy constraints on target kicking position, velocity, and
acceleration while simultaneously optimizing swing phase duration. Experimental
results demonstrate that the optimized trajectories closely mimic human kicking
behavior, featuring a backswing motion. Simulation and hardware experiments
confirm the algorithm's efficiency, with trajectory planning times under 1 ms,
and its reliability, achieving nearly 100 % task completion accuracy when the
soccer goal is within the range of -90{\deg} to 90{\deg}.

### 9. [GreenhouseSplat: A Dataset of Photorealistic Greenhouse Simulations for Mobile Robotics](http://arxiv.org/pdf/2510.01848v1)

Authors: Diram Tabaa, Gianni Di Caro

Simulating greenhouse environments is critical for developing and evaluating
robotic systems for agriculture, yet existing approaches rely on simplistic or
synthetic assets that limit simulation-to-real transfer. Recent advances in
radiance field methods, such as Gaussian splatting, enable photorealistic
reconstruction but have so far been restricted to individual plants or
controlled laboratory conditions. In this work, we introduce GreenhouseSplat, a
framework and dataset for generating photorealistic greenhouse assets directly
from inexpensive RGB images. The resulting assets are integrated into a
ROS-based simulation with support for camera and LiDAR rendering, enabling
tasks such as localization with fiducial markers. We provide a dataset of 82
cucumber plants across multiple row configurations and demonstrate its utility
for robotics evaluation. GreenhouseSplat represents the first step toward
greenhouse-scale radiance-field simulation and offers a foundation for future
research in agricultural robotics.

### 10. [EC3R-SLAM: Efficient and Consistent Monocular Dense SLAM with Feed-Forward 3D Reconstruction](http://arxiv.org/pdf/2510.02080v1)

Authors: Lingxiang Hu, Naima Ait Oufroukh, Fabien Bonardi, Raymond Ghandour

The application of monocular dense Simultaneous Localization and Mapping
(SLAM) is often hindered by high latency, large GPU memory consumption, and
reliance on camera calibration. To relax this constraint, we propose EC3R-SLAM,
a novel calibration-free monocular dense SLAM framework that jointly achieves
high localization and mapping accuracy, low latency, and low GPU memory
consumption. This enables the framework to achieve efficiency through the
coupling of a tracking module, which maintains a sparse map of feature points,
and a mapping module based on a feed-forward 3D reconstruction model that
simultaneously estimates camera intrinsics. In addition, both local and global
loop closures are incorporated to ensure mid-term and long-term data
association, enforcing multi-view consistency and thereby enhancing the overall
accuracy and robustness of the system. Experiments across multiple benchmarks
show that EC3R-SLAM achieves competitive performance compared to
state-of-the-art methods, while being faster and more memory-efficient.
Moreover, it runs effectively even on resource-constrained platforms such as
laptops and Jetson Orin NX, highlighting its potential for real-world robotics
applications.

### Software Engineering

### 1. [MIMIC: Integrating Diverse Personality Traits for Better Game Testing Using Large Language Model](http://arxiv.org/pdf/2510.01635v1)

Authors: Yifei Chen, Sarra Habchi, Lili Wei

Modern video games pose significant challenges for traditional automated
testing algorithms, yet intensive testing is crucial to ensure game quality. To
address these challenges, researchers designed gaming agents using
Reinforcement Learning, Imitation Learning, or Large Language Models. However,
these agents often neglect the diverse strategies employed by human players due
to their different personalities, resulting in repetitive solutions in similar
situations. Without mimicking varied gaming strategies, these agents struggle
to trigger diverse in-game interactions or uncover edge cases.
  In this paper, we present MIMIC, a novel framework that integrates diverse
personality traits into gaming agents, enabling them to adopt different gaming
strategies for similar situations. By mimicking different playstyles, MIMIC can
achieve higher test coverage and richer in-game interactions across different
games. It also outperforms state-of-the-art agents in Minecraft by achieving a
higher task completion rate and providing more diverse solutions. These results
highlight MIMIC's significant potential for effective game testing.

### 2. [FOSS-chain: using blockchain for Open Source Software license compliance](http://arxiv.org/pdf/2510.01740v1)

Authors: Kypros Iacovou, Georgia M. Kapitsaki, Evangelia Vanezi

Open Source Software (OSS) is widely used and carries licenses that indicate
the terms under which the software is provided for use, also specifying
modification and distribution rules. Ensuring that users are respecting OSS
license terms when creating derivative works is a complex process. Compliance
issues arising from incompatibilities among licenses may lead to legal
disputes. At the same time, the blockchain technology with immutable entries
offers a mechanism to provide transparency when it comes to licensing and
ensure software changes are recorded. In this work, we are introducing an
integration of blockchain and license management when creating derivative
works, in order to tackle the issue of OSS license compatibility. We have
designed, implemented and performed a preliminary evaluation of FOSS-chain, a
web platform that uses blockchain and automates the license compliance process,
covering 14 OSS licenses. We have evaluated the initial prototype version of
the FOSS-chain platform via a small scale user study. Our preliminary results
are promising, demonstrating the potential of the platform for adaptation on
realistic software systems.

### 3. [ARENA: A tool for measuring and analysing the energy efficiency of Android apps](http://arxiv.org/pdf/2510.01754v1)

Authors: Hina Anwar

To build energy-efficient apps, there is a need to estimate and analyze their
energy consumption in typical usage scenarios. The energy consumption of
Android apps could be estimated via software-based and hardware-based
approaches. Software-based approaches, while easier to implement, are not as
accurate as hardware-based approaches. The process of measuring the energy
consumption of an Android app via a hardware-based approach typically involves
1) setting up a measurement environment, 2) executing the app under test on a
mobile device, 3) recording current/voltage data via a hardware device to
measure energy consumption, and 4) cleaning and aggregating data for analyses,
reports, and visualizations. Specialized scripts are written for selected
hardware and software components to ensure reliable energy measurements. The
energy measurement process is repeated many times and aggregated to remove
noise. These steps make the hardware-based energy measurement process
time-consuming and not easy to adapt or reproduce. There is a lack of
open-source tools available for developers and researchers to take reliable
energy measurements via hardware devices. In this paper, we present and
demonstrate ARENA, a support tool that enables developers and researchers to
connect to a physical measurement device without leaving the comfort of their
IDE. Developers could use ARENA during development to compare energy
consumption between different apps or versions of the same app. ARENA
calculates energy consumption on an Android smartphone by executing a test
scenario on the app under development. Further, ARENA helps aggregate,
statistically analyze, report, and visualize the data, allowing developers and
researchers to dig into the data directly or visually. We implemented ARENA as
an IntelliJ and Android Studio plugin.

### 4. [Towards Speeding up Program Repair with Non-Autoregressive Model](http://arxiv.org/pdf/2510.01825v1)

Authors: Zhenyu Yang, Yue Pan, Zhen Yang, Zhongxing Yu

Enlightened by the success of machine learning techniques in various
application areas, recent years have witnessed a surge of research efforts on
automatic program repair (APR) using machine learning techniques. Previous
machine learning-based APR techniques essentially modified bugs in the
autoregressive (AR) manner, which predicts future values based on past values.
Due to the manner of token-by-token generation, the AR-based APR technique has
a huge time delay. In particular, the delay of the APR model with a large
number of parameters is more serious. To address the issue, we aim to apply the
non-autoregressive (NAR) method to the APR task, which can output target code
in a parallel manner to avoid huge repair delays. However, the naive use of the
NAR manner for the APR task suffers from the issue of compromised patch
quality. To effectively adapt the NAR manner for the APR task, we in this paper
propose NARRepair, the first customized NAR code generation model for the APR
task. The NARRepair model features three major novelties, including 1) the
repair action predictor for alleviating the over-correction issue, 2) the
inter-token dependency extractor for alleviating the issue of lacking
inter-token dependency information, and 3) the two-stage decoder for
alleviating the issue of lacking contextual information. We evaluated NARRepair
on three widely used datasets in the APR community, and the results show that
1) compared to other APR techniques, the NARRepair model has the best
performance within the limited repair time, and 2) compared to AR-based APR
techniques, the repair speed of NARRepair has been increased by 1.4-6.4 times
in the GPU environment. Overall, the results show that NARRepair has achieved
state-of-the-art comprehensive performance in terms of repair speed and
accuracy.

### 5. [RefFilter: Improving Semantic Conflict Detection via Refactoring-Aware Static Analysis](http://arxiv.org/pdf/2510.01960v1)

Authors: Victor Lira, Paulo Borba, Rodrigo Bonifácio, Galileu Santos e Matheus barbosa

Detecting semantic interference remains a challenge in collaborative software
development. Recent lightweight static analysis techniques improve efficiency
over SDG-based methods, but they still suffer from a high rate of false
positives. A key cause of these false positives is the presence of
behavior-preserving code refactorings, which current techniques cannot
effectively distinguish from changes that impact behavior and can interfere
with others. To handle this problem we present RefFilter, a refactoring-aware
tool for semantic interference detection. It builds on existing static
techniques by incorporating automated refactoring detection to improve
precision. RefFilter discards behavior-preserving refactorings from reports,
reducing false positives while preserving detection coverage. To evaluate
effectiveness and scalability, use two datasets: a labeled dataset with 99
scenarios and ground truth, and a novel dataset of 1,087 diverse merge
scenarios that we have built. Experimental results show that RefFilter reduces
false positives by nearly 32% on the labeled dataset. While this reduction
comes with a non significant increase in false negatives, the overall gain in
precision significantly outweighs the minor trade-off in recall. These findings
demonstrate that refactoring-aware interference detection is a practical and
effective strategy for improving merge support in modern development workflows.

### 6. [Automatic Generation of Combinatorial Reoptimisation Problem Specifications: A Vision](http://arxiv.org/pdf/2510.02002v1)

Authors: Maximilian Kratz, Steffen Zschaler, Jens Kosiol, Gabriele Taentzer

Once an optimisation problem has been solved, the solution may need
adaptation when contextual factors change. This challenge, also known as
reoptimisation, has been addressed in various problem domains, such as railway
crew rescheduling, nurse rerostering, or aircraft recovery. This requires a
modified problem to be solved again to ensure that the adapted solution is
optimal in the new context. However, the new optimisation problem differs
notably from the original problem: (i) we want to make only minimal changes to
the original solution to minimise the impact; (ii) we may be unable to change
some parts of the original solution (e.g., because they refer to past
allocations); and (iii) we need to derive a change script from the original
solution to the new solution. In this paper, we argue that Model-Driven
Engineering (MDE) - in particular, the use of declarative modelling languages
and model transformations for the high-level specification of optimisation
problems - offers new opportunities for the systematic derivation of
reoptimisation problems from the original optimisation problem specification.
We focus on combinatorial reoptimisation problems and provide an initial
categorisation of changing problems and strategies for deriving the
corresponding reoptimisation specifications. We introduce an initial
proof-of-concept implementation based on the GIPS (Graph-Based (Mixed) Integer
Linear Programming Problem Specification) tool and apply it to an example
resource-allocation problem: the allocation of teaching assistants to teaching
sessions.

### 7. [ACM SIGSOFT SEN Empirical Software Engineering: Introducing Our New Regular Column](http://arxiv.org/pdf/2510.02007v1)

Authors: Justus Bogner, Roberto Verdecchia

From its early foundations in the 1970s, empirical software engineering (ESE)
has evolved into a mature research discipline that embraces a plethora of
different topics, methodologies, and industrial practices. Despite its
remarkable progress, the ESE research field still needs to keep evolving, as
new impediments, shortcoming, and technologies emerge. Research
reproducibility, limited external validity, subjectivity of reviews, and
porting research results to industrial practices are just some examples of the
drivers for improvements to ESE research. Additionally, several facets of ESE
research are not documented very explicitly, which makes it difficult for
newcomers to pick them up. With this new regular ACM SIGSOFT SEN column
(SEN-ESE), we introduce a venue for discussing meta-aspects of ESE research,
ranging from general topics such as the nature and best practices for
replication packages, to more nuanced themes such as statistical methods,
interview transcription tools, and publishing interdisciplinary research. Our
aim for the column is to be a place where we can regularly spark conversations
on ESE topics that might not often be touched upon or are left implicit.
Contributions to this column will be grounded in expert interviews, focus
groups, surveys, and position pieces, with the goal of encouraging reflection
and improvement in how we conduct, communicate, teach, and ultimately improve
ESE research. Finally, we invite feedback from the ESE community on
challenging, controversial, or underexplored topics, as well as suggestions for
voices you would like to hear from. While we cannot promise to act on every
idea, we aim to shape this column around the community interests and are
grateful for all contributions.

### 8. [Towards fairer public transit: Real-time tensor-based multimodal fare evasion and fraud detection](http://arxiv.org/pdf/2510.02165v1)

Authors: Peter Wauyo, Dalia Bwiza, Alain Murara, Edwin Mugume, Eric Umuhoza

This research introduces a multimodal system designed to detect fraud and
fare evasion in public transportation by analyzing closed circuit television
(CCTV) and audio data. The proposed solution uses the Vision Transformer for
Video (ViViT) model for video feature extraction and the Audio Spectrogram
Transformer (AST) for audio analysis. The system implements a Tensor Fusion
Network (TFN) architecture that explicitly models unimodal and bimodal
interactions through a 2-fold Cartesian product. This advanced fusion technique
captures complex cross-modal dynamics between visual behaviors (e.g.,
tailgating,unauthorized access) and audio cues (e.g., fare transaction sounds).
The system was trained and tested on a custom dataset, achieving an accuracy of
89.5%, precision of 87.2%, and recall of 84.0% in detecting fraudulent
activities, significantly outperforming early fusion baselines and exceeding
the 75% recall rates typically reported in state-of-the-art transportation
fraud detection systems. Our ablation studies demonstrate that the tensor
fusion approach provides a 7.0% improvement in the F1 score and an 8.8% boost
in recall compared to traditional concatenation methods. The solution supports
real-time detection, enabling public transport operators to reduce revenue
loss, improve passenger safety, and ensure operational compliance.

### 9. [KTBox: A Modular LaTeX Framework for Semantic Color, Structured Highlighting, and Scholarly Communication](http://arxiv.org/pdf/2510.01961v1)

Authors: Bhaskar Mangal, Ashutosh Bhatia, Yashvardhan Sharma, Kamlesh Tiwari, Rashmi Verma

The communication of technical insight in scientific manuscripts often relies
on ad-hoc formatting choices, resulting in inconsistent visual emphasis and
limited portability across document classes. This paper introduces ktbox, a
modular LaTeX framework that unifies semantic color palettes, structured
highlight boxes, taxonomy trees, and author metadata utilities into a coherent
system for scholarly writing. The framework is distributed as a set of
lightweight, namespaced components: ktcolor.sty for semantic palettes,
ktbox.sty for structured highlight and takeaway environments, ktlrtree.sty for
taxonomy trees with fusion and auxiliary annotations, and ktorcid.sty for
ORCID-linked author metadata. Each component is independently usable yet
interoperable, ensuring compatibility with major templates such as IEEEtran,
acmart, iclr conference, and beamer. Key features include auto-numbered
takeaway boxes, wide-format highlights, flexible taxonomy tree visualizations,
and multi-column layouts supporting embedded tables, enumerations, and code
blocks. By adopting a clear separation of concerns and enforcing a consistent
naming convention under the kt namespace, the framework transforms visual
styling from cosmetic add-ons into reproducible, extensible building blocks of
scientific communication, improving clarity, portability, and authoring
efficiency across articles, posters, and presentations.

### 10. [Clarifying Semantics of In-Context Examples for Unit Test Generation](http://arxiv.org/pdf/2510.01994v1)

Authors: Chen Yang, Lin Yang, Ziqi Wang, Dong Wang, Jianyi Zhou, Junjie Chen

Recent advances in large language models (LLMs) have enabled promising
performance in unit test generation through in-context learning (ICL). However,
the quality of in-context examples significantly influences the effectiveness
of generated tests-poorly structured or semantically unclear test examples
often lead to suboptimal outputs. In this paper, we propose CLAST, a novel
technique that systematically refines unit tests to improve their semantic
clarity, thereby enhancing their utility as in-context examples. The approach
decomposes complex tests into logically clearer ones and improves semantic
clarity through a combination of program analysis and LLM-based rewriting. We
evaluated CLAST on four open-source and three industrial projects. The results
demonstrate that CLAST largely outperforms UTgen, the state-of-the-art
refinement technique, in both preserving test effectiveness and enhancing
semantic clarity. Specifically, CLAST fully retains the original effectiveness
of unit tests, while UTgen reduces compilation success rate (CSR), pass rate
(PR), test coverage (Cov), and mutation score (MS) by an average of 12.90%,
35.82%, 4.65%, and 5.07%, respectively. Over 85.33% of participants in our user
study preferred the semantic clarity of CLAST-refined tests. Notably,
incorporating CLAST-refined tests as examples effectively improves ICL-based
unit test generation approaches such as RAGGen and TELPA, resulting in an
average increase of 25.97% in CSR, 28.22% in PR, and 45.99% in Cov for
generated tests, compared to incorporating UTgen-refined tests. The insights
from the follow-up user study not only reinforce CLAST's potential impact in
software testing practice but also illuminate avenues for future research.

### Social and Information Networks

### 1. [Framing Unionization on Facebook: Communication around Representation Elections in the United States](http://arxiv.org/pdf/2510.01757v1)

Authors: Arianna Pera, Veronica Jude, Ceren Budak, Luca Maria Aiello

Digital media have become central to how labor unions communicate, organize,
and sustain collective action. Yet little is known about how unions' online
discourse relates to concrete outcomes such as representation elections. This
study addresses the gap by combining National Labor Relations Board (NLRB)
election data with 158k Facebook posts published by U.S. labor unions between
2015 and 2024. We focused on five discourse frames widely recognized in labor
and social movement communication research: diagnostic (identifying problems),
prognostic (proposing solutions), motivational (mobilizing action), community
(emphasizing solidarity), and engagement (promoting interaction). Using a
fine-tuned RoBERTa classifier, we systematically annotated unions' posts and
analyzed patterns of frame usage around election events. Our findings showed
that diagnostic and community frames dominated union communication overall, but
that frame usage varied substantially across organizations. In election cases
that unions won, communication leading up to the vote showed an increased use
of diagnostic, prognostic, and community frames, followed by a reduction in
prognostic and motivational framing after the event--patterns consistent with
strategic preparation. By contrast, in lost election cases unions showed little
adjustment in their communication, suggesting an absence of tailored
communication strategies. By examining variation in message-level framing, the
study highlights how communication strategies adapt to organizational contexts,
contributing open tools and data and complementing prior research in
understanding digital communication of unions and social movements.

### Systems and Control

### 1. [A Scalable Design Approach to Resilient Architectures for Interconnected Cyber-Physical Systems: Safety Guarantees under Multiple Attacks](http://arxiv.org/pdf/2510.01541v1)

Authors: Eman Badr, Abdullah Al Maruf

Complex, interconnected cyber-physical systems (CPS) are increasingly
prevalent in domains such as power systems. Cyber-resilient architectures have
been proposed to recover compromised cyber components of CPS. Recent works have
studied tuning the recovery times of such architectures to guarantee safety in
single-system settings. Extending these designs to interconnected CPS is more
challenging, since solutions must account for attacks on multiple subsystems
that can occur in any order and potentially infinite possible temporal overlap.
This paper aims to address the aforementioned challenge by developing a
scalable framework to assign resilient architectures and to inform the tuning
of their recovery times. Our approach introduces a scalar index that quantifies
the impact of each subsystem on safety under compromised input. These indices
aggregate linearly across subsystems, enabling scalable analysis under
arbitrary attack orderings and temporal overlaps. We establish a linear
inequality relating each subsystem's index and recovery time that guarantees
safety and guides resilient architecture assignment. We also propose a
segmentation-based approach to strengthen the previously derived conditions. We
then present algorithms to compute the proposed indices and to find a
cost-optimal architecture assignment with a safety guarantee. We validate the
framework through a case study on temperature regulation in interconnected
rooms under different attack scenarios.

### 2. [Stability and Robustness of Time-Varying Opinion Dynamics: A Graph-Theoretic Approach](http://arxiv.org/pdf/2510.01580v1)

Authors: M. Hossein Abedinzadeh, Emrah Akyol

We study the stability of opinion dynamics in the time-varying
Friedkin-Johnsen (TVFJ) model, which captures both persistent individual biases
and adaptive social influence. We introduce two temporal structures, defected
temporal graphs (DTGs) and weakly defected temporal graphs (WDTGs), that serve
as graph-theoretic certificates linking stubborn influence and temporal
connectivity to contraction of the state-transition matrix. Using these tools,
we prove asymptotic stability of TVFJ dynamics under infinitely recurring DTGs,
exponential stability in semi-periodic defected networks, and asymptotic
stability of a trust-based extension under the weaker condition of recurring
WDTGs. We also establish boundedness of the omega-limit set, showing that
long-run opinions remain within the convex hull of innate beliefs, and
characterize the limit set for periodically switching systems via a p-LTI
decomposition with the tight bound that the size of the omega-limit set is at
most p. Finally, we show that exponential stability persists under bounded
perturbations, ensuring robustness in noisy or imperfect networks. These
results unify algebraic contraction tests with interpretable graph-based
reasoning, providing scalable and resilient tools for analyzing opinion
formation in evolving social and human-AI networks.

### 3. [A TSO-DSO Coordination Framework via Analytical Representation and Monetization of PQV-Based Distribution System Flexibility](http://arxiv.org/pdf/2510.01854v1)

Authors: Burak Dindar, Can Berk Saner, Hüseyin Kemal Çakmak, Veit Hagenmeyer

As the role of distribution system (DS) flexibility in transmission system
operator (TSO) network management becomes increasingly vital, data privacy
concerns hinder seamless interoperability. The notion of the feasible operating
region (FOR), defined in the PQ domain, has emerged as a promising
privacy-preserving approach. However, effectively leveraging FOR in TSO
operations remains challenging due to three key factors: its accurate
determination in large-scale, meshed DS networks; its tractable analytical
representation; and its economic valuation. In the present paper, we propose a
novel AC optimal power flow (OPF)-based method to construct a three-dimensional
PQV-FOR, explicitly accounting for voltage variability and diverse
flexibility-providing unit (FPU) characteristics. The construction process
employs a two-stage sampling strategy that combines bounding box projection and
Fibonacci direction techniques to efficiently capture the FOR. We then
introduce an implicit polynomial fitting approach to analytically represent the
FOR. Furthermore, we derive a quadratic cost function over the PQV domain to
monetize the FOR. Thus, the proposed framework enables single-round TSO-DSO
coordination: the DSO provides an analytical FOR and cost model; the TSO
determines operating point at the point of common coupling (PCC) within the
FOR-based AC-OPF; and the DSO computes FPU dispatch by solving its local OPF,
without computationally intensive disaggregation or iterative coordination.
Case studies on meshed DS with up to 533 buses, integrated into TS,
demonstrates the method's efficiency compared to standard AC-OPF. On average,
the proposed approach yields negligible cost deviations of at most 0.058%
across test cases, while reducing computation times by up to 58.11%.

### 4. [Coordinated Car-following Using Distributed MPC](http://arxiv.org/pdf/2510.02010v1)

Authors: Di Shen, Qi Dai, Suzhou Huang

Within the modeling framework of Markov games, we propose a series of
algorithms for coordinated car-following using distributed model predictive
control (DMPC). Instead of tracking prescribed feasible trajectories, driving
policies are solved directly as outcomes of the DMPC optimization given the
driver's perceivable states. The coordinated solutions are derived using the
best response dynamics via iterated self-play, and are facilitated by direct
negotiation using inter-agent or agent-infrastructure communication. These
solutions closely approximate either Nash equilibrium or centralized
optimization. By re-parameterizing the action sequence in DMPC as a curve along
the planning horizon, we are able to systematically reduce the original DMPC to
very efficient grid searches such that the optimal solution to the original
DMPC can be well executed in real-time. Within our modeling framework, it is
natural to cast traffic control problems as mechanism design problems, in which
all agents are endogenized on an equal footing with full incentive
compatibility. We show how traffic efficiency can be dramatically improved
while keeping stop-and-go phantom waves tamed at high vehicle densities. Our
approach can be viewed as an alternative way to formulate coordinated adaptive
cruise control (CACC) without an explicit platooning (or with all vehicles in
the traffic system treated as a single extended platoon). We also address the
issue of linear stability of the associated discrete-time traffic dynamics and
demonstrate why it does not always tell the full story about the traffic
stability.

### 5. [Event-triggered control and communication for single-master multi-slave teleoperation systems with Try-Once-Discard protocol](http://arxiv.org/pdf/2510.02072v1)

Authors: Yuling Li, Chenxi Li, Kun Liu, Jie Dong, Rolf Johansson

Single-master multi-slave (SMMS) teleoperation systems can perform multiple
tasks remotely in a shorter time, cover large-scale areas, and adapt more
easily to single-point failures, thereby effectively encompassing a broader
range of applications. As the number of slave manipulators sharing a
communication network increases, the limitation of communication bandwidth
becomes critical. To alleviate bandwidth usage, the Try-Once-Discard (TOD)
scheduling protocol and event-triggered mechanisms are often employed
separately. In this paper, we combine both strategies to optimize network
bandwidth and energy consumption for SMMS teleoperation systems. Specifically,
we propose event-triggered control and communication schemes for a class of
SMMS teleoperation systems using the TOD scheduling protocol. Considering
dynamic uncertainties, the unavailability of relative velocities, and
time-varying delays, we develop adaptive controllers with virtual observers
based on event-triggered schemes to achieve master-slave synchronization.
Stability criteria for the SMMS teleoperation systems under these
event-triggered control and communication schemes are established,
demonstrating that Zeno behavior is excluded. Finally, experiments are
conducted to validate the effectiveness of the proposed algorithms.

### 6. [Recurrent Control Barrier Functions: A Path Towards Nonparametric Safety Verification](http://arxiv.org/pdf/2510.02127v1)

Authors: Jixian Liu, Enrique Mallada

Ensuring the safety of complex dynamical systems often relies on
Hamilton-Jacobi (HJ) Reachability Analysis or Control Barrier Functions (CBFs).
Both methods require computing a function that characterizes a safe set that
can be made (control) invariant. However, the computational burden of solving
high-dimensional partial differential equations (for HJ Reachability) or
large-scale semidefinite programs (for CBFs) makes finding such functions
challenging. In this paper, we introduce the notion of Recurrent Control
Barrier Functions (RCBFs), a novel class of CBFs that leverages a recurrent
property of the trajectories, i.e., coming back to a safe set, for safety
verification. Under mild assumptions, we show that the RCBF condition holds for
the signed-distance function, turning function design into set identification.
Notably, the resulting set need not be invariant to certify safety. We further
propose a data-driven nonparametric method to compute safe sets that is
massively parallelizable and trades off conservativeness against computational
cost.

### 7. [Detection and Identification of Sensor Attacks Using Data](http://arxiv.org/pdf/2510.02183v1)

Authors: Takumi Shinohara, Karl H. Johansson, Henrik Sandberg

In this paper, we investigate data-driven attack detection and identification
in a model-free setting. Unlike existing studies, we consider the case where
the available output data include malicious false-data injections. We aim to
detect and identify such attacks solely from the compromised data. We address
this problem in two scenarios: (1) when the system operator is aware of the
system's sparse observability condition, and (2) when the data are partially
clean (i.e., attack-free). In both scenarios, we derive conditions and
algorithms for detecting and identifying attacks using only the compromised
data. Finally, we demonstrate the effectiveness of the proposed framework via
numerical simulations on a three-inertia system.

### 8. [Game-theoretic Social Distancing in Competitive Bi-Virus SIS Epidemics](http://arxiv.org/pdf/2510.02269v1)

Authors: Benjamin Catalano, Keith Paarporn, Sebin Gracy

Numerous elements drive the spread of infectious diseases in complex
real-world networks. Of particular interest is social behaviors that evolve in
tandem with the spread of disease. Moreover, recent studies highlight the
importance of understanding how multiple strains spread simultaneously through
a population (e.g. Delta and Omicron variants of SARS-CoV-2). In this paper, we
propose a bi-virus SIS epidemic model coupled with a game-theoretic social
distancing behavior model. The behaviors are governed by replicator equations
from evolutionary game theory. The prevalence of each strain impacts the choice
of an individual to social distance, and, in turn, their behavior affects the
spread of each virus in the SIS model. Our analysis identifies equilibria of
the system and their local stability properties, which reveal several isolated
fixed points with varying levels of social distancing. We find that endemic
co-existence is possible only when the reproduction numbers of both strains are
equal. Assuming the reproduction number for each virus is the same, we identify
suitable parameter regimes that give rise to lines of coexistence equilibria.
Moreover, we also identify conditions for local exponential stability of said
lines of equilibria. We illustrate our findings with several numerical
simulations.

### 9. [Bi-Virus SIS Epidemic Propagation under Mutation and Game-theoretic Protection Adoption](http://arxiv.org/pdf/2510.01570v1)

Authors: Urmee Maitra, Ashish R. Hota, Vaibhav Srivastava

We study a bi-virus susceptible-infected-susceptible (SIS) epidemic model in
which individuals are either susceptible or infected with one of two virus
strains, and consider mutation-driven transitions between strains. The general
case of bi-directional mutation is first analyzed, where we characterize the
disease-free equilibrium and establish its global asymptotic stability, as well
as the existence, uniqueness, and stability of an endemic equilibrium. We then
present a game-theoretic framework where susceptible individuals strategically
choose whether to adopt protection or remain unprotected, to maximize their
instantaneous payoffs. We derive Nash strategies under bi-directional mutation,
and subsequently consider the special case of unidirectional mutation. In the
latter case, we show that coexistence of both strains is impossible when
mutation occurs from the strain with lower reproduction number and transmission
rate to the other strain. Furthermore, we fully characterize the stationary
Nash equilibrium (SNE) in the setting permitting coexistence, and examine how
mutation rates influence protection adoption and infection prevalence at the
SNE. Numerical simulations corroborate the analytical results, demonstrating
that infection levels decrease monotonically with higher protection adoption,
and highlight the impact of mutation rates and protection cost on infection
state trajectories.

### 10. [Geometric Backstepping Control of Omnidirectional Tiltrotors Incorporating Servo-Rotor Dynamics for Robustness against Sudden Disturbances](http://arxiv.org/pdf/2510.01675v1)

Authors: Jaewoo Lee, Dongjae Lee, Jinwoo Lee, Hyungyu Lee, Yeonjoon Kim, H. Jin Kim

This work presents a geometric backstepping controller for a variable-tilt
omnidirectional multirotor that explicitly accounts for both servo and rotor
dynamics. Considering actuator dynamics is essential for more effective and
reliable operation, particularly during aggressive flight maneuvers or recovery
from sudden disturbances. While prior studies have investigated actuator-aware
control for conventional and fixed-tilt multirotors, these approaches rely on
linear relationships between actuator input and wrench, which cannot capture
the nonlinearities induced by variable tilt angles. In this work, we exploit
the cascade structure between the rigid-body dynamics of the multirotor and its
nonlinear actuator dynamics to design the proposed backstepping controller and
establish exponential stability of the overall system. Furthermore, we reveal
parametric uncertainty in the actuator model through experiments, and we
demonstrate that the proposed controller remains robust against such
uncertainty. The controller was compared against a baseline that does not
account for actuator dynamics across three experimental scenarios: fast
translational tracking, rapid rotational tracking, and recovery from sudden
disturbance. The proposed method consistently achieved better tracking
performance, and notably, while the baseline diverged and crashed during the
fastest translational trajectory tracking and the recovery experiment, the
proposed controller maintained stability and successfully completed the tasks,
thereby demonstrating its effectiveness.

### Machine Learning (Statistics Category)

### 1. [AI Foundation Model for Time Series with Innovations Representation](http://arxiv.org/pdf/2510.01560v1)

Authors: Lang Tong, Xinyi Wang

This paper introduces an Artificial Intelligence (AI) foundation model for
time series in engineering applications, where causal operations are required
for real-time monitoring and control. Since engineering time series are
governed by physical, rather than linguistic, laws, large-language-model-based
AI foundation models may be ineffective or inefficient. Building on the
classical innovations representation theory of Wiener, Kallianpur, and
Rosenblatt, we propose Time Series GPT (TS-GPT) -- an
innovations-representation-based Generative Pre-trained Transformer for
engineering monitoring and control. As an example of foundation model
adaptation, we consider Probabilistic Generative Forecasting, which produces
future time series samples from conditional probability distributions given
past realizations. We demonstrate the effectiveness of TS-GPT in forecasting
real-time locational marginal prices using historical data from U.S.
independent system operators.

### 2. [A reproducible comparative study of categorical kernels for Gaussian process regression, with new clustering-based nested kernels](http://arxiv.org/pdf/2510.01840v1)

Authors: Raphaël Carpintero Perez, Sébastien Da Veiga, Josselin Garnier

Designing categorical kernels is a major challenge for Gaussian process
regression with continuous and categorical inputs. Despite previous studies, it
is difficult to identify a preferred method, either because the evaluation
metrics, the optimization procedure, or the datasets change depending on the
study. In particular, reproducible code is rarely available. The aim of this
paper is to provide a reproducible comparative study of all existing
categorical kernels on many of the test cases investigated so far. We also
propose new evaluation metrics inspired by the optimization community, which
provide quantitative rankings of the methods across several tasks. From our
results on datasets which exhibit a group structure on the levels of
categorical inputs, it appears that nested kernels methods clearly outperform
all competitors. When the group structure is unknown or when there is no prior
knowledge of such a structure, we propose a new clustering-based strategy using
target encodings of categorical variables. We show that on a large panel of
datasets, which do not necessarily have a known group structure, this
estimation strategy still outperforms other approaches while maintaining low
computational cost.

### 3. [Deep Hedging Under Non-Convexity: Limitations and a Case for AlphaZero](http://arxiv.org/pdf/2510.01874v1)

Authors: Matteo Maggiolo, Giuseppe Nuti, Miroslav Štrupl, Oleg Szehr

This paper examines replication portfolio construction in incomplete markets
- a key problem in financial engineering with applications in pricing, hedging,
balance sheet management, and energy storage planning. We model this as a
two-player game between an investor and the market, where the investor makes
strategic bets on future states while the market reveals outcomes. Inspired by
the success of Monte Carlo Tree Search in stochastic games, we introduce an
AlphaZero-based system and compare its performance to deep hedging - a widely
used industry method based on gradient descent. Through theoretical analysis
and experiments, we show that deep hedging struggles in environments where the
$Q$-function is not subject to convexity constraints - such as those involving
non-convex transaction costs, capital constraints, or regulatory limitations -
converging to local optima. We construct specific market environments to
highlight these limitations and demonstrate that AlphaZero consistently finds
near-optimal replication strategies. On the theoretical side, we establish a
connection between deep hedging and convex optimization, suggesting that its
effectiveness is contingent on convexity assumptions. Our experiments further
suggest that AlphaZero is more sample-efficient - an important advantage in
data-scarce, overfitting-prone derivative markets.

### 4. [Predictively Oriented Posteriors](http://arxiv.org/pdf/2510.01915v1)

Authors: Yann McLatchie, Badr-Eddine Cherief-Abdellatif, David T. Frazier, Jeremias Knoblauch

We advocate for a new statistical principle that combines the most desirable
aspects of both parameter inference and density estimation. This leads us to
the predictively oriented (PrO) posterior, which expresses uncertainty as a
consequence of predictive ability. Doing so leads to inferences which
predictively dominate both classical and generalised Bayes posterior predictive
distributions: up to logarithmic factors, PrO posteriors converge to the
predictively optimal model average at rate $n^{-1/2}$. Whereas classical and
generalised Bayes posteriors only achieve this rate if the model can recover
the data-generating process, PrO posteriors adapt to the level of model
misspecification. This means that they concentrate around the true model at
rate $n^{1/2}$ in the same way as Bayes and Gibbs posteriors if the model can
recover the data-generating distribution, but do \textit{not} concentrate in
the presence of non-trivial forms of model misspecification. Instead, they
stabilise towards a predictively optimal posterior whose degree of irreducible
uncertainty admits an interpretation as the degree of model misspecification --
a sharp contrast to how Bayesian uncertainty and its existing extensions
behave. Lastly, we show that PrO posteriors can be sampled from by evolving
particles based on mean field Langevin dynamics, and verify the practical
significance of our theoretical developments on a number of numerical examples.

### 5. [Uniform-in-time convergence bounds for Persistent Contrastive Divergence Algorithms](http://arxiv.org/pdf/2510.01944v1)

Authors: Paul Felix Valsecchi Oliva, O. Deniz Akyildiz, Andrew Duncan

We propose a continuous-time formulation of persistent contrastive divergence
(PCD) for maximum likelihood estimation (MLE) of unnormalised densities. Our
approach expresses PCD as a coupled, multiscale system of stochastic
differential equations (SDEs), which perform optimisation of the parameter and
sampling of the associated parametrised density, simultaneously.
  From this novel formulation, we are able to derive explicit bounds for the
error between the PCD iterates and the MLE solution for the model parameter.
This is made possible by deriving uniform-in-time (UiT) bounds for the
difference in moments between the multiscale system and the averaged regime. An
efficient implementation of the continuous-time scheme is introduced,
leveraging a class of explicit, stable intregators, stochastic orthogonal
Runge-Kutta Chebyshev (S-ROCK), for which we provide explicit error estimates
in the long-time regime. This leads to a novel method for training energy-based
models (EBMs) with explicit error guarantees.

### 6. [Adaptive Heterogeneous Mixtures of Normalising Flows for Robust Variational Inference](http://arxiv.org/pdf/2510.02056v1)

Authors: Benjamin Wiriyapong, Oktay Karakuş, Kirill Sidorov

Normalising-flow variational inference (VI) can approximate complex
posteriors, yet single-flow models often behave inconsistently across
qualitatively different distributions. We propose Adaptive Mixture Flow
Variational Inference (AMF-VI), a heterogeneous mixture of complementary flows
(MAF, RealNVP, RBIG) trained in two stages: (i) sequential expert training of
individual flows, and (ii) adaptive global weight estimation via
likelihood-driven updates, without per-sample gating or architectural changes.
Evaluated on six canonical posterior families of banana, X-shape, two-moons,
rings, a bimodal, and a five-mode mixture, AMF-VI achieves consistently lower
negative log-likelihood than each single-flow baseline and delivers stable
gains in transport metrics (Wasserstein-2) and maximum mean discrepancy (MDD),
indicating improved robustness across shapes and modalities. The procedure is
efficient and architecture-agnostic, incurring minimal overhead relative to
standard flow training, and demonstrates that adaptive mixtures of diverse
flows provide a reliable route to robust VI across diverse posterior families
whilst preserving each expert's inductive bias.

### 7. [Adaptive Kernel Selection for Stein Variational Gradient Descent](http://arxiv.org/pdf/2510.02067v1)

Authors: Moritz Melcher, Simon Weissmann, Ashia C. Wilson, Jakob Zech

A central challenge in Bayesian inference is efficiently approximating
posterior distributions. Stein Variational Gradient Descent (SVGD) is a popular
variational inference method which transports a set of particles to approximate
a target distribution. The SVGD dynamics are governed by a reproducing kernel
Hilbert space (RKHS) and are highly sensitive to the choice of the kernel
function, which directly influences both convergence and approximation quality.
The commonly used median heuristic offers a simple approach for setting kernel
bandwidths but lacks flexibility and often performs poorly, particularly in
high-dimensional settings. In this work, we propose an alternative strategy for
adaptively choosing kernel parameters over an abstract family of kernels.
Recent convergence analyses based on the kernelized Stein discrepancy (KSD)
suggest that optimizing the kernel parameters by maximizing the KSD can improve
performance. Building on this insight, we introduce Adaptive SVGD (Ad-SVGD), a
method that alternates between updating the particles via SVGD and adaptively
tuning kernel bandwidths through gradient ascent on the KSD. We provide a
simplified theoretical analysis that extends existing results on minimizing the
KSD for fixed kernels to our adaptive setting, showing convergence properties
for the maximal KSD over our kernel class. Our empirical results further
support this intuition: Ad-SVGD consistently outperforms standard heuristics in
a variety of tasks.

### 8. [Hybrid Physics-ML Framework for Pan-Arctic Permafrost Infrastructure Risk at Record 2.9-Million Observation Scale](http://arxiv.org/pdf/2510.02189v1)

Authors: Boris Kriuk

Arctic warming threatens over 100 billion in permafrost-dependent
infrastructure across Northern territories, yet existing risk assessment
frameworks lack spatiotemporal validation, uncertainty quantification, and
operational decision-support capabilities. We present a hybrid physics-machine
learning framework integrating 2.9 million observations from 171,605 locations
(2005-2021) combining permafrost fraction data with climate reanalysis. Our
stacked ensemble model (Random Forest + Histogram Gradient Boosting + Elastic
Net) achieves R2=0.980 (RMSE=5.01 pp) with rigorous spatiotemporal
cross-validation preventing data leakage. To address machine learning
limitations in extrapolative climate scenarios, we develop a hybrid approach
combining learned climate-permafrost relationships (60%) with physical
permafrost sensitivity models (40%, -10 pp/C). Under RCP8.5 forcing (+5C over
10 years), we project mean permafrost fraction decline of -20.3 pp (median:
-20.0 pp), with 51.5% of Arctic Russia experiencing over 20 percentage point
loss. Infrastructure risk classification identifies 15% high-risk zones (25%
medium-risk) with spatially explicit uncertainty maps. Our framework represents
the largest validated permafrost ML dataset globally, provides the first
operational hybrid physics-ML forecasting system for Arctic infrastructure, and
delivers open-source tools enabling probabilistic permafrost projections for
engineering design codes and climate adaptation planning. The methodology is
generalizable to other permafrost regions and demonstrates how hybrid
approaches can overcome pure data-driven limitations in climate change
applications.

### 9. [Efficiently Generating Correlated Sample Paths from Multi-step Time Series Foundation Models](http://arxiv.org/pdf/2510.02224v1)

Authors: Ethan Baron, Boris Oreshkin, Ruijun Ma, Hanyu Zhang, Kari Torkkola, Michael W. Mahoney, Andrew Gordon Wilson, Tatiana Konstantinova

Many time series applications require access to multi-step forecast
trajectories in the form of sample paths. Recently, time series foundation
models have leveraged multi-step lookahead predictions to improve the quality
and efficiency of multi-step forecasts. However, these models only predict
independent marginal distributions for each time step, rather than a full joint
predictive distribution. To generate forecast sample paths with realistic
correlation structures, one typically resorts to autoregressive sampling, which
can be extremely expensive. In this paper, we present a copula-based approach
to efficiently generate accurate, correlated sample paths from existing
multi-step time series foundation models in one forward pass. Our copula-based
approach generates correlated sample paths orders of magnitude faster than
autoregressive sampling, and it yields improved sample path quality by
mitigating the snowballing error phenomenon.

### 10. [Median2Median: Zero-shot Suppression of Structured Noise in Images](http://arxiv.org/pdf/2510.01666v1)

Authors: Jianxu Wang, Ge Wang

Image denoising is a fundamental problem in computer vision and medical
imaging. However, real-world images are often degraded by structured noise with
strong anisotropic correlations that existing methods struggle to remove. Most
data-driven approaches rely on large datasets with high-quality labels and
still suffer from limited generalizability, whereas existing zero-shot methods
avoid this limitation but remain effective only for independent and identically
distributed (i.i.d.) noise. To address this gap, we propose Median2Median
(M2M), a zero-shot denoising framework designed for structured noise. M2M
introduces a novel sampling strategy that generates pseudo-independent
sub-image pairs from a single noisy input. This strategy leverages directional
interpolation and generalized median filtering to adaptively exclude values
distorted by structured artifacts. To further enlarge the effective sampling
space and eliminate systematic bias, a randomized assignment strategy is
employed, ensuring that the sampled sub-image pairs are suitable for
Noise2Noise training. In our realistic simulation studies, M2M performs on par
with state-of-the-art zero-shot methods under i.i.d. noise, while consistently
outperforming them under correlated noise. These findings establish M2M as an
efficient, data-free solution for structured noise suppression and mark the
first step toward effective zero-shot denoising beyond the strict i.i.d.
assumption.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-03 PST.

### 1. [Automated assessment and detection of third molar and inferior alveolar nerve relations using UNet and transfer learning models](https://www.nature.com/articles/s41598-025-17753-0)

Authors: Ahmad F. Klaib et al.

### 2. [Interactive digital twins enabling responsible extended reality applications](https://www.nature.com/articles/s41598-025-17855-9)

Authors: Maya Antoun et al.

### 3. [Deep transfer learning approach for the classification of single and multiple power quality disturbances](https://www.nature.com/articles/s41598-025-18064-0)

Authors: Uvesh Sipai et al.

### 4. [Explainability and importance estimate of time series classifier via embedded neural network](https://www.nature.com/articles/s41598-025-17703-w)

Authors: Ho Tung Jeremy Chan et al.

### 5. [Network attack knowledge inference with graph convolutional networks and convolutional 2D KG embeddings](https://www.nature.com/articles/s41598-025-17941-y)

Authors: Weiwu Ren et al.

### 6. [Diagnostics of diabetic retinopathy based on fundus photos using machine learning methods with advanced feature engineering algorithms](https://www.nature.com/articles/s41598-025-06973-z)

Authors: Michał Gandor et al.

### 7. [Exploiting deep transfer learning based precise classification and grading of renal cell carcinoma using histopathological images](https://www.nature.com/articles/s41598-025-17923-0)

Authors: Ala Saleh Alluhaidan et al.

