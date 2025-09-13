# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-12 17:00:25.944241 PST.

### Artificial Intelligence

### 1. [Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users](http://arxiv.org/pdf/2509.09066v1)

Authors: Haowei Yang, Yushang Zhao, Sitao Min, Bo Su, Chao Yao, Wei Xu

The cold-start user issue further compromises the effectiveness of
recommender systems in limiting access to the historical behavioral
information. It is an effective pipeline to optimize instructional prompts on a
few-shot large language model (LLM) used in recommender tasks. We introduce a
context-conditioned prompt formulation method P(u,\ Ds)\ \rightarrow\
R\widehat, where u is a cold-start user profile, Ds is a curated support set,
and R\widehat is the predicted ranked list of items. Based on systematic
experimentation with transformer-based autoregressive LLMs (BioGPT, LLaMA-2,
GPT-4), we provide empirical evidence that optimal exemplar injection and
instruction structuring can significantly improve the precision@k and NDCG
scores of such models in low-data settings. The pipeline uses token-level
alignments and embedding space regularization with a greater semantic fidelity.
Our findings not only show that timely composition is not merely syntactic but
also functional as it is in direct control of attention scales and decoder
conduct through inference. This paper shows that prompt-based adaptation may be
considered one of the ways to address cold-start recommendation issues in
LLM-based pipelines.

### 2. [Anti-Money Laundering Machine Learning Pipelines; A Technical Analysis on Identifying High-risk Bank Clients with Supervised Learning](http://arxiv.org/pdf/2509.09127v1)

Authors: Khashayar Namdar, Pin-Chien Wang, Tushar Raju, Steven Zheng, Fiona Li, Safwat Tahmin Khan

Anti-money laundering (AML) actions and measurements are among the priorities
of financial institutions, for which machine learning (ML) has shown to have a
high potential. In this paper, we propose a comprehensive and systematic
approach for developing ML pipelines to identify high-risk bank clients in a
dataset curated for Task 1 of the University of Toronto 2023-2024 Institute for
Management and Innovation (IMI) Big Data and Artificial Intelligence
Competition. The dataset included 195,789 customer IDs, and we employed a
16-step design and statistical analysis to ensure the final pipeline was
robust. We also framed the data in a SQLite database, developed SQL-based
feature engineering algorithms, connected our pre-trained model to the
database, and made it inference-ready, and provided explainable artificial
intelligence (XAI) modules to derive feature importance. Our pipeline achieved
a mean area under the receiver operating characteristic curve (AUROC) of 0.961
with a standard deviation (SD) of 0.005. The proposed pipeline achieved second
place in the competition.

### 3. [Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](http://arxiv.org/pdf/2509.09245v1)

Authors: Shuocheng Li, Yihao Liu, Silin Du, Wenxuan Zeng, Zhe Xu, Mengyu Zhou, Yeye He, Haoyu Dong, Shi Han, Dongmei Zhang

Large language models (LLMs) have shown great promise in automating data
science workflows, but existing models still struggle with multi-step reasoning
and tool use, which limits their effectiveness on complex data analysis tasks.
To address this, we propose a scalable pipeline that extracts high-quality,
tool-based data analysis tasks and their executable multi-step solutions from
real-world Jupyter notebooks and associated data files. Using this pipeline, we
introduce NbQA, a large-scale dataset of standardized task-solution pairs that
reflect authentic tool-use patterns in practical data science scenarios. To
further enhance multi-step reasoning, we present Jupiter, a framework that
formulates data analysis as a search problem and applies Monte Carlo Tree
Search (MCTS) to generate diverse solution trajectories for value model
learning. During inference, Jupiter combines the value model and node visit
counts to efficiently collect executable multi-step plans with minimal search
steps. Experimental results show that Qwen2.5-7B and 14B-Instruct models on
NbQA solve 77.82% and 86.38% of tasks on InfiAgent-DABench,
respectively-matching or surpassing GPT-4o and advanced agent frameworks.
Further evaluations demonstrate improved generalization and stronger tool-use
reasoning across diverse multi-step reasoning tasks.

### 4. [Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs](http://arxiv.org/pdf/2509.09272v1)

Authors: Vaibhav Chaudhary, Neha Soni, Narotam Singh, Amita Kapoor

Knowledge graphs, a powerful tool for structuring information through
relational triplets, have recently become the new front-runner in enhancing
question-answering systems. While traditional Retrieval Augmented Generation
(RAG) approaches are proficient in fact-based and local context-based
extraction from concise texts, they encounter limitations when addressing the
thematic and holistic understanding of complex, extensive texts, requiring a
deeper analysis of both text and context. This paper presents a comprehensive
technical comparative study of three different methodologies for constructing
knowledge graph triplets and integrating them with Large Language Models (LLMs)
for question answering: spaCy, Stanford CoreNLP-OpenIE, and GraphRAG, all
leveraging open source technologies. We evaluate the effectiveness,
feasibility, and adaptability of these methods by analyzing their capabilities,
state of development, and their impact on the performance of LLM-based question
answering. Experimental results indicate that while OpenIE provides the most
comprehensive coverage of triplets, GraphRAG demonstrates superior reasoning
abilities among the three. We conclude with a discussion on the strengths and
limitations of each method and provide insights into future directions for
improving knowledge graph-based question answering.

### 5. [LightAgent: Production-level Open-source Agentic AI Framework](http://arxiv.org/pdf/2509.09292v1)

Authors: Weige Cai, Tong Zhu, Jinyi Niu, Ruiqi Hu, Lingyao Li, Tenglong Wang, Xiaowu Dai, Weining Shen, Liwen Zhang

With the rapid advancement of large language models (LLMs), Multi-agent
Systems (MAS) have achieved significant progress in various application
scenarios. However, substantial challenges remain in designing versatile,
robust, and efficient platforms for agent deployment. To address these
limitations, we propose \textbf{LightAgent}, a lightweight yet powerful agentic
framework, effectively resolving the trade-off between flexibility and
simplicity found in existing frameworks. LightAgent integrates core
functionalities such as Memory (mem0), Tools, and Tree of Thought (ToT), while
maintaining an extremely lightweight structure. As a fully open-source
solution, it seamlessly integrates with mainstream chat platforms, enabling
developers to easily build self-learning agents. We have released LightAgent at
\href{https://github.com/wxai-space/LightAgent}{https://github.com/wxai-space/LightAgent}

### 6. [Explaining Tournament Solutions with Minimal Supports](http://arxiv.org/pdf/2509.09312v1)

Authors: Clément Contet, Umberto Grandi, Jérôme Mengin

Tournaments are widely used models to represent pairwise dominance between
candidates, alternatives, or teams. We study the problem of providing certified
explanations for why a candidate appears among the winners under various
tournament rules. To this end, we identify minimal supports, minimal
sub-tournaments in which the candidate is guaranteed to win regardless of how
the rest of the tournament is completed (that is, the candidate is a necessary
winner of the sub-tournament). This notion corresponds to an abductive
explanation for the question,"Why does the winner win the tournament", a
central concept in formal explainable AI. We focus on common tournament
solutions: the top cycle, the uncovered set, the Copeland rule, the Borda rule,
the maximin rule, and the weighted uncovered set. For each rule we determine
the size of the smallest minimal supports, and we present polynomial-time
algorithms to compute them for all but the weighted uncovered set, for which
the problem is NP-complete. Finally, we show how minimal supports can serve to
produce compact, certified, and intuitive explanations.

### 7. [Towards Adaptive ML Benchmarks: Web-Agent-Driven Construction, Domain Expansion, and Metric Optimization](http://arxiv.org/pdf/2509.09321v1)

Authors: Hangyi Jia, Yuxi Qian, Hanwen Tong, Xinhui Wu, Lin Chen, Feng Wei

Recent advances in large language models (LLMs) have enabled the emergence of
general-purpose agents for automating end-to-end machine learning (ML)
workflows, including data analysis, feature engineering, model training, and
competition solving. However, existing benchmarks remain limited in task
coverage, domain diversity, difficulty modeling, and evaluation rigor, failing
to capture the full capabilities of such agents in realistic settings. We
present TAM Bench, a diverse, realistic, and structured benchmark for
evaluating LLM-based agents on end-to-end ML tasks. TAM Bench features three
key innovations: (1) A browser automation and LLM-based task acquisition system
that automatically collects and structures ML challenges from platforms such as
Kaggle, AIcrowd, and Biendata, spanning multiple task types and data modalities
(e.g., tabular, text, image, graph, audio); (2) A leaderboard-driven difficulty
modeling mechanism that estimates task complexity using participant counts and
score dispersion, enabling scalable and objective task calibration; (3) A
multi-dimensional evaluation framework incorporating performance, format
compliance, constraint adherence, and task generalization. Based on 150 curated
AutoML tasks, we construct three benchmark subsets of different sizes -- Lite,
Medium, and Full -- designed for varying evaluation scenarios. The Lite
version, with 18 tasks and balanced coverage across modalities and difficulty
levels, serves as a practical testbed for daily benchmarking and comparative
studies.

### 8. [TORSO: Template-Oriented Reasoning Towards General Tasks](http://arxiv.org/pdf/2509.09448v1)

Authors: Minhyuk Kim, Seungyoon Lee, Heuiseok Lim

The approaches that guide Large Language Models (LLMs) to emulate human
reasoning during response generation have emerged as an effective method for
enabling them to solve complex problems in a step-by-step manner, thereby
achieving superior performance. However, most existing approaches using
few-shot prompts to generate responses heavily depend on the provided examples,
limiting the utilization of the model's inherent reasoning capabilities.
Moreover, constructing task-specific few-shot prompts is often costly and may
lead to inconsistencies across different tasks. In this work, we introduce
Template-Oriented Reasoning (TORSO), which elicits the model to utilize
internal reasoning abilities to generate proper responses across various tasks
without the need for manually crafted few-shot examples. Our experimental
results demonstrate that TORSO achieves strong performance on diverse LLMs
benchmarks with reasonable rationales.

### 9. [Inteligencia Artificial jurídica y el desafío de la veracidad: análisis de alucinaciones, optimización de RAG y principios para una integración responsable](http://arxiv.org/pdf/2509.09467v1)

Authors: Alex Dantart

This technical report analyzes the challenge of "hallucinations" (false
information) in LLMs applied to law. It examines their causes, manifestations,
and the effectiveness of the RAG mitigation strategy, highlighting its
limitations and proposing holistic optimizations. The paper explores the
ethical and regulatory implications, emphasizing human oversight as an
irreplaceable role. It concludes that the solution lies not in incrementally
improving generative models, but in adopting a "consultative" AI paradigm that
prioritizes veracity and traceability, acting as a tool to amplify, not
replace, professional judgment.
  --
  Este informe t\'ecnico analiza el desaf\'io de las "alucinaciones"
(informaci\'on falsa) en los LLMs aplicados al derecho. Se examinan sus causas,
manifestaciones y la efectividad de la estrategia de mitigaci\'on RAG,
exponiendo sus limitaciones y proponiendo optimizaciones hol\'isticas. Se
exploran las implicaciones \'eticas y regulatorias, enfatizando la
supervisi\'on humana como un rol insustituible. El documento concluye que la
soluci\'on no reside en mejorar incrementalmente los modelos generativos, sino
en adoptar un paradigma de IA "consultiva" que priorice la veracidad y la
trazabilidad, actuando como una herramienta para amplificar, y no sustituir, el
juicio profesional.

### 10. [SEDM: Scalable Self-Evolving Distributed Memory for Agents](http://arxiv.org/pdf/2509.09498v1)

Authors: Haoran Xu, Jiacong Hu, Ke Zhang, Lei Yu, Yuxin Tang, Xinyuan Song, Yiqun Duan, Lynn Ai, Bill Shi

Long-term multi-agent systems inevitably generate vast amounts of
trajectories and historical interactions, which makes efficient memory
management essential for both performance and scalability. Existing methods
typically depend on vector retrieval and hierarchical storage, yet they are
prone to noise accumulation, uncontrolled memory expansion, and limited
generalization across domains. To address these challenges, we present SEDM,
Self-Evolving Distributed Memory, a verifiable and adaptive framework that
transforms memory from a passive repository into an active, self-optimizing
component. SEDM integrates verifiable write admission based on reproducible
replay, a self-scheduling memory controller that dynamically ranks and
consolidates entries according to empirical utility, and cross-domain knowledge
diffusion that abstracts reusable insights to support transfer across
heterogeneous tasks. Evaluations on benchmark datasets demonstrate that SEDM
improves reasoning accuracy while reducing token overhead compared with strong
memory baselines, and further enables knowledge distilled from fact
verification to enhance multi-hop reasoning. The results highlight SEDM as a
scalable and sustainable memory mechanism for open-ended multi-agent
collaboration. The code will be released in the later stage of this project.

### Hardware Architecture

### 1. [Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference](http://arxiv.org/pdf/2509.09505v1)

Authors: Haoran Wu, Can Xiao, Jiayi Nie, Xuan Guo, Binglei Lou, Jeffrey T. H. Wong, Zhiwen Mo, Cheng Zhang, Przemyslaw Forys, Wayne Luk, Hongxiang Fan, Jianyi Cheng, Timothy M. Jones, Rika Antonova, Robert Mullins, Aaron Zhao

LLMs now form the backbone of AI agents for a diverse array of applications,
including tool use, command-line agents, and web or computer use agents. These
agentic LLM inference tasks are fundamentally different from chatbot-focused
inference -- they often have much larger context lengths to capture complex,
prolonged inputs, such as entire webpage DOMs or complicated tool call
trajectories. This, in turn, generates significant off-chip memory traffic for
the underlying hardware at the inference stage and causes the workload to be
constrained by two memory walls, namely the bandwidth and capacity memory
walls, preventing the on-chip compute units from achieving high utilization.
  In this paper, we introduce PLENA, a hardware-software co-designed system
that applies three core optimization pathways to tackle these challenges. PLENA
includes an efficient hardware implementation of compute and memory units
supporting an asymmetric quantization scheme. PLENA also features a novel
flattened systolic array architecture that has native support for
FlashAttention to tackle these memory walls in the scenario of inference
serving for long-context LLMs. Additionally, PLENA is developed with a complete
stack, including a custom ISA, a compiler, a cycle-emulated simulator, and an
automated design space exploration flow. The simulated results show that PLENA
achieves up to 8.5x higher utilization than existing accelerators, and delivers
2.24x higher throughput than the A100 GPU and 3.85x higher throughput than the
TPU v6e, under the same multiplier count and memory settings. The full PLENA
system will also be open-sourced.

### 2. [Implementation of a 8-bit Wallace Tree Multiplier](http://arxiv.org/pdf/2509.09178v1)

Authors: Ayan Biswas, Jimmy Jin

Wallace tree multipliers are a parallel digital multiplier architecture
designed to minimize the worst-case time complexity of the circuit depth
relative to the input size [1]. In particular, it seeks to perform long
multiplication in the binary sense, reducing as many partial products per stage
as possible through full and half adders circuits, achieving O(log(n)) where n
= bit length of input. This paper provides an overview of the design, progress
and methodology in the final project of ECE 55900, consisting of the schematic
and layout of a Wallace tree 8-bit input multiplier on the gpdk45 technology in
Cadence Virtuoso, as well as any design attempts prior to the final product.
This also includes our endeavors in designing the final MAC (Multiply
Accumulate) unit with undefined targets, which we chose to implement as a 16
bit combinational multiply-add.

### Computational Complexity

### 1. [Uniformity within Parameterized Circuit Classes](http://arxiv.org/pdf/2509.09657v1)

Authors: Steef Hegeman, Jan Martens, Alfons Laarman

We study uniformity conditions for parameterized Boolean circuit families.
Uniformity conditions require that the infinitely many circuits in a circuit
family are in some sense easy to construct from one shared description. For
shallow circuit families, logtime-uniformity is often desired but quite
technical to prove. Despite that, proving it is often left as an exercise for
the reader -- even for recently introduced classes in parameterized circuit
complexity, where uniformity conditions have not yet been explicitly studied.
We formally define parameterized versions of linear-uniformity,
logtime-uniformity, and FO-uniformity, and prove that these result in
equivalent complexity classes when imposed on $\text{para-}\textsf{AC}^0$ and
$\text{para-}\textsf{AC}^{0\uparrow}$. Overall, we provide a convenient way to
verify uniformity for shallow parameterized circuit classes, and thereby
substantiate claims of uniformity in the literature.

### Computational Engineering

### 1. [Isogeometric Topology Optimization Based on Topological Derivatives](http://arxiv.org/pdf/2509.09236v1)

Authors: Guilherme Henrique Teixeira, Nepomuk Krenn, Peter Gangl, Benjamin Marussig

Topology optimization is a valuable tool in engineering, facilitating the
design of optimized structures. However, topological changes often require a
remeshing step, which can become challenging. In this work, we propose an
isogeometric approach to topology optimization driven by topological
derivatives. The combination of a level-set method together with an immersed
isogeometric framework allows seamless geometry updates without the necessity
of remeshing. At the same time, topological derivatives provide topological
modifications without the need to define initial holes [7]. We investigate the
influence of higher-degree basis functions in both the level-set representation
and the approximation of the solution. Two numerical examples demonstrate the
proposed approach, showing that employing higher-degree basis functions for
approximating the solution improves accuracy, while linear basis functions
remain sufficient for the level-set function representation.

### 2. [A modified RIME algorithm with covariance learning and diversity enhancement for numerical optimization](http://arxiv.org/pdf/2509.09529v1)

Authors: Shangqing Shi, Luoxiao Zhang, Yuchen Yin, Xiong Yang, Hoileong Lee

Metaheuristics are widely applied for their ability to provide more efficient
solutions. The RIME algorithm is a recently proposed physical-based
metaheuristic algorithm with certain advantages. However, it suffers from rapid
loss of population diversity during optimization and is prone to fall into
local optima, leading to unbalanced exploitation and exploration. To address
the shortcomings of RIME, this paper proposes a modified RIME with covariance
learning and diversity enhancement (MRIME-CD). The algorithm applies three
strategies to improve the optimization capability. First, a covariance learning
strategy is introduced in the soft-rime search stage to increase the population
diversity and balance the over-exploitation ability of RIME through the
bootstrapping effect of dominant populations. Second, in order to moderate the
tendency of RIME population to approach the optimal individual in the early
search stage, an average bootstrapping strategy is introduced into the
hard-rime puncture mechanism, which guides the population search through the
weighted position of the dominant populations, thus enhancing the global search
ability of RIME in the early stage. Finally, a new stagnation indicator is
proposed, and a stochastic covariance learning strategy is used to update the
stagnant individuals in the population when the algorithm gets stagnant, thus
enhancing the ability to jump out of the local optimal solution. The proposed
MRIME-CD algorithm is subjected to a series of validations on the CEC2017 test
set, the CEC2022 test set, and the experimental results are analyzed using the
Friedman test, the Wilcoxon rank sum test, and the Kruskal Wallis test. The
results show that MRIME-CD can effectively improve the performance of basic
RIME and has obvious superiorities in terms of solution accuracy, convergence
speed and stability.

### 3. [An improved educational competition optimizer with multi-covariance learning operators for global optimization problems](http://arxiv.org/pdf/2509.09552v1)

Authors: Baoqi Zhao, Xiong Yang, Hoileong Lee, Bowen Dong

The educational competition optimizer is a recently introduced metaheuristic
algorithm inspired by human behavior, originating from the dynamics of
educational competition within society. Nonetheless, ECO faces constraints due
to an imbalance between exploitation and exploration, rendering it susceptible
to local optima and demonstrating restricted effectiveness in addressing
complex optimization problems. To address these limitations, this study
presents an enhanced educational competition optimizer (IECO-MCO) utilizing
multi-covariance learning operators. In IECO, three distinct covariance
learning operators are introduced to improve the performance of ECO. Each
operator effectively balances exploitation and exploration while preventing
premature convergence of the population. The effectiveness of IECO is assessed
through benchmark functions derived from the CEC 2017 and CEC 2022 test suites,
and its performance is compared with various basic and improved algorithms
across different categories. The results demonstrate that IECO-MCO surpasses
the basic ECO and other competing algorithms in convergence speed, stability,
and the capability to avoid local optima. Furthermore, statistical analyses,
including the Friedman test, Kruskal-Wallis test, and Wilcoxon rank-sum test,
are conducted to validate the superiority of IECO-MCO over the compared
algorithms. Compared with the basic algorithm (improved algorithm), IECO-MCO
achieved an average ranking of 2.213 (2.488) on the CE2017 and CEC2022 test
suites. Additionally, the practical applicability of the proposed IECO-MCO
algorithm is verified by solving constrained optimization problems. The
experimental outcomes demonstrate the superior performance of IECO-MCO in
tackling intricate optimization problems, underscoring its robustness and
practical effectiveness in real-world scenarios.

### Computational Geometry

### 1. [Swept Volume Computation with Enhanced Geometric Detail Preservation](http://arxiv.org/pdf/2509.09325v1)

Authors: Pengfei Wang, Yuexin Yang, Shuangmin Chen, Shiqing Xin, Changhe Tu, Wenping Wang

Swept volume computation, the determination of regions occupied by moving
objects, is essential in graphics, robotics, and manufacturing. Existing
approaches either explicitly track surfaces, suffering from robustness issues
under complex interactions, or employ implicit representations that trade off
geometric fidelity and face optimization difficulties. We propose a novel
inversion of motion perspective: rather than tracking object motion, we fix the
object and trace spatial points backward in time, reducing complex trajectories
to efficiently linearizable point motions. Based on this, we introduce a multi
field tetrahedral framework that maintains multiple distance fileds per
element, preserving fine geometric details at trajectory intersections where
single field methods fail. Our method robustly computes swept volumes for
diverse motions, including translations and screw motions, and enables
practical applications in path planning and collision detection.

### 2. [Toward Precise Curve Offsetting Constrained to Parametric Surfaces](http://arxiv.org/pdf/2509.09333v1)

Authors: Jin Zhao, Pengfei Wang, Shuangmin Chen, Jiong Guo, Shiqing Xin, Changhe Tu, Wenping Wang

Computing offsets of curves on parametric surfaces is a fundamental yet
challenging operation in computer aided design and manufacturing. Traditional
analytical approaches suffer from time-consuming geodesic distance queries and
complex self intersection handling, while discrete methods often struggle with
precision. In this paper, we propose a totally different algorithm paradigm.
Our key insight is that by representing the source curve as a sequence of line
segment primitives, the Voronoi decomposition constrained to the parametric
surface enables localized offset computation. Specifically, the offsetting
process can be efficiently traced by independently visiting the corresponding
Voronoi cells. To address the challenge of computing the Voronoi decomposition
on parametric surfaces, we introduce two key techniques. First, we employ
intrinsic triangulation in the parameter space to accurately capture geodesic
distances. Second, instead of directly computing the surface-constrained
Voronoi decomposition, we decompose the triangulated parameter plane using a
series of plane cutting operations. Experimental results demonstrate that our
algorithm achieves superior accuracy and runtime performance compared to
existing methods. We also present several practical applications enabled by our
approach.

### 3. [A New Algorithm for Computing Integer Hulls of 2D Polyhedral Sets](http://arxiv.org/pdf/2509.09134v1)

Authors: Chirantan Mukherjee

The $\texttt{IntegerHull}$ function is part of Maple's
$\texttt{PolyhedralSets}$ library, which calculates the integer hull of a given
polyhedral set. This algorithm works by translating the supporting hyperplanes
of the facets of the input polyhedral set inwards till each hyperplane
encounters at least one integer point. The polyhedral set is then divided into
smaller regions and a brute force method is applied to find the remaining
vertices.
  There are certain edge case scenarios where the computational cost of the
existing algorithm can be high, for which we propose a new algorithm. We
translate the supporting hyperplanes of the facets inwards from the (relative)
opposite vertex till the hyperplanes encounter at least one integer point.
Then, we can follow the same procedure as the old algorithm or use a recursive
technique on the smaller regions.
  The edge case scenarios mentioned above occurs when there are integer points
present on the supporting hyperplanes of the facets of the polyhedral set. This
increases the region on which the brute force method is applied to find the
remaining vertices.

### 4. [Characterization of the computed homology and cohomology bases -- technical report](http://arxiv.org/pdf/2509.09350v1)

Authors: Yann-Situ Gazull, Aldo Gonzalez-Lorenzo, Alexandra Bac

Computing homology and cohomology is at the heart of many recent works and a
key issue for topological data analysis. Among homological objects, homology
generators are useful to locate or understand holes (especially for geometric
objects). The present paper provides a characterization of the class of
homology bases that are computed by standard algorithmic methods. The proof of
this characterization relies on the Homological Discrete Vector Field, a
combinatorial structure for computing homology, which encompasses several
standard methods (persistent homology, tri-partitions, Smith Normal Form,
discrete Morse theory). These results refine the combinatorial homology theory
and provide novel ideas to gain more control over the computation of homology
generators.

### Computation and Language

### 1. [MR-UIE: Multi-Perspective Reasoning with Reinforcement Learning for Universal Information Extraction](http://arxiv.org/pdf/2509.09082v1)

Authors: Zhongqiu Li, Shiquan Wang, Ruiyu Fang, Mengjiao Bao, Zhenhe Wu, Shuangyong Song, Yongxiang Li, Zhongjiang He

Large language models (LLMs) demonstrate robust capabilities across diverse
research domains. However, their performance in universal information
extraction (UIE) remains insufficient, especially when tackling structured
output scenarios that involve complex schema descriptions and require
multi-step reasoning. While existing approaches enhance the performance of LLMs
through in-context learning and instruction tuning, significant limitations
nonetheless persist. To enhance the model's generalization ability, we propose
integrating reinforcement learning (RL) with multi-perspective reasoning for
information extraction (IE) tasks. Our work transitions LLMs from passive
extractors to active reasoners, enabling them to understand not only what to
extract but also how to reason. Experiments conducted on multiple IE benchmarks
demonstrate that MR-UIE consistently elevates extraction accuracy across
domains and surpasses state-of-the-art methods on several datasets.
Furthermore, incorporating multi-perspective reasoning into RL notably enhances
generalization in complex IE tasks, underscoring the critical role of reasoning
in challenging scenarios.

### 2. [TigerCoder: A Novel Suite of LLMs for Code Generation in Bangla](http://arxiv.org/pdf/2509.09101v1)

Authors: Nishat Raihan, Antonios Anastasopoulos, Marcos Zampieri

Despite being the 5th most spoken language, Bangla remains underrepresented
in Large Language Models (LLMs), particularly for code generation. This
primarily stems from the scarcity of high-quality data to pre-train and/or
finetune such models. Hence, we introduce the first dedicated family of Code
LLMs for Bangla (1B & 9B). We offer three major contributions: (1) a
comprehensive Bangla code instruction datasets for programming domain
adaptation; (2) MBPP-Bangla, an evaluation benchmark for Bangla code
generation; and (3) the TigerCoder-family of Code LLMs, achieving significant
~11-18% performance gains at Pass@1 over existing multilingual and
general-purpose Bangla LLMs. Our findings show that curated, high-quality
datasets can overcome limitations of smaller models for low-resource languages.
We open-source all resources to advance further Bangla LLM research.

### 3. [Compass-v3: Scaling Domain-Specific LLMs for Multilingual E-Commerce in Southeast Asia](http://arxiv.org/pdf/2509.09121v1)

Authors: Sophia Maria

Large language models (LLMs) excel in general-domain applications, yet their
performance often degrades in specialized tasks requiring domain-specific
knowledge. E-commerce is particularly challenging, as its data are noisy,
heterogeneous, multilingual, and highly dynamic. We present Compass-v3, a
vertical-domain Mixture-of-Experts (MoE) model with 245B total parameters and
71B active per token, designed for Southeast Asian e-commerce. Compass-v3
adopts fewer but larger experts, combined with hardware-efficient
optimizations-such as intra-node expert parallelism and a customized memcpy
operator-to maximize GPU utilization. The model is trained on 12T tokens of
curated multilingual corpora and large-scale synthetic e-commerce instructions
using a mixed-training strategy. To enhance alignment, we propose
Optimal-Transport Direct Preference Optimization (OTPO), which captures
token-level distinctions and improves instruction adherence in
commerce-specific scenarios. Extensive evaluations demonstrate that Compass-v3
delivers state-of-the-art e-commerce performance, surpassing DeepSeek-V3.1,
GPT-4 series, and Qwen3-235B. Moreover, Compass-v3 demonstrates strong
multilingual capability across low-resource Southeast Asian languages
(Indonesian, Thai, Filipino, Vietnamese, Malay, Taglog) and Portuguese while
sustaining competitive performance on general benchmarks. It has already been
widely applied in Shopee's industrial-scale e-commerce platform and is
gradually replacing OpenAI's traffic, now accounting for over 70\% of total LLM
usage, highlighting its dual strengths in specialized commerce expertise and
broad linguistic competence.

### 4. [GmSLM : Generative Marmoset Spoken Language Modeling](http://arxiv.org/pdf/2509.09198v1)

Authors: Talia Sternberg, Michael London, David Omer, Yossi Adi

Marmoset monkeys exhibit complex vocal communication, challenging the view
that nonhuman primates vocal communication is entirely innate, and show similar
features of human speech, such as vocal labeling of others and turn-taking.
Studying their vocal communication offers a unique opportunity to link it with
brain activity-especially given the difficulty of accessing the human brain in
speech and language research. Since Marmosets communicate primarily through
vocalizations, applying standard LLM approaches is not straightforward. We
introduce Generative Marmoset Spoken Language Modeling (GmSLM), an optimized
spoken language model pipeline for Marmoset vocal communication. We designed a
novel zero-shot evaluation metrics using unsupervised in-the-wild data,
alongside weakly labeled conversational data, to assess GmSLM and demonstrate
its advantage over a basic human-speech-based baseline. GmSLM generated
vocalizations closely matched real resynthesized samples acoustically and
performed well on downstream tasks. Despite being fully unsupervised, GmSLM
effectively distinguish real from artificial conversations and may support
further investigations of the neural basis of vocal communication and provides
a practical framework linking vocalization and brain activity. We believe GmSLM
stands to benefit future work in neuroscience, bioacoustics, and evolutionary
biology. Samples are provided under: pages.cs.huji.ac.il/adiyoss-lab/GmSLM.

### 5. [CCF: A Context Compression Framework for Efficient Long-Sequence Language Modeling](http://arxiv.org/pdf/2509.09199v1)

Authors: Wenhao Li, Bangcheng Sun, Weihao Ye, Tianyi Zhang, Daohai Yu, Fei Chao, Rongrong Ji

Scaling language models to longer contexts is essential for capturing rich
dependencies across extended discourse. However, na\"ive context extension
imposes significant computational and memory burdens, often resulting in
inefficiencies during both training and inference. In this work, we propose
CCF, a novel context compression framework designed to enable efficient
long-context modeling by learning hierarchical latent representations that
preserve global semantics while aggressively reducing input redundancy. CCF
integrates segment-wise semantic aggregation with key-value memory encoding,
forming compact representations that support accurate reconstruction and
long-range understanding. To further enhance scalability, we introduce a
training-efficient optimization strategy that couples incremental segment
decoding with sparse reservoir sampling, substantially reducing memory overhead
without degrading performance. Empirical results on multiple long-context
language modeling benchmarks demonstrate that CCF achieves competitive
perplexity under high compression ratios, and significantly improves throughput
and memory efficiency compared to existing approaches. These findings highlight
the potential of structured compression for scalable and effective long-context
language modeling.

### 6. [Reading Between the Lines: Classifying Resume Seniority with Large Language Models](http://arxiv.org/pdf/2509.09229v1)

Authors: Matan Cohen, Shira Shani, Eden Menahem, Yehudit Aperstein, Alexander Apartsin

Accurately assessing candidate seniority from resumes is a critical yet
challenging task, complicated by the prevalence of overstated experience and
ambiguous self-presentation. In this study, we investigate the effectiveness of
large language models (LLMs), including fine-tuned BERT architectures, for
automating seniority classification in resumes. To rigorously evaluate model
performance, we introduce a hybrid dataset comprising both real-world resumes
and synthetically generated hard examples designed to simulate exaggerated
qualifications and understated seniority. Using the dataset, we evaluate the
performance of Large Language Models in detecting subtle linguistic cues
associated with seniority inflation and implicit expertise. Our findings
highlight promising directions for enhancing AI-driven candidate evaluation
systems and mitigating bias introduced by self-promotional language. The
dataset is available for the research community at https://bit.ly/4mcTovt

### 7. [Agentic LLMs for Question Answering over Tabular Data](http://arxiv.org/pdf/2509.09234v1)

Authors: Rishit Tyagi, Mohit Gupta, Rahul Bouri

Question Answering over Tabular Data (Table QA) presents unique challenges
due to the diverse structure, size, and data types of real-world tables. The
SemEval 2025 Task 8 (DataBench) introduced a benchmark composed of large-scale,
domain-diverse datasets to evaluate the ability of models to accurately answer
structured queries. We propose a Natural Language to SQL (NL-to-SQL) approach
leveraging large language models (LLMs) such as GPT-4o, GPT-4o-mini, and
DeepSeek v2:16b to generate SQL queries dynamically. Our system follows a
multi-stage pipeline involving example selection, SQL query generation, answer
extraction, verification, and iterative refinement. Experiments demonstrate the
effectiveness of our approach, achieving 70.5\% accuracy on DataBench QA and
71.6\% on DataBench Lite QA, significantly surpassing baseline scores of 26\%
and 27\% respectively. This paper details our methodology, experimental
results, and alternative approaches, providing insights into the strengths and
limitations of LLM-driven Table QA.

### 8. [From scratch to silver: Creating trustworthy training data for patent-SDG classification using Large Language Models](http://arxiv.org/pdf/2509.09303v1)

Authors: Grazia Sveva Ascione, Nicolò Tamagnone

Classifying patents by their relevance to the UN Sustainable Development
Goals (SDGs) is crucial for tracking how innovation addresses global
challenges. However, the absence of a large, labeled dataset limits the use of
supervised learning. Existing methods, such as keyword searches, transfer
learning, and citation-based heuristics, lack scalability and generalizability.
This paper frames patent-to-SDG classification as a weak supervision problem,
using citations from patents to SDG-tagged scientific publications (NPL
citations) as a noisy initial signal. To address its sparsity and noise, we
develop a composite labeling function (LF) that uses large language models
(LLMs) to extract structured concepts, namely functions, solutions, and
applications, from patents and SDG papers based on a patent ontology.
Cross-domain similarity scores are computed and combined using a rank-based
retrieval approach. The LF is calibrated via a custom positive-only loss that
aligns with known NPL-SDG links without penalizing discovery of new SDG
associations. The result is a silver-standard, soft multi-label dataset mapping
patents to SDGs, enabling the training of effective multi-label regression
models. We validate our approach through two complementary strategies: (1)
internal validation against held-out NPL-based labels, where our method
outperforms several baselines including transformer-based models, and zero-shot
LLM; and (2) external validation using network modularity in patent citation,
co-inventor, and co-applicant graphs, where our labels reveal greater thematic,
cognitive, and organizational coherence than traditional technological
classifications. These results show that weak supervision and semantic
alignment can enhance SDG classification at scale.

### 9. [MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems](http://arxiv.org/pdf/2509.09360v1)

Authors: Channdeth Sok, David Luz, Yacine Haddam

Large Language Models (LLMs) are increasingly deployed in enterprise
applications, yet their reliability remains limited by hallucinations, i.e.,
confident but factually incorrect information. Existing detection approaches,
such as SelfCheckGPT and MetaQA, primarily target standalone LLMs and do not
address the unique challenges of Retrieval-Augmented Generation (RAG) systems,
where responses must be consistent with retrieved evidence. We therefore
present MetaRAG, a metamorphic testing framework for hallucination detection in
Retrieval-Augmented Generation (RAG) systems. MetaRAG operates in a real-time,
unsupervised, black-box setting, requiring neither ground-truth references nor
access to model internals, making it suitable for proprietary and high-stakes
domains. The framework proceeds in four stages: (1) decompose answers into
atomic factoids, (2) generate controlled mutations of each factoid using
synonym and antonym substitutions, (3) verify each variant against the
retrieved context (synonyms are expected to be entailed and antonyms
contradicted), and (4) aggregate penalties for inconsistencies into a
response-level hallucination score. Crucially for identity-aware AI, MetaRAG
localizes unsupported claims at the factoid span where they occur (e.g.,
pregnancy-specific precautions, LGBTQ+ refugee rights, or labor eligibility),
allowing users to see flagged spans and enabling system designers to configure
thresholds and guardrails for identity-sensitive queries. Experiments on a
proprietary enterprise dataset illustrate the effectiveness of MetaRAG for
detecting hallucinations and enabling trustworthy deployment of RAG-based
conversational agents. We also outline a topic-based deployment design that
translates MetaRAG's span-level scores into identity-aware safeguards; this
design is discussed but not evaluated in our experiments.

### 10. [Modelling Analogies and Analogical Reasoning: Connecting Cognitive Science Theory and NLP Research](http://arxiv.org/pdf/2509.09381v1)

Authors: Molly R Petersen, Claire E Stevenson, Lonneke van der Plas

Analogical reasoning is an essential aspect of human cognition. In this
paper, we summarize key theory about the processes underlying analogical
reasoning from the cognitive science literature and relate it to current
research in natural language processing. While these processes can be easily
linked to concepts in NLP, they are generally not viewed through a cognitive
lens. Furthermore, we show how these notions are relevant for several major
challenges in NLP research, not directly related to analogy solving. This may
guide researchers to better optimize relational understanding in text, as
opposed to relying heavily on entity-level similarity.

### Cryptography and Security

### 1. [Beyond Tag Collision: Cluster-based Memory Management for Tag-based Sanitizers](http://arxiv.org/pdf/2509.09089v1)

Authors: Mengfei Xie, Yan Lin, Hongtao Wu, Jianming Fu, Chenke Luo, Guojun Peng

Tag-based sanitizers attach a small "key" to each pointer and a matching
"lock" tag to its target memory object, enabling runtime verification of
pointer-object consistency and helping developers to detect potential memory
violations. However, the limited tag encoding space challenges existing studies
in assigning distinct tags to memory objects across temporal and spatial
dimensions, leading to potential tag collisions. In this paper, we present
ClusterTag, a novel cluster-based memory allocator aimed at simultaneously
mitigating tag collisions in both temporal and spatial dimensions. The core
design of ClusterTag effectively balances the significant mismatch between tag
encoding space and memory objects: it divides memory objects into multiple
independent clusters, thereby limiting tag collisions to finite chunks within
each cluster. To mitigate tag collisions across clusters, we design a
cluster-grained heap randomization scheme. This approach introduces random
address intervals between clusters and further breaks the entropy limitation of
the tag space. ClusterTag has been implemented as an independent memory
allocator that seamlessly integrates with tag-based sanitizers such as HWASan,
and maintains comparable performance overhead (within 1%) at various
randomization densities. Security evaluations on the Juliet dataset indicate
that ClusterTag exhibits deterministic results across 500 repeated tests (5,652
reported and 1,530 missed), while the existing three types of tag assignment
strategies all exhibit probabilistic false negatives due to tag collisions.
Quantitative analysis across three tag collision distance metrics-minimum,
average, and unpredictability-demonstrates that ClusterTag achieves balanced
improvements across all three, whereas prior tag assignment schemes (random,
staggered, fixed) show significant trade-offs in at least one metric.

### 2. [AgriSentinel: Privacy-Enhanced Embedded-LLM Crop Disease Alerting System](http://arxiv.org/pdf/2509.09103v1)

Authors: Chanti Raju Mylay, Bobin Deng, Zhipeng Cai, Honghui Xu

Crop diseases pose significant threats to global food security, agricultural
productivity, and sustainable farming practices, directly affecting farmers'
livelihoods and economic stability. To address the growing need for effective
crop disease management, AI-based disease alerting systems have emerged as
promising tools by providing early detection and actionable insights for timely
intervention. However, existing systems often overlook critical aspects such as
data privacy, market pricing power, and farmer-friendly usability, leaving
farmers vulnerable to privacy breaches and economic exploitation. To bridge
these gaps, we propose AgriSentinel, the first Privacy-Enhanced Embedded-LLM
Crop Disease Alerting System. AgriSentinel incorporates a differential privacy
mechanism to protect sensitive crop image data while maintaining classification
accuracy. Its lightweight deep learning-based crop disease classification model
is optimized for mobile devices, ensuring accessibility and usability for
farmers. Additionally, the system includes a fine-tuned, on-device large
language model (LLM) that leverages a curated knowledge pool to provide farmers
with specific, actionable suggestions for managing crop diseases, going beyond
simple alerting. Comprehensive experiments validate the effectiveness of
AgriSentinel, demonstrating its ability to safeguard data privacy, maintain
high classification performance, and deliver practical, actionable disease
management strategies. AgriSentinel offers a robust, farmer-friendly solution
for automating crop disease alerting and management, ultimately contributing to
improved agricultural decision-making and enhanced crop productivity.

### 3. [IoTFuzzSentry: A Protocol Guided Mutation Based Fuzzer for Automatic Vulnerability Testing in Commercial IoT Devices](http://arxiv.org/pdf/2509.09158v1)

Authors: Priyanka Rushikesh Chaudhary, Rajib Ranjan Maiti

Protocol fuzzing is a scalable and cost-effective technique for identifying
security vulnerabilities in deployed Internet of Things devices. During their
operational phase, IoT devices often run lightweight servers to handle user
interactions, such as video streaming or image capture in smart cameras.
Implementation flaws in transport or application-layer security mechanisms can
expose IoT devices to a range of threats, including unauthorized access and
data leakage. This paper addresses the challenge of uncovering such
vulnerabilities by leveraging protocol fuzzing techniques that inject crafted
transport and application-layer packets into IoT communications. We present a
mutation-based fuzzing tool, named IoTFuzzSentry, to identify specific
non-trivial vulnerabilities in commercial IoT devices. We further demonstrate
how these vulnerabilities can be exploited in real-world scenarios. We
integrated our fuzzing tool into a well-known testing tool Cotopaxi and
evaluated it with commercial-off-the-shelf IoT devices such as IP cameras and
Smart Plug. Our evaluation revealed vulnerabilities categorized into 4 types
(IoT Access Credential Leakage, Sneak IoT Live Video Stream, Creep IoT Live
Image, IoT Command Injection) and we show their exploits using three IoT
devices. We have responsibly disclosed all these vulnerabilities to the
respective vendors. So far, we have published two CVEs, CVE-2024-41623 and
CVE-2024-42531, and one is awaiting. To extend the applicability, we have
investigated the traffic of six additional IoT devices and our analysis shows
that these devices can have similar vulnerabilities, due to the presence of a
similar set of application protocols. We believe that IoTFuzzSentry has the
potential to discover unconventional security threats and allow IoT vendors to
strengthen the security of their commercialized IoT devices automatically with
negligible overhead.

### 4. [Enhancing Cyber Threat Hunting -- A Visual Approach with the Forensic Visualization Toolkit](http://arxiv.org/pdf/2509.09185v1)

Authors: Jihane Najar, Marinos Tsantekidis, Aris Sotiropoulos, Vassilis Prevelakis

In today's dynamic cyber threat landscape, organizations must take proactive
steps to bolster their cybersecurity defenses. Cyber threat hunting is a
proactive and iterative process aimed at identifying and mitigating advanced
threats that may go undetected by traditional security measures. Rather than
waiting for automated security systems to flag potential threats, threat
hunting involves actively searching for signs of malicious activity within an
organization's network. In this paper, we present the Forensic Visualization
Toolkit, a powerful tool designed for digital forensics investigations,
analysis of digital evidence, and advanced visualizations to enhance
cybersecurity situational awareness and risk management and empower security
analysts with an intuitive and interactive tool. Through practical, real-world
scenarios, we demonstrate how FVT significantly amplifies the capabilities of
cybersecurity professionals, enabling them to effectively identify, analyze,
and respond to threats. Furthermore, it is important to highlight that FVT has
been integrated into, utilized, and continually enhanced within various
EU-funded research projects over recent years.

### 5. [Shell or Nothing: Real-World Benchmarks and Memory-Activated Agents for Automated Penetration Testing](http://arxiv.org/pdf/2509.09207v1)

Authors: Wuyuao Mai, Geng Hong, Qi Liu, Jinsong Chen, Jiarun Dai, Xudong Pan, Yuan Zhang, Min Yang

Penetration testing is critical for identifying and mitigating security
vulnerabilities, yet traditional approaches remain expensive, time-consuming,
and dependent on expert human labor. Recent work has explored AI-driven
pentesting agents, but their evaluation relies on oversimplified
capture-the-flag (CTF) settings that embed prior knowledge and reduce
complexity, leading to performance estimates far from real-world practice. We
close this gap by introducing the first real-world, agent-oriented pentesting
benchmark, TermiBench, which shifts the goal from 'flag finding' to achieving
full system control. The benchmark spans 510 hosts across 25 services and 30
CVEs, with realistic environments that require autonomous reconnaissance,
discrimination between benign and exploitable services, and robust exploit
execution. Using this benchmark, we find that existing systems can hardly
obtain system shells under realistic conditions.
  To address these challenges, we propose TermiAgent, a multi-agent penetration
testing framework. TermiAgent mitigates long-context forgetting with a Located
Memory Activation mechanism and builds a reliable exploit arsenal via
structured code understanding rather than naive retrieval. In evaluations, our
work outperforms state-of-the-art agents, exhibiting stronger penetration
testing capability, reducing execution time and financial cost, and
demonstrating practicality even on laptop-scale deployments. Our work delivers
both the first open-source benchmark for real-world autonomous pentesting and a
novel agent framework that establishes a milestone for AI-driven penetration
testing.

### 6. [On the Security of SSH Client Signatures](http://arxiv.org/pdf/2509.09331v1)

Authors: Fabian Bäumer, Marcus Brinkmann, Maximilian Radoy, Jörg Schwenk, Juraj Somorovsky

Administrators and developers use SSH client keys and signatures for
authentication, for example, to access internet backbone servers or to commit
new code on platforms like GitHub. However, unlike servers, SSH clients cannot
be measured through internet scans. We close this gap in two steps. First, we
collect SSH client public keys. Such keys are regularly published by their
owners on open development platforms like GitHub and GitLab. We systematize
previous non-academic work by subjecting these keys to various security tests
in a longitudinal study. Second, in a series of black-box lab experiments, we
analyze the implementations of algorithms for SSH client signatures in 24
popular SSH clients for Linux, Windows, and macOS.
  We extracted 31,622,338 keys from three public sources in two scans. Compared
to previous work, we see a clear tendency to abandon RSA signatures in favor of
EdDSA signatures. Still, in January 2025, we found 98 broken short keys, 139
keys generated from weak randomness, and 149 keys with common or small
factors-the large majority of the retrieved keys exposed no weakness.
  Weak randomness can not only compromise a secret key through its public key,
but also through signatures. It is well-known that a bias in random nonces in
ECDSA can reveal the secret key through public signatures. For the first time,
we show that the use of deterministic nonces in ECDSA can also be dangerous:
The private signing key of a PuTTY client can be recovered from just 58 valid
signatures if ECDSA with NIST curve P-521 is used. PuTTY acknowledged our
finding in CVE-2024-31497, and they subsequently replaced the nonce generation
algorithm.

### 7. [[Extended] Ethics in Computer Security Research: A Data-Driven Assessment of the Past, the Present, and the Possible Future](http://arxiv.org/pdf/2509.09351v1)

Authors: Harshini Sri Ramulu, Helen Schmitt, Bogdan Rerich, Rachel Gonzalez Rodriguez, Tadayoshi Kohno, Yasemin Acar

Ethical questions are discussed regularly in computer security. Still,
researchers in computer security lack clear guidance on how to make, document,
and assess ethical decisions in research when what is morally right or
acceptable is not clear-cut. In this work, we give an overview of the
discussion of ethical implications in current published work in computer
security by reviewing all 1154 top-tier security papers published in 2024,
finding inconsistent levels of ethics reporting with a strong focus of
reporting institutional or ethics board approval, human subjects protection,
and responsible disclosure, and a lack of discussion of balancing harms and
benefits. We further report on the results of a semi-structured interview study
with 24 computer security and privacy researchers (among whom were also:
reviewers, ethics committee members, and/or program chairs) and their ethical
decision-making both as authors and during peer review, finding a strong desire
for ethical research, but a lack of consistency in considered values, ethical
frameworks (if articulated), decision-making, and outcomes. We present an
overview of the current state of the discussion of ethics and current de-facto
standards in computer security research, and contribute suggestions to improve
the state of ethics in computer security research.

### 8. [Bridging the Gap in Phishing Detection: A Comprehensive Phishing Dataset Collector](http://arxiv.org/pdf/2509.09592v1)

Authors: Aditya Kulkarni, Shahil Manishbhai Patel, Shivam Pradip Tirmare, Vivek Balachandran, Tamal Das

To combat phishing attacks -- aimed at luring web users to divulge their
sensitive information -- various phishing detection approaches have been
proposed. As attackers focus on devising new tactics to bypass existing
detection solutions, researchers have adapted by integrating machine learning
and deep learning into phishing detection. Phishing dataset collection is vital
to developing effective phishing detection approaches, which highly depend on
the diversity of the gathered datasets. The lack of diversity in the dataset
results in a biased model. Since phishing websites are often short-lived,
collecting them is also a challenge. Consequently, very few phishing webpage
dataset repositories exist to date. No single repository comprehensively
consolidates all phishing elements corresponding to a phishing webpage, namely,
URL, webpage source code, screenshot, and related webpage resources. This paper
introduces a resource collection tool designed to gather various resources
associated with a URL, such as CSS, Javascript, favicons, webpage images, and
screenshots. Our tool leverages PhishTank as the primary source for obtaining
active phishing URLs. Our tool fetches several additional webpage resources
compared to PyWebCopy Python library, which provides webpage content for a
given URL. Additionally, we share a sample dataset generated using our tool
comprising 4,056 legitimate and 5,666 phishing URLs along with their associated
resources. We also remark on the top correlated phishing features with their
associated class label found in our dataset. Our tool offers a comprehensive
resource set that can aid researchers in developing effective phishing
detection approaches.

### 9. [Fingerprinting Deep Packet Inspection Devices by Their Ambiguities](http://arxiv.org/pdf/2509.09081v1)

Authors: Diwen Xue, Armin Huremagic, Wayne Wang, Ram Sundara Raman, Roya Ensafi

Users around the world face escalating network interference such as
censorship, throttling, and interception, largely driven by the commoditization
and growing availability of Deep Packet Inspection (DPI) devices. Once reserved
for a few well-resourced nation-state actors, the ability to interfere with
traffic at scale is now within reach of nearly any network operator. Despite
this proliferation, our understanding of DPIs and their deployments on the
Internet remains limited -- being network intermediary leaves DPI unresponsive
to conventional host-based scanning tools, and DPI vendors actively obscuring
their products further complicates measurement efforts.
  In this work, we present a remote measurement framework, dMAP (DPI Mapper),
that derives behavioral fingerprints for DPIs to differentiate and cluster
these otherwise indistinguishable middleboxes at scale, as a first step toward
active reconnaissance of DPIs on the Internet. Our key insight is that parsing
and interpreting traffic as network intermediaries inherently involves
ambiguities -- from under-specified protocol behaviors to differing RFC
interpretations -- forcing DPI vendors into independent implementation choices
that create measurable variance among DPIs. Based on differential fuzzing, dMAP
systematically discovers, selects, and deploys specialized probes that
translate DPI internal parsing behaviors into externally observable
fingerprints. Applying dMAP to DPI deployments globally, we demonstrate its
practical feasibility, showing that even a modest set of 20-40 discriminative
probes reliably differentiates a wide range of DPI implementations, including
major nation-state censorship infrastructures and commercial DPI products. We
discuss how our fingerprinting methodology generalizes beyond censorship to
other forms of targeted interference.

### 10. [Towards Confidential and Efficient LLM Inference with Dual Privacy Protection](http://arxiv.org/pdf/2509.09091v1)

Authors: Honglan Yu, Yibin Wang, Feifei Dai, Dong Liu, Haihui Fan, Xiaoyan Gu

CPU-based trusted execution environments (TEEs) and differential privacy (DP)
have gained wide applications for private inference. Due to high inference
latency in TEEs, researchers use partition-based approaches that offload linear
model components to GPUs. However, dense nonlinear layers of large language
models (LLMs) result in significant communication overhead between TEEs and
GPUs. DP-based approaches apply random noise to protect data privacy, but this
compromises LLM performance and semantic understanding. To overcome the above
drawbacks, this paper proposes CMIF, a Confidential and efficient Model
Inference Framework. CMIF confidentially deploys the embedding layer in the
client-side TEE and subsequent layers on GPU servers. Meanwhile, it optimizes
the Report-Noisy-Max mechanism to protect sensitive inputs with a slight
decrease in model performance. Extensive experiments on Llama-series models
demonstrate that CMIF reduces additional inference overhead in TEEs while
preserving user data privacy.

### Computer Vision and Pattern Recognition

### 1. [Enhancing 3D Medical Image Understanding with Pretraining Aided by 2D Multimodal Large Language Models](http://arxiv.org/pdf/2509.09064v1)

Authors: Qiuhui Chen, Xuancheng Yao, Huping Ye, Yi Hong

Understanding 3D medical image volumes is critical in the medical field, yet
existing 3D medical convolution and transformer-based self-supervised learning
(SSL) methods often lack deep semantic comprehension. Recent advancements in
multimodal large language models (MLLMs) provide a promising approach to
enhance image understanding through text descriptions. To leverage these 2D
MLLMs for improved 3D medical image understanding, we propose Med3DInsight, a
novel pretraining framework that integrates 3D image encoders with 2D MLLMs via
a specially designed plane-slice-aware transformer module. Additionally, our
model employs a partial optimal transport based alignment, demonstrating
greater tolerance to noise introduced by potential noises in LLM-generated
content. Med3DInsight introduces a new paradigm for scalable multimodal 3D
medical representation learning without requiring human annotations. Extensive
experiments demonstrate our state-of-the-art performance on two downstream
tasks, i.e., segmentation and classification, across various public datasets
with CT and MRI modalities, outperforming current SSL methods. Med3DInsight can
be seamlessly integrated into existing 3D medical image understanding networks,
potentially enhancing their performance. Our source code, generated datasets,
and pre-trained models will be available at
https://github.com/Qybc/Med3DInsight.

### 2. [Improvement of Human-Object Interaction Action Recognition Using Scene Information and Multi-Task Learning Approach](http://arxiv.org/pdf/2509.09067v1)

Authors: Hesham M. Shehata, Mohammad Abdolrahmani

Recent graph convolutional neural networks (GCNs) have shown high performance
in the field of human action recognition by using human skeleton poses.
However, it fails to detect human-object interaction cases successfully due to
the lack of effective representation of the scene information and appropriate
learning architectures. In this context, we propose a methodology to utilize
human action recognition performance by considering fixed object information in
the environment and following a multi-task learning approach. In order to
evaluate the proposed method, we collected real data from public environments
and prepared our data set, which includes interaction classes of hands-on fixed
objects (e.g., ATM ticketing machines, check-in/out machines, etc.) and
non-interaction classes of walking and standing. The multi-task learning
approach, along with interaction area information, succeeds in recognizing the
studied interaction and non-interaction actions with an accuracy of 99.25%,
outperforming the accuracy of the base model using only human skeleton poses by
2.75%.

### 3. [IRDFusion: Iterative Relation-Map Difference guided Feature Fusion for Multispectral Object Detection](http://arxiv.org/pdf/2509.09085v1)

Authors: Jifeng Shen, Haibo Zhan, Xin Zuo, Heng Fan, Xiaohui Yuan, Jun Li, Wankou Yang

Current multispectral object detection methods often retain extraneous
background or noise during feature fusion, limiting perceptual performance.To
address this, we propose an innovative feature fusion framework based on
cross-modal feature contrastive and screening strategy, diverging from
conventional approaches. The proposed method adaptively enhances salient
structures by fusing object-aware complementary cross-modal features while
suppressing shared background interference.Our solution centers on two novel,
specially designed modules: the Mutual Feature Refinement Module (MFRM) and the
Differential Feature Feedback Module (DFFM). The MFRM enhances intra- and
inter-modal feature representations by modeling their relationships, thereby
improving cross-modal alignment and discriminative power.Inspired by feedback
differential amplifiers, the DFFM dynamically computes inter-modal differential
features as guidance signals and feeds them back to the MFRM, enabling adaptive
fusion of complementary information while suppressing common-mode noise across
modalities. To enable robust feature learning, the MFRM and DFFM are integrated
into a unified framework, which is formally formulated as an Iterative
Relation-Map Differential Guided Feature Fusion mechanism, termed IRDFusion.
IRDFusion enables high-quality cross-modal fusion by progressively amplifying
salient relational signals through iterative feedback, while suppressing
feature noise, leading to significant performance gains.In extensive
experiments on FLIR, LLVIP and M$^3$FD datasets, IRDFusion achieves
state-of-the-art performance and consistently outperforms existing methods
across diverse challenging scenarios, demonstrating its robustness and
effectiveness. Code will be available at
https://github.com/61s61min/IRDFusion.git.

### 4. [S-BEVLoc: BEV-based Self-supervised Framework for Large-scale LiDAR Global Localization](http://arxiv.org/pdf/2509.09110v1)

Authors: Chenghao Zhang, Lun Luo, Si-Yuan Cao, Xiaokai Bai, Yuncheng Jin, Zhu Yu, Beinan Yu, Yisen Wang, Hui-Liang Shen

LiDAR-based global localization is an essential component of simultaneous
localization and mapping (SLAM), which helps loop closure and re-localization.
Current approaches rely on ground-truth poses obtained from GPS or SLAM
odometry to supervise network training. Despite the great success of these
supervised approaches, substantial cost and effort are required for
high-precision ground-truth pose acquisition. In this work, we propose
S-BEVLoc, a novel self-supervised framework based on bird's-eye view (BEV) for
LiDAR global localization, which eliminates the need for ground-truth poses and
is highly scalable. We construct training triplets from single BEV images by
leveraging the known geographic distances between keypoint-centered BEV
patches. Convolutional neural network (CNN) is used to extract local features,
and NetVLAD is employed to aggregate global descriptors. Moreover, we introduce
SoftCos loss to enhance learning from the generated triplets. Experimental
results on the large-scale KITTI and NCLT datasets show that S-BEVLoc achieves
state-of-the-art performance in place recognition, loop closure, and global
localization tasks, while offering scalability that would require extra effort
for supervised approaches.

### 5. [FPI-Det: a face--phone Interaction Dataset for phone-use detection and understanding](http://arxiv.org/pdf/2509.09111v1)

Authors: Jianqin Gao, Tianqi Wang, Yu Zhang, Yishu Zhang, Chenyuan Wang, Allan Dong, Zihao Wang

The widespread use of mobile devices has created new challenges for vision
systems in safety monitoring, workplace productivity assessment, and attention
management. Detecting whether a person is using a phone requires not only
object recognition but also an understanding of behavioral context, which
involves reasoning about the relationship between faces, hands, and devices
under diverse conditions. Existing generic benchmarks do not fully capture such
fine-grained human--device interactions. To address this gap, we introduce the
FPI-Det, containing 22{,}879 images with synchronized annotations for faces and
phones across workplace, education, transportation, and public scenarios. The
dataset features extreme scale variation, frequent occlusions, and varied
capture conditions. We evaluate representative YOLO and DETR detectors,
providing baseline results and an analysis of performance across object sizes,
occlusion levels, and environments. Source code and dataset is available at
https://github.com/KvCgRv/FPI-Det.

### 6. [Zero-shot Hierarchical Plant Segmentation via Foundation Segmentation Models and Text-to-image Attention](http://arxiv.org/pdf/2509.09116v1)

Authors: Junhao Xing, Ryohei Miyakawa, Yang Yang, Xinpeng Liu, Risa Shinoda, Hiroaki Santo, Yosuke Toda, Fumio Okura

Foundation segmentation models achieve reasonable leaf instance extraction
from top-view crop images without training (i.e., zero-shot). However,
segmenting entire plant individuals with each consisting of multiple
overlapping leaves remains challenging. This problem is referred to as a
hierarchical segmentation task, typically requiring annotated training
datasets, which are often species-specific and require notable human labor. To
address this, we introduce ZeroPlantSeg, a zero-shot segmentation for
rosette-shaped plant individuals from top-view images. We integrate a
foundation segmentation model, extracting leaf instances, and a vision-language
model, reasoning about plants' structures to extract plant individuals without
additional training. Evaluations on datasets with multiple plant species,
growth stages, and shooting environments demonstrate that our method surpasses
existing zero-shot methods and achieves better cross-domain performance than
supervised methods. Implementations are available at
https://github.com/JunhaoXing/ZeroPlantSeg.

### 7. [Gradient-Attention Guided Dual-Masking Synergetic Framework for Robust Text-based Person Retrieval](http://arxiv.org/pdf/2509.09118v1)

Authors: Tianlu Zheng, Yifan Zhang, Xiang An, Ziyong Feng, Kaicheng Yang, Qichuan Ding

Although Contrastive Language-Image Pre-training (CLIP) exhibits strong
performance across diverse vision tasks, its application to person
representation learning faces two critical challenges: (i) the scarcity of
large-scale annotated vision-language data focused on person-centric images,
and (ii) the inherent limitations of global contrastive learning, which
struggles to maintain discriminative local features crucial for fine-grained
matching while remaining vulnerable to noisy text tokens. This work advances
CLIP for person representation learning through synergistic improvements in
data curation and model architecture. First, we develop a noise-resistant data
construction pipeline that leverages the in-context learning capabilities of
MLLMs to automatically filter and caption web-sourced images. This yields
WebPerson, a large-scale dataset of 5M high-quality person-centric image-text
pairs. Second, we introduce the GA-DMS (Gradient-Attention Guided Dual-Masking
Synergetic) framework, which improves cross-modal alignment by adaptively
masking noisy textual tokens based on the gradient-attention similarity score.
Additionally, we incorporate masked token prediction objectives that compel the
model to predict informative text tokens, enhancing fine-grained semantic
representation learning. Extensive experiments show that GA-DMS achieves
state-of-the-art performance across multiple benchmarks.

### 8. [ALL-PET: A Low-resource and Low-shot PET Foundation Model in the Projection Domain](http://arxiv.org/pdf/2509.09130v1)

Authors: Bin Huang, Kang Chen, Bingxuan Li, Huafeng Liu, Qiegen Liu

Building large-scale foundation model for PET imaging is hindered by limited
access to labeled data and insufficient computational resources. To overcome
data scarcity and efficiency limitations, we propose ALL-PET, a low-resource,
low-shot PET foundation model operating directly in the projection domain.
ALL-PET leverages a latent diffusion model (LDM) with three key innovations.
First, we design a Radon mask augmentation strategy (RMAS) that generates over
200,000 structurally diverse training samples by projecting randomized
image-domain masks into sinogram space, significantly improving generalization
with minimal data. This is extended by a dynamic multi-mask (DMM) mechanism
that varies mask quantity and distribution, enhancing data diversity without
added model complexity. Second, we implement positive/negative mask constraints
to embed strict geometric consistency, reducing parameter burden while
preserving generation quality. Third, we introduce transparent medical
attention (TMA), a parameter-free, geometry-driven mechanism that enhances
lesion-related regions in raw projection data. Lesion-focused attention maps
are derived from coarse segmentation, covering both hypermetabolic and
hypometabolic areas, and projected into sinogram space for physically
consistent guidance. The system supports clinician-defined ROI adjustments,
ensuring flexible, interpretable, and task-adaptive emphasis aligned with PET
acquisition physics. Experimental results show ALL-PET achieves high-quality
sinogram generation using only 500 samples, with performance comparable to
models trained on larger datasets. ALL-PET generalizes across tasks including
low-dose reconstruction, attenuation correction, delayed-frame prediction, and
tracer separation, operating efficiently with memory use under 24GB.

### 9. [Noise-Robust Topology Estimation of 2D Image Data via Neural Networks and Persistent Homology](http://arxiv.org/pdf/2509.09140v1)

Authors: Dylan Peek, Matthew P. Skerritt, Stephan Chalup

Persistent Homology (PH) and Artificial Neural Networks (ANNs) offer
contrasting approaches to inferring topological structure from data. In this
study, we examine the noise robustness of a supervised neural network trained
to predict Betti numbers in 2D binary images. We compare an ANN approach
against a PH pipeline based on cubical complexes and the Signed Euclidean
Distance Transform (SEDT), which is a widely adopted strategy for noise-robust
topological analysis. Using one synthetic and two real-world datasets, we show
that ANNs can outperform this PH approach under noise, likely due to their
capacity to learn contextual and geometric priors from training data. Though
still emerging, the use of ANNs for topology estimation offers a compelling
alternative to PH under structural noise.

### 10. [RT-DETR++ for UAV Object Detection](http://arxiv.org/pdf/2509.09157v1)

Authors: Yuan Shufang

Object detection in unmanned aerial vehicle (UAV) imagery presents
significant challenges. Issues such as densely packed small objects, scale
variations, and occlusion are commonplace. This paper introduces RT-DETR++,
which enhances the encoder component of the RT-DETR model. Our improvements
focus on two key aspects. First, we introduce a channel-gated attention-based
upsampling/downsampling (AU/AD) mechanism. This dual-path system minimizes
errors and preserves details during feature layer propagation. Second, we
incorporate CSP-PAC during feature fusion. This technique employs parallel
hollow convolutions to process local and contextual information within the same
layer, facilitating the integration of multi-scale features. Evaluation
demonstrates that our novel neck design achieves superior performance in
detecting small and densely packed objects. The model maintains sufficient
speed for real-time detection without increasing computational complexity. This
study provides an effective approach for feature encoding design in real-time
detection systems.

### Computers and Society

### 1. [Content Moderation Futures](http://arxiv.org/pdf/2509.09076v1)

Authors: Lindsay Blackwell

This study examines the failures and possibilities of contemporary social
media governance through the lived experiences of various content moderation
professionals. Drawing on participatory design workshops with 33 practitioners
in both the technology industry and broader civil society, this research
identifies significant structural misalignments between corporate incentives
and public interests. While experts agree that successful content moderation is
principled, consistent, contextual, proactive, transparent, and accountable,
current technology companies fail to achieve these goals, due in part to
exploitative labor practices, chronic underinvestment in user safety, and
pressures of global scale. I argue that successful governance is undermined by
the pursuit of technological novelty and rapid growth, resulting in platforms
that necessarily prioritize innovation and expansion over public trust and
safety. To counter this dynamic, I revisit the computational history of care
work, to motivate present-day solidarity amongst platform governance workers
and inspire systemic change.

### 2. [Quantum Machine Learning, Quantitative Trading, Reinforcement Learning, Deep Learning](http://arxiv.org/pdf/2509.09176v1)

Authors: Jun-Hao Chen, Yu-Chien Huang, Yun-Cheng Tsai, Samuel Yen-Chi Chen

The convergence of quantum-inspired neural networks and deep reinforcement
learning offers a promising avenue for financial trading. We implemented a
trading agent for USD/TWD by integrating Quantum Long Short-Term Memory (QLSTM)
for short-term trend prediction with Quantum Asynchronous Advantage
Actor-Critic (QA3C), a quantum-enhanced variant of the classical A3C. Trained
on data from 2000-01-01 to 2025-04-30 (80\% training, 20\% testing), the
long-only agent achieves 11.87\% return over around 5 years with 0.92\% max
drawdown, outperforming several currency ETFs. We detail state design (QLSTM
features and indicators), reward function for trend-following/risk control, and
multi-core training. Results show hybrid models yield competitive FX trading
performance. Implications include QLSTM's effectiveness for small-profit trades
with tight risk and future enhancements. Key hyperparameters: QLSTM sequence
length$=$4, QA3C workers$=$8. Limitations: classical quantum simulation and
simplified strategy. \footnote{The views expressed in this article are those of
the authors and do not represent the views of Wells Fargo. This article is for
informational purposes only. Nothing contained in this article should be
construed as investment advice. Wells Fargo makes no express or implied
warranties and expressly disclaims all legal, tax, and accounting implications
related to this article.

### 3. [Digital Iran Reloaded: Gamer Mitigation Tactics of IRI Information Controls](http://arxiv.org/pdf/2509.09063v1)

Authors: Melinda Cohoon

Internet censorship in the Islamic Republic of Iran restricts access to
global platforms and services, forcing users to rely on circumvention
technologies such as VPNs, proxies, and tunneling tools. This report presents
findings from a mixed-methods study of 660 Iranian internet users, with a focus
on gamers as a digitally literate and socially networked community. Survey data
are combined with network measurements of latency and VPN performance to
identify both technical and social strategies of circumvention. Results show
that while younger users report higher confidence with circumvention, peer
networks, rather than formal training, are the strongest predictors of
resilience. Gaming communities, particularly those active on platforms such as
Discord and Telegram, serve as hubs for sharing tactics and lowering barriers
to adoption. These findings extend existing work on usable security and
censorship circumvention by highlighting the intersection of infrastructural
conditions and social learning. The study concludes with design and policy
implications for developers, researchers, and funders working on digital rights
and information controls.

### 4. [Incorporating AI Incident Reporting into Telecommunications Law and Policy: Insights from India](http://arxiv.org/pdf/2509.09508v1)

Authors: Avinash Agarwal, Manisha J. Nene

The integration of artificial intelligence (AI) into telecommunications
infrastructure introduces novel risks, such as algorithmic bias and
unpredictable system behavior, that fall outside the scope of traditional
cybersecurity and data protection frameworks. This paper introduces a precise
definition and a detailed typology of telecommunications AI incidents,
establishing them as a distinct category of risk that extends beyond
conventional cybersecurity and data protection breaches. It argues for their
recognition as a distinct regulatory concern. Using India as a case study for
jurisdictions that lack a horizontal AI law, the paper analyzes the country's
key digital regulations. The analysis reveals that India's existing legal
instruments, including the Telecommunications Act, 2023, the CERT-In Rules, and
the Digital Personal Data Protection Act, 2023, focus on cybersecurity and data
breaches, creating a significant regulatory gap for AI-specific operational
incidents, such as performance degradation and algorithmic bias. The paper also
examines structural barriers to disclosure and the limitations of existing AI
incident repositories. Based on these findings, the paper proposes targeted
policy recommendations centered on integrating AI incident reporting into
India's existing telecom governance. Key proposals include mandating reporting
for high-risk AI failures, designating an existing government body as a nodal
agency to manage incident data, and developing standardized reporting
frameworks. These recommendations aim to enhance regulatory clarity and
strengthen long-term resilience, offering a pragmatic and replicable blueprint
for other nations seeking to govern AI risks within their existing sectoral
frameworks.

### 5. [Explaining the Reputational Risks of AI-Mediated Communication: Messages Labeled as AI-Assisted Are Viewed as Less Diagnostic of the Sender's Moral Character](http://arxiv.org/pdf/2509.09645v1)

Authors: Pranav Khadpe, Kimi Wenzel, George Loewenstein, Geoff Kaufman

When someone sends us a thoughtful message, we naturally form judgments about
their character. But what happens when that message carries a label indicating
it was written with the help of AI? This paper investigates how the appearance
of AI assistance affects our perceptions of message senders. Adding nuance to
previous research, through two studies (N=399) featuring vignette scenarios, we
find that AI-assistance labels don't necessarily make people view senders
negatively. Rather, they dampen the strength of character signals in
communication. We show that when someone sends a warmth-signalling message
(like thanking or apologizing) without AI help, people more strongly categorize
the sender as warm. At the same time, when someone sends a coldness-signalling
message (like bragging or blaming) without assistance, people more confidently
categorize them as cold. Interestingly, AI labels weaken both these
associations: An AI-assisted apology makes the sender appear less warm than if
they had written it themselves, and an AI-assisted blame makes the sender
appear less cold than if they had composed it independently. This supports our
signal diagnosticity explanation: messages labeled as AI-assisted are viewed as
less diagnostic than messages which seem unassisted. We discuss how our
findings shed light on the causal origins of previously reported observations
in AI-Mediated Communication.

### 6. [Personality-Enhanced Social Recommendations in SAMI: Exploring the Role of Personality Detection in Matchmaking](http://arxiv.org/pdf/2509.09583v1)

Authors: Brittany Harbison, Samuel Taubman, Travis Taylor, Ashok. K. Goel

Social connection is a vital part of learning, yet online course environments
present barriers to the organic formation of social groups. SAMI offers one
solution by facilitating student connections, but its effectiveness is
constrained by an incomplete Theory of Mind, limiting its ability to create an
effective mental model of a student. One facet of this is its inability to
intuit personality, which may influence the relevance of its recommendations.
To explore this, we propose a personality detection model utilizing GPTs
zero-shot capability to infer Big-Five personality traits from forum
introduction posts, often encouraged in online courses. We benchmark its
performance against established models, demonstrating its efficacy in this
task. Furthermore, we integrate this model into SAMIs entity-based matchmaking
system, enabling personality-informed social recommendations. Initial
integration suggests personality traits can complement existing matching
factors, though additional evaluation is required to determine their full
impact on student engagement and match quality.

### Databases

### 1. [Koza and Koza-Hub for born-interoperable knowledge graph generation using KGX](http://arxiv.org/pdf/2509.09096v1)

Authors: Daniel R Korn, Patrick Golden, Aaron Odell, Katherina Cortes, Shilpa Sundar, Kevin Schaper, Sarah Gehrke, Corey Cox, Harry Caufield, Justin Reese, Evan Morris, Christopher J Mungall, Melissa Haendel

Knowledge graph construction has become an essential domain for the future of
biomedical research. But current approaches demand a high amount of redundant
labor. These redundancies are the result of the lack of data standards and
"knowledge-graph ready" data from sources. Using the KGX standard, we aim to
solve these issues. Herein we introduce Koza and the Koza-Hub, a Python
software package which streamlines ingesting raw biomedical information into
the KGX format, and an associated set of conversion processes for thirty gold
standard biomedical data sources. Our approach is to turn knowledge graph
ingests into a set of primitive operations, provide configuration through YAML
files, and enforce compliance with the chosen data schema.

### 2. [Let's Simply Count: Quantifying Distributional Similarity Between Activities in Event Data](http://arxiv.org/pdf/2509.09440v1)

Authors: Henrik Kirchmann, Stephan A. Fahrenkrog-Petersen, Xixi Lu, Matthias Weidlich

To obtain insights from event data, advanced process mining methods assess
the similarity of activities to incorporate their semantic relations into the
analysis. Here, distributional similarity that captures similarity from
activity co-occurrences is commonly employed. However, existing work for
distributional similarity in process mining adopt neural network-based
approaches as developed for natural language processing, e.g., word2vec and
autoencoders. While these approaches have been shown to be effective, their
downsides are high computational costs and limited interpretability of the
learned representations.
  In this work, we argue for simplicity in the modeling of distributional
similarity of activities. We introduce count-based embeddings that avoid a
complex training process and offer a direct interpretable representation. To
underpin our call for simple embeddings, we contribute a comprehensive
benchmarking framework, which includes means to assess the intrinsic quality of
embeddings, their performance in downstream applications, and their
computational efficiency. In experiments that compare against the state of the
art, we demonstrate that count-based embeddings provide a highly effective and
efficient basis for distributional similarity between activities in event data.

### 3. [Database Views as Explanations for Relational Deep Learning](http://arxiv.org/pdf/2509.09482v1)

Authors: Agapi Rissaki, Ilias Fountalis, Wolfgang Gatterbauer, Benny Kimelfeld

In recent years, there has been significant progress in the development of
deep learning models over relational databases, including architectures based
on heterogeneous graph neural networks (hetero-GNNs) and heterogeneous graph
transformers. In effect, such architectures state how the database records and
links (e.g., foreign-key references) translate into a large, complex numerical
expression, involving numerous learnable parameters. This complexity makes it
hard to explain, in human-understandable terms, how a model uses the available
data to arrive at a given prediction. We present a novel framework for
explaining machine-learning models over relational databases, where
explanations are view definitions that highlight focused parts of the database
that mostly contribute to the model's prediction. We establish such global
abductive explanations by adapting the classic notion of determinacy by Nash,
Segoufin, and Vianu (2010). In addition to tuning the tradeoff between
determinacy and conciseness, the framework allows controlling the level of
granularity by adopting different fragments of view definitions, such as ones
highlighting whole columns, foreign keys between tables, relevant groups of
tuples, and so on. We investigate the realization of the framework in the case
of hetero-GNNs. We develop heuristic algorithms that avoid the exhaustive
search over the space of all databases. We propose techniques that are
model-agnostic, and others that are tailored to hetero-GNNs via the notion of
learnable masking. Our approach is evaluated through an extensive empirical
study on the RelBench collection, covering a variety of domains and different
record-level tasks. The results demonstrate the usefulness of the proposed
explanations, as well as the efficiency of their generation.

### Distributed, Parallel, and Cluster Computing

### 1. [Coherence-Aware Task Graph Modeling for Realistic Application](http://arxiv.org/pdf/2509.09094v1)

Authors: Guochu Xiong, Xiangzhong Luo, Weichen Liu

As multicore systems continue to scale, cache coherence has emerged as a
critical determinant of system performance, with coherence behavior and task
execution closely intertwined, reshaping inter-task dependencies. Task graph
modeling provides a structured way to capture such dependencies and serves as
the foundation for many system-level design strategies. However, these
strategies typically rely on predefined task graphs, while many real-world
applications lack explicit graphs and exhibit dynamic, data-dependent behavior,
limiting the effectiveness of static approaches. To address this, several task
graph modeling methods for realistic workloads have been developed. Yet, they
either rely on implicit techniques that use application-specific features
without producing explicit graphs, or they generate graphs tailored to fixed
scheduling models, which limits generality. More importantly, they often
overlook coherence interactions, creating a gap between design assumptions and
actual runtime behavior. To overcome these limitations, we propose CoTAM, a
Coherence-Aware Task Graph Modeling framework for realistic workloads that
constructs a unified task graph reflecting runtime behavior. CoTAM analyzes the
impact of coherence by decoupling its effects from overall execution,
quantifies its influence through a learned weighting scheme, and infers
inter-task dependencies for coherence-aware graph generation. Extensive
experiments show that CoTAM outperforms implicit methods, bridging the gap
between dynamic workload behavior and existing designs while demonstrating the
importance of incorporating cache coherence into task graph modeling for
accurate and generalizable system-level analysis.

### 2. [Barycentric Coded Distributed Computing with Flexible Recovery Threshold for Collaborative Mobile Edge Computing](http://arxiv.org/pdf/2509.09435v1)

Authors: Houming Qiu, Kun Zhu, Dusit Niyato, Nguyen Cong Luong, Changyan Yi, Chen Dai

Collaborative mobile edge computing (MEC) has emerged as a promising paradigm
to enable low-capability edge nodes to cooperatively execute
computation-intensive tasks. However, straggling edge nodes (stragglers)
significantly degrade the performance of MEC systems by prolonging computation
latency. While coded distributed computing (CDC) as an effective technique is
widely adopted to mitigate straggler effects, existing CDC schemes exhibit two
critical limitations: (i) They cannot successfully decode the final result
unless the number of received results reaches a fixed recovery threshold, which
seriously restricts their flexibility; (ii) They suffer from inherent poles in
their encoding/decoding functions, leading to decoding inaccuracies and
numerical instability in the computational results. To address these
limitations, this paper proposes an approximated CDC scheme based on
barycentric rational interpolation. The proposed CDC scheme offers several
outstanding advantages. Firstly, it can decode the final result leveraging any
returned results from workers. Secondly, it supports computations over both
finite and real fields while ensuring numerical stability. Thirdly, its
encoding/decoding functions are free of poles, which not only enhances
approximation accuracy but also achieves flexible accuracy tuning. Fourthly, it
integrates a novel BRI-based gradient coding algorithm accelerating the
training process while providing robustness against stragglers. Finally,
experimental results reveal that the proposed scheme is superior to existing
CDC schemes in both waiting time and approximate accuracy.

### 3. [Weaker Assumptions for Asymmetric Trust](http://arxiv.org/pdf/2509.09493v1)

Authors: Ignacio Amores-Sesar, Christian Cachin, Juan Villacis

In distributed systems with asymmetric trust, each participant is free to
make its own trust assumptions about others, captured by an asymmetric quorum
system. This contrasts with ordinary, symmetric quorum systems and threshold
models, where trust assumptions are uniformly shared among participants.
Fundamental problems like reliable broadcast and consensus are unsolvable in
the asymmetric model if quorum systems satisfy only the classical properties of
consistency and availability. Existing approaches overcome this by introducing
stronger assumptions. We show that some of these assumptions are overly
restrictive, so much so that they effectively eliminate the benefits of
asymmetric trust. To address this, we propose a new approach to characterize
asymmetric problems and, building upon it, present algorithms for reliable
broadcast and consensus that require weaker assumptions than previous
solutions. Our methods are general and can be extended to other core problems
in systems with asymmetric trust.

### 4. [WebAssembly and Unikernels: A Comparative Study for Serverless at the Edge](http://arxiv.org/pdf/2509.09400v1)

Authors: Valerio Besozzi, Enrico Fiasco, Marco Danelutto, Patrizio Dazzi

Serverless computing at the edge requires lightweight execution environments
to minimize cold start latency, especially in Urgent Edge Computing (UEC). This
paper compares WebAssembly and unikernel-based MicroVMs for serverless
workloads. We present Limes, a WebAssembly runtime built on Wasmtime, and
evaluate it against the Firecracker-based environment used in SPARE. Results
show that WebAssembly offers lower cold start times for lightweight functions
but suffers with complex workloads, while Firecracker provides higher, but
stable, cold starts and better execution performance, particularly for
I/O-heavy tasks.

### 5. [TrEnv: Transparently Share Serverless Execution Environments Across Different Functions and Nodes](http://arxiv.org/pdf/2509.09525v1)

Authors: Jialiang Huang, Teng Ma, Zheng Liu, Sixing Lin, Kang Chen, Jinlei Jiang, Xia Liao, Yingdi Shan, Yongwei Wu, Ning Zhang, Mengting Lu, Tao Ma, Haifeng Gong, Mingxing Zhang

Serverless computing provides dynamic scalability, but its infrastructure
overhead becomes a bottleneck for emerging workloads such as LLM agents, which
exhibit unpredictable invocation patterns and variable resource demands. Our
analysis shows that for these agents, the cost of running on serverless
platforms can reach up to 70% of the cost of LLM API calls. This finding
motivates the need for a more efficient, high-density serverless platform. We
present TrEnv, a co-designed serverless platform that supports both container-
and VM-based environments, optimized for the unique demands of LLM agents.
TrEnv reduces startup latency and memory usage through repurposable sandboxes
and memory templates, which enable fast reuse and restoration of execution
environments. To further reduce overhead in VM-based agent workloads, TrEnv
leverages browser sharing and a page cache bypassing mechanism. Evaluations
show that TrEnv reduces P99 latency by up to 7X and memory usage by 48% in
container-based settings, and achieves up to 58% lower P99 latency and 61%
memory savings for VM-based agents compared to state-of-the-art systems like
E2B.

### 6. [ProDiGy: Proximity- and Dissimilarity-Based Byzantine-Robust Federated Learning](http://arxiv.org/pdf/2509.09534v1)

Authors: Sena Ergisi, Luis Maßny, Rawad Bitar

Federated Learning (FL) emerged as a widely studied paradigm for distributed
learning. Despite its many advantages, FL remains vulnerable to adversarial
attacks, especially under data heterogeneity. We propose a new Byzantine-robust
FL algorithm called ProDiGy. The key novelty lies in evaluating the client
gradients using a joint dual scoring system based on the gradients' proximity
and dissimilarity. We demonstrate through extensive numerical experiments that
ProDiGy outperforms existing defenses in various scenarios. In particular, when
the clients' data do not follow an IID distribution, while other defense
mechanisms fail, ProDiGy maintains strong defense capabilities and model
accuracy. These findings highlight the effectiveness of a dual perspective
approach that promotes natural similarity among honest clients while detecting
suspicious uniformity as a potential indicator of an attack.

### 7. [Towards A High-Performance Quantum Data Center Network Architecture](http://arxiv.org/pdf/2509.09653v1)

Authors: Yufeng Xin, Liang Zhang

Quantum Data Centers (QDCs) are needed to support large-scale quantum
processing for both academic and commercial applications. While large-scale
quantum computers are constrained by technological and financial barriers, a
modular approach that clusters small quantum computers offers an alternative.
This approach, however, introduces new challenges in network scalability,
entanglement generation, and quantum memory management. In this paper, we
propose a three-layer fat-tree network architecture for QDCs, designed to
address these challenges. Our architecture features a unique leaf switch and an
advanced swapping spine switch design, optimized to handle high volumes of
entanglement requests as well as a queue scheduling mechanism that efficiently
manages quantum memory to prevent decoherence. Through queuing-theoretical
models and simulations in NetSquid, we demonstrate the proposed architecture's
scalability and effectiveness in maintaining high entanglement fidelity,
offering a practical path forward for modular QDC networks.

### Digital Libraries

### 1. [How much are LLMs changing the language of academic papers after ChatGPT? A multi-database and full text analysis](http://arxiv.org/pdf/2509.09596v1)

Authors: Kayvan Kousha, Mike Thelwall

This study investigates how Large Language Models (LLMs) are influencing the
language of academic papers by tracking 12 LLM-associated terms across six
major scholarly databases (Scopus, Web of Science, PubMed, PubMed Central
(PMC), Dimensions, and OpenAlex) from 2015 to 2024. Using over 2.4 million PMC
open-access publications (2021-July 2025), we also analysed full texts to
assess changes in the frequency and co-occurrence of these terms before and
after ChatGPT's initial public release. Across databases, delve (+1,500%),
underscore (+1,000%), and intricate (+700%) had the largest increases between
2022 and 2024. Growth in LLM-term usage was much higher in STEM fields than in
social sciences and arts and humanities. In PMC full texts, the proportion of
papers using underscore six or more times increased by over 10,000% from 2022
to 2025, followed by intricate (+5,400%) and meticulous (+2,800%). Nearly half
of all 2024 PMC papers using any LLM term also included underscore, compared
with only 3%-14% of papers before ChatGPT in 2022. Papers using one LLM term
are now much more likely to include other terms. For example, in 2024,
underscore strongly correlated with pivotal (0.449) and delve (0.311), compared
with very weak associations in 2022 (0.032 and 0.018, respectively). These
findings provide the first large-scale evidence based on full-text publications
and multiple databases that some LLM-related terms are now being used much more
frequently and together. The rapid uptake of LLMs to support scholarly
publishing is a welcome development reducing the language barrier to academic
publishing for non-English speakers.

### Discrete Mathematics

### 1. [Orthogonal Latin Squares of Order Ten with Two Relations: A SAT Investigation](http://arxiv.org/pdf/2509.09633v1)

Authors: Curtis Bright, Amadou Keita, Brett Stevens

A $k$-net($n$) is a combinatorial design equivalent to $k-2$ mutually
orthogonal Latin squares of order $n$. A relation in a net is a linear
dependency over $\mathbb{F}_2$ in the incidence matrix of the net. A
computational enumeration of all orthogonal pairs of Latin squares of order 10
whose corresponding nets have at least two nontrivial relations was achieved by
Delisle in 2010 and verified by an independent search of Myrvold. In this
paper, we confirm the correctness of their exhaustive enumerations with a
satisfiability (SAT) solver approach instead of using custom-written
backtracking code. Performing the enumeration using a SAT solver has at least
three advantages. First, it reduces the amount of trust necessary, as SAT
solvers produce independently-verifiable certificates that their enumerations
are complete. These certificates can be checked by formal proof verifiers that
are relatively simple pieces of software, and therefore easier to trust.
Second, it is typically more straightforward and less error-prone to use a SAT
solver over writing search code. Third, it can be more efficient to use a
SAT-based approach, as SAT solvers are highly optimized pieces of software
incorporating backtracking-with-learning for improving the efficiency of the
backtracking search. For example, the SAT solver completely enumerates all
orthogonal pairs of Latin squares of order ten with two nontrivial relations in
under 2 hours on a desktop machine, while Delisle's 2010 search used 11,700 CPU
hours. Although computer hardware was slower in 2010, this alone cannot explain
the improvement in the efficiency of our SAT-based search.

### 2. [A New Algorithm for Computing Integer Hulls of 2D Polyhedral Sets](http://arxiv.org/pdf/2509.09134v1)

Authors: Chirantan Mukherjee

The $\texttt{IntegerHull}$ function is part of Maple's
$\texttt{PolyhedralSets}$ library, which calculates the integer hull of a given
polyhedral set. This algorithm works by translating the supporting hyperplanes
of the facets of the input polyhedral set inwards till each hyperplane
encounters at least one integer point. The polyhedral set is then divided into
smaller regions and a brute force method is applied to find the remaining
vertices.
  There are certain edge case scenarios where the computational cost of the
existing algorithm can be high, for which we propose a new algorithm. We
translate the supporting hyperplanes of the facets inwards from the (relative)
opposite vertex till the hyperplanes encounter at least one integer point.
Then, we can follow the same procedure as the old algorithm or use a recursive
technique on the smaller regions.
  The edge case scenarios mentioned above occurs when there are integer points
present on the supporting hyperplanes of the facets of the polyhedral set. This
increases the region on which the brute force method is applied to find the
remaining vertices.

### 3. [Discrepancy Beyond Additive Functions with Applications to Fair Division](http://arxiv.org/pdf/2509.09252v1)

Authors: Alexandros Hollender, Pasin Manurangsi, Raghu Meka, Warut Suksompong

We consider a setting where we have a ground set $M$ together with
real-valued set functions $f_1, \dots, f_n$, and the goal is to partition $M$
into two sets $S_1,S_2$ such that $|f_i(S_1) - f_i(S_2)|$ is small for every
$i$. Many results in discrepancy theory can be stated in this form with the
functions $f_i$ being additive. In this work, we initiate the study of the
unstructured case where $f_i$ is not assumed to be additive. We show that even
without the additivity assumption, the upper bound remains at most $O(\sqrt{n
\log n})$.
  Our result has implications on the fair allocation of indivisible goods. In
particular, we show that a consensus halving up to $O(\sqrt{n \log n})$ goods
always exists for $n$ agents with monotone utilities. Previously, only an
$O(n)$ bound was known for this setting.

### 4. [Characterization of the computed homology and cohomology bases -- technical report](http://arxiv.org/pdf/2509.09350v1)

Authors: Yann-Situ Gazull, Aldo Gonzalez-Lorenzo, Alexandra Bac

Computing homology and cohomology is at the heart of many recent works and a
key issue for topological data analysis. Among homological objects, homology
generators are useful to locate or understand holes (especially for geometric
objects). The present paper provides a characterization of the class of
homology bases that are computed by standard algorithmic methods. The proof of
this characterization relies on the Homological Discrete Vector Field, a
combinatorial structure for computing homology, which encompasses several
standard methods (persistent homology, tri-partitions, Smith Normal Form,
discrete Morse theory). These results refine the combinatorial homology theory
and provide novel ideas to gain more control over the computation of homology
generators.

### Data Structures and Algorithms

### 1. [Additive Approximation Schemes for Low-Dimensional Embeddings](http://arxiv.org/pdf/2509.09652v1)

Authors: Prashanti Anderson, Ainesh Bakshi, Samuel B. Hopkins

We consider the task of fitting low-dimensional embeddings to
high-dimensional data. In particular, we study the $k$-Euclidean Metric
Violation problem ($\textsf{$k$-EMV}$), where the input is $D \in
\mathbb{R}^{\binom{n}{2}}_{\geq 0}$ and the goal is to find the closest vector
$X \in \mathbb{M}_{k}$, where $\mathbb{M}_k \subset
\mathbb{R}^{\binom{n}{2}}_{\geq 0}$ is the set of all $k$-dimensional Euclidean
metrics on $n$ points, and closeness is formulated as the following
optimization problem, where $\| \cdot \|$ is the entry-wise $\ell_2$ norm: \[
  \textsf{OPT}_{\textrm{EMV}} = \min_{X \in \mathbb{M}_{k} } \Vert D - X
\Vert_2^2\,.\] Cayton and Dasgupta [CD'06] showed that this problem is NP-Hard,
even when $k=1$. Dhamdhere [Dha'04] obtained a $O(\log(n))$-approximation for
$\textsf{$1$-EMV}$ and leaves finding a PTAS for it as an open question
(reiterated recently by Lee [Lee'25]). Although $\textsf{$k$-EMV}$ has been
studied in the statistics community for over 70 years, under the name
"multi-dimensional scaling", there are no known efficient approximation
algorithms for $k > 1$, to the best of our knowledge.
  We provide the first polynomial-time additive approximation scheme for
$\textsf{$k$-EMV}$. In particular, we obtain an embedding with objective value
$\textsf{OPT}_{\textrm{EMV}} + \varepsilon \Vert D\Vert_2^2$ in $(n\cdot
B)^{\mathsf{poly}(k, \varepsilon^{-1})}$ time, where each entry in $D$ can be
represented by $B$ bits. We believe our algorithm is a crucial first step
towards obtaining a PTAS for $\textsf{$k$-EMV}$. Our key technical contribution
is a new analysis of correlation rounding for Sherali-Adams / Sum-of-Squares
relaxations, tailored to low-dimensional embeddings. We also show that our
techniques allow us to obtain additive approximation schemes for two related
problems: a weighted variant of $\textsf{$k$-EMV}$ and $\ell_p$ low-rank
approximation for $p>2$.

### 2. [Improved Approximation Guarantees and Hardness Results for MNL-Driven Product Ranking](http://arxiv.org/pdf/2509.09180v1)

Authors: Danny Segev, Gidi Steinberg

In this paper, we address open computational questions regarding the market
share ranking problem, recently introduced by Derakhshan et al. (2022). Their
modelling framework incorporates the extremely popular Multinomial Logit (MNL)
choice model, along with a novel search-based consider-then-choose paradigm. In
a nutshell, the authors devised a Pandora's-Box-type search model, where
different customer segments sequentially screen through a ranked list of
products, one position after the other, forming their consideration set by
including all products viewed up until terminating their inspection procedure.
Subsequently, a purchasing decision out of this set is made based on a joint
MNL choice model.
  Our main contribution consists in devising a polynomial-time approximation
scheme for the market share ranking problem, utilizing fresh technical
developments and analytical ideas, in conjunction with revising the original
insights of Derakhshan et al. (2022). Along the way, we introduce a black-box
reduction, mapping general instances of the market share ranking problem into
``bounded ratio'' instances, showing that this result directly leads to an
elegant and easily-implementable quasi-PTAS. Finally, to provide a complete
computational characterization, we prove that the market share ranking problem
is strongly $\mathrm{NP}$-hard.

### 3. [Maximizing social welfare among EF1 allocations at the presence of two types of agents](http://arxiv.org/pdf/2509.09641v1)

Authors: Jiaxuan Ma, Yong Chen, Guangting Chen, Mingyang Gong, Guohui Lin, An Zhang

We study the fair allocation of indivisible items to $n$ agents to maximize
the utilitarian social welfare, where the fairness criterion is envy-free up to
one item and there are only two different utility functions shared by the
agents. We present a $2$-approximation algorithm when the two utility functions
are normalized, improving the previous best ratio of $16 \sqrt{n}$ shown for
general normalized utility functions; thus this constant ratio approximation
algorithm confirms the APX-completeness in this special case previously shown
APX-hard. When there are only three agents, i.e., $n = 3$, the previous best
ratio is $3$ shown for general utility functions, and we present an improved
and tight $\frac 53$-approximation algorithm when the two utility functions are
normalized, and a best possible and tight $2$-approximation algorithm when the
two utility functions are unnormalized.

### Emerging Technologies

### 1. [Explaining the Reputational Risks of AI-Mediated Communication: Messages Labeled as AI-Assisted Are Viewed as Less Diagnostic of the Sender's Moral Character](http://arxiv.org/pdf/2509.09645v1)

Authors: Pranav Khadpe, Kimi Wenzel, George Loewenstein, Geoff Kaufman

When someone sends us a thoughtful message, we naturally form judgments about
their character. But what happens when that message carries a label indicating
it was written with the help of AI? This paper investigates how the appearance
of AI assistance affects our perceptions of message senders. Adding nuance to
previous research, through two studies (N=399) featuring vignette scenarios, we
find that AI-assistance labels don't necessarily make people view senders
negatively. Rather, they dampen the strength of character signals in
communication. We show that when someone sends a warmth-signalling message
(like thanking or apologizing) without AI help, people more strongly categorize
the sender as warm. At the same time, when someone sends a coldness-signalling
message (like bragging or blaming) without assistance, people more confidently
categorize them as cold. Interestingly, AI labels weaken both these
associations: An AI-assisted apology makes the sender appear less warm than if
they had written it themselves, and an AI-assisted blame makes the sender
appear less cold than if they had composed it independently. This supports our
signal diagnosticity explanation: messages labeled as AI-assisted are viewed as
less diagnostic than messages which seem unassisted. We discuss how our
findings shed light on the causal origins of previously reported observations
in AI-Mediated Communication.

### 2. [Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles](http://arxiv.org/pdf/2509.09349v1)

Authors: Ian Nell, Shane Gilroy

Road traffic accidents remain a significant global concern, with human error,
particularly distracted and impaired driving, among the leading causes. This
study introduces a novel driver behavior classification system that uses
external observation techniques to detect indicators of distraction and
impairment. The proposed framework employs advanced computer vision
methodologies, including real-time object tracking, lateral displacement
analysis, and lane position monitoring. The system identifies unsafe driving
behaviors such as excessive lateral movement and erratic trajectory patterns by
implementing the YOLO object detection model and custom lane estimation
algorithms. Unlike systems reliant on inter-vehicular communication, this
vision-based approach enables behavioral analysis of non-connected vehicles.
Experimental evaluations on diverse video datasets demonstrate the framework's
reliability and adaptability across varying road and environmental conditions.

### Graphics

### 1. [Objectness Similarity: Capturing Object-Level Fidelity in 3D Scene Evaluation](http://arxiv.org/pdf/2509.09143v1)

Authors: Yuiko Uchida, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama

This paper presents Objectness SIMilarity (OSIM), a novel evaluation metric
for 3D scenes that explicitly focuses on "objects," which are fundamental units
of human visual perception. Existing metrics assess overall image quality,
leading to discrepancies with human perception. Inspired by neuropsychological
insights, we hypothesize that human recognition of 3D scenes fundamentally
involves attention to individual objects. OSIM enables object-centric
evaluations by leveraging an object detection model and its feature
representations to quantify the "objectness" of each object in the scene. Our
user study demonstrates that OSIM aligns more closely with human perception
compared to existing metrics. We also analyze the characteristics of OSIM using
various approaches. Moreover, we re-evaluate recent 3D reconstruction and
generation models under a standardized experimental setup to clarify
advancements in this field. The code is available at
https://github.com/Objectness-Similarity/OSIM.

### Computer Science and Game Theory

### 1. [Mechanism Design with Outliers and Predictions](http://arxiv.org/pdf/2509.09561v1)

Authors: Argyrios Deligkas, Eduard Eiben, Sophie Klumper, Guido Schäfer, Artem Tsikiridis

We initiate the study of mechanism design with outliers, where the designer
can discard $z$ agents from the social cost objective. This setting is
particularly relevant when some agents exhibit extreme or atypical preferences.
As a natural case study, we consider facility location on the line: $n$
strategic agents report their preferred locations, and a mechanism places a
facility to minimize a social cost function. In our setting, the $z$ agents
farthest from the chosen facility are excluded from the social cost. While it
may seem intuitive that discarding outliers improves efficiency, our results
reveal that the opposite can hold.
  We derive tight bounds for deterministic strategyproof mechanisms under the
two most-studied objectives: utilitarian and egalitarian social cost. Our
results offer a comprehensive view of the impact of outliers. We first show
that when $z \ge n/2$, no strategyproof mechanism can achieve a bounded
approximation for either objective. For egalitarian cost, selecting the $(z +
1)$-th order statistic is strategyproof and 2-approximate. In fact, we show
that this is best possible by providing a matching lower bound. Notably, this
lower bound of 2 persists even when the mechanism has access to a prediction of
the optimal location, in stark contrast to the setting without outliers. For
utilitarian cost, we show that strategyproof mechanisms cannot effectively
exploit outliers, leading to the counterintuitive outcome that approximation
guarantees worsen as the number of outliers increases. However, in this case,
access to a prediction allows us to design a strategyproof mechanism achieving
the best possible trade-off between consistency and robustness. Finally, we
also establish lower bounds for randomized mechanisms that are truthful in
expectation.

### 2. [Persuasion Gains and Losses from Peer Communication](http://arxiv.org/pdf/2509.09099v1)

Authors: Toygar T. Kerman, Anastas P. Tenev, Konstantin Zabarnyi

We study a Bayesian persuasion setting in which a sender wants to persuade a
critical mass of receivers by revealing partial information about the state to
them. The homogeneous binary-action receivers are located on a communication
network, and each observes the private messages sent to them and their
immediate neighbors. We examine how the sender's expected utility varies with
increased communication among receivers. We show that for general families of
networks, extending the network can strictly benefit the sender. Thus, the
sender's gain from persuasion is not monotonic in network density. Moreover,
many network extensions can achieve the upper bound on the sender's expected
utility among all networks, which corresponds to the payoff in an empty
network. This is the case in networks reflecting a clear informational
hierarchy (e.g., in global corporations), as well as in decentralized networks
in which information originates from multiple sources (e.g., influencers in
social media). Finally, we show that a slight modification to the structure of
some of these networks precludes the possibility of such beneficial extensions.
Overall, our results caution against presuming that more communication
necessarily leads to better collective outcomes.

### 3. [Maximizing social welfare among EF1 allocations at the presence of two types of agents](http://arxiv.org/pdf/2509.09641v1)

Authors: Jiaxuan Ma, Yong Chen, Guangting Chen, Mingyang Gong, Guohui Lin, An Zhang

We study the fair allocation of indivisible items to $n$ agents to maximize
the utilitarian social welfare, where the fairness criterion is envy-free up to
one item and there are only two different utility functions shared by the
agents. We present a $2$-approximation algorithm when the two utility functions
are normalized, improving the previous best ratio of $16 \sqrt{n}$ shown for
general normalized utility functions; thus this constant ratio approximation
algorithm confirms the APX-completeness in this special case previously shown
APX-hard. When there are only three agents, i.e., $n = 3$, the previous best
ratio is $3$ shown for general utility functions, and we present an improved
and tight $\frac 53$-approximation algorithm when the two utility functions are
normalized, and a best possible and tight $2$-approximation algorithm when the
two utility functions are unnormalized.

### 4. [Understanding Economic Tradeoffs Between Human and AI Agents in Bargaining Games](http://arxiv.org/pdf/2509.09071v1)

Authors: Crystal Qian, Kehang Zhu, John Horton, Benjamin S. Manning, Vivian Tsai, James Wexler, Nithum Thain

Coordination tasks traditionally performed by humans are increasingly being
delegated to autonomous agents. As this pattern progresses, it becomes critical
to evaluate not only these agents' performance but also the processes through
which they negotiate in dynamic, multi-agent environments. Furthermore,
different agents exhibit distinct advantages: traditional statistical agents,
such as Bayesian models, may excel under well-specified conditions, whereas
large language models (LLMs) can generalize across contexts. In this work, we
compare humans (N = 216), LLMs (GPT-4o, Gemini 1.5 Pro), and Bayesian agents in
a dynamic negotiation setting that enables direct, identical-condition
comparisons across populations, capturing both outcomes and behavioral
dynamics. Bayesian agents extract the highest surplus through aggressive
optimization, at the cost of frequent trade rejections. Humans and LLMs can
achieve similar overall surplus, but through distinct behaviors: LLMs favor
conservative, concessionary trades with few rejections, while humans employ
more strategic, risk-taking, and fairness-oriented behaviors. Thus, we find
that performance parity -- a common benchmark in agent evaluation -- can
conceal fundamental differences in process and alignment, which are critical
for practical deployment in real-world coordination tasks.

### 5. [Discrepancy Beyond Additive Functions with Applications to Fair Division](http://arxiv.org/pdf/2509.09252v1)

Authors: Alexandros Hollender, Pasin Manurangsi, Raghu Meka, Warut Suksompong

We consider a setting where we have a ground set $M$ together with
real-valued set functions $f_1, \dots, f_n$, and the goal is to partition $M$
into two sets $S_1,S_2$ such that $|f_i(S_1) - f_i(S_2)|$ is small for every
$i$. Many results in discrepancy theory can be stated in this form with the
functions $f_i$ being additive. In this work, we initiate the study of the
unstructured case where $f_i$ is not assumed to be additive. We show that even
without the additivity assumption, the upper bound remains at most $O(\sqrt{n
\log n})$.
  Our result has implications on the fair allocation of indivisible goods. In
particular, we show that a consensus halving up to $O(\sqrt{n \log n})$ goods
always exists for $n$ agents with monotone utilities. Previously, only an
$O(n)$ bound was known for this setting.

### Human-Computer Interaction

### 1. [User Exploration and Exploitation Behavior Under the Influence of Real-time Interactions in Live Streaming Environments](http://arxiv.org/pdf/2509.09138v1)

Authors: Akira Matsui, Kazuki Fujikawa, Ryo Sasaki, Ryo Adachi

Live streaming platforms offer a distinctive way for users and content
creators to interact with each other through real-time communication. While
research on user behavior in online platforms has explored how users discover
their favorite content from creators and engage with them, the role of
real-time features remains unclear. There are open questions as to what
commonalities and differences exist in users' relationships with live streaming
platforms compared to traditional on-demand style platforms. To understand
this, we employ the concept of Exploration/Exploitation (E/E) and analyze a
large-scale dataset from a live streaming platform over two years. Our results
indicate that even on live streaming platforms, users exhibit E/E behavior but
experience a longer exploration period. We also identify external factors, such
as circadian rhythms, that influence E/E dynamics and user loyalty. The
presented study emphasizes the importance of balancing E/E in online platform
design, especially for live streaming platforms, providing implications that
suggest design strategies for platform developers and content creators to
facilitate timely engagement and retention.

### 2. [Sensible Agent: A Framework for Unobtrusive Interaction with Proactive AR Agents](http://arxiv.org/pdf/2509.09255v1)

Authors: Geonsun Lee, Min Xia, Nels Numan, Xun Qian, David Li, Yanhe Chen, Achin Kulshrestha, Ishan Chatterjee, Yinda Zhang, Dinesh Manocha, David Kim, Ruofei Du

Proactive AR agents promise context-aware assistance, but their interactions
often rely on explicit voice prompts or responses, which can be disruptive or
socially awkward. We introduce Sensible Agent, a framework designed for
unobtrusive interaction with these proactive agents. Sensible Agent dynamically
adapts both "what" assistance to offer and, crucially, "how" to deliver it,
based on real-time multimodal context sensing. Informed by an expert workshop
(n=12) and a data annotation study (n=40), the framework leverages egocentric
cameras, multimodal sensing, and Large Multimodal Models (LMMs) to infer
context and suggest appropriate actions delivered via minimally intrusive
interaction modes. We demonstrate our prototype on an XR headset through a user
study (n=10) in both AR and VR scenarios. Results indicate that Sensible Agent
significantly reduces perceived interaction effort compared to voice-prompted
baseline, while maintaining high usability and achieving higher preference.

### 3. [Flip Co-op: Cooperative Takeovers in Shared Autonomy](http://arxiv.org/pdf/2509.09281v1)

Authors: Sandeep Banik, Naira Hovakimyan

Shared autonomy requires principled mechanisms for allocating and
transferring control between a human and an autonomous agent. Existing
approaches often rely on blending control inputs between human and autonomous
agent or switching rules, which lack theoretical guarantees. This paper
develops a game-theoretic framework for modeling cooperative takeover in shared
autonomy. We formulate the switching interaction as a dynamic game in which
authority is embedded directly into the system dynamics, resulting in Nash
equilibrium(NE)-based strategies rather than ad hoc switching rules. We
establish the existence and characterization of NE in the space of pure
takeover strategies under stochastic human intent. For the class of
linear-quadratic systems, we derive closed-form recursions for takeover
strategies and saddle-point value functions, providing analytical insight and
efficient computation of cooperative takeover policies. We further introduce a
bimatrix potential game reformulation to address scenarios where human and
autonomy utilities are not perfectly aligned, yielding a unifying potential
function that preserves tractability while capturing intent deviations. The
framework is applied to a vehicle trajectory tracking problem, demonstrating
how equilibrium takeover strategies adapt across straight and curved path
segments. The results highlight the trade-off between human adaptability and
autonomous efficiency and illustrate the practical benefits of grounding shared
autonomy in cooperative game theory.

### 4. [The Impact of Device Type, Data Practices, and Use Case Scenarios on Privacy Concerns about Eye-tracked Augmented Reality in the United States and Germany](http://arxiv.org/pdf/2509.09285v1)

Authors: Efe Bozkir, Babette Bühler, Xiaoyuan Wu, Enkelejda Kasneci, Lujo Bauer, Lorrie Faith Cranor

Augmented reality technology will likely be prevalent with more affordable
head-mounted displays. Integrating novel interaction modalities such as eye
trackers into head-mounted displays could lead to collecting vast amounts of
biometric data, which may allow inference of sensitive user attributes like
health status or sexual preference, posing privacy issues. While previous works
broadly examined privacy concerns about augmented reality, ours is the first to
extensively explore privacy concerns on behavioral data, particularly eye
tracking in augmented reality. We crowdsourced four survey studies in the
United States (n1 = 48, n2 = 525) and Germany (n3 = 48, n4 = 525) to understand
the impact of user attributes, augmented reality devices, use cases, data
practices, and country on privacy concerns. Our findings indicate that
participants are generally concerned about privacy when they know what
inferences can be made based on the collected data. Despite the more prominent
use of smartphones in daily life than augmented reality glasses, we found no
indications of differing privacy concerns depending on the device type. In
addition, our participants are more comfortable when a particular use case
benefits them and less comfortable when other humans can consume their data.
Furthermore, participants in the United States are less concerned about their
privacy than those in Germany. Based on our findings, we provide several
recommendations to practitioners and policymakers for privacy-aware augmented
reality.

### 5. [Proactive AI Adoption can be Threatening: When Help Backfires](http://arxiv.org/pdf/2509.09309v1)

Authors: Dana Harari, Ofra Amir

Artificial intelligence (AI) assistants are increasingly embedded in
workplace tools, raising the question of how initiative-taking shapes adoption.
Prior work highlights trust and expectation mismatches as barriers, but the
underlying psychological mechanisms remain unclear. Drawing on self-affirmation
and social exchange theories, we theorize that unsolicited help elicits
self-threat, reducing willingness to accept assistance, likelihood of future
use, and performance expectancy. We report two vignette-based experiments
(Study~1: $N=761$; Study~2: $N=571$, preregistered). Study~1 compared
anticipatory and reactive help provided by an AI vs. a human, while Study~2
distinguished between \emph{offering} (suggesting help) and \emph{providing}
(acting automatically). In Study 1, AI help was more threatening than human
help. Across both studies, anticipatory help increased perceived threat and
reduced adoption outcomes. Our findings identify self-threat as a mechanism
explaining why proactive AI features may backfire and suggest design
implications for AI initiative.

### 6. [Smart Device Development for Gait Monitoring: Multimodal Feedback in an Interactive Foot Orthosis, Walking Aid, and Mobile Application](http://arxiv.org/pdf/2509.09359v1)

Authors: Stefan Resch, André Kousha, Anna Carroll, Noah Severinghaus, Felix Rehberg, Marco Zatschker, Yunus Söyleyici, Daniel Sanchez-Morillo

Smart assistive technologies such as sensor-based footwear and walking aids
offer promising opportunities to support rehabilitation through real-time
feedback and patient-centered monitoring. However, most orthotic devices remain
passive and lack integrated sensing or feedback functionalities, while existing
research often focuses on isolated prototypes rather than cohesive, interactive
systems. In this work, we present the design and implementation of a novel
modular sensor system that combines a smart foot orthosis with an instrumented
forearm crutch. The system integrates plantar pressure and motion sensing,
vibrotactile feedback, and wireless communication via a smartphone application.
We conducted an experimental user study with eight participants to validate the
feasibility of the smart foot orthosis for mobile gait detection, explore the
potential of haptic feedback for user interaction, and assess the usability of
the accompanying mobile health application. Our work contributes to the field
of smart assistive technology in rehabilitation and prevention by demonstrating
a functional and comprehensive system. We further discuss system limitations,
outline potential application scenarios, and provide recommendations for future
development and clinical integration.

### 7. [Real-Time Kinematic Positioning and Optical See-Through Head-Mounted Display for Outdoor Tracking: Hybrid System and Preliminary Assessment](http://arxiv.org/pdf/2509.09412v1)

Authors: Muhannad Ismael, Maël Cornil

This paper presents an outdoor tracking system using Real-Time Kinematic
(RTK) positioning and Optical See-Through Head Mounted Display(s) (OST-HMD(s))
in urban areas where the accurate tracking of objects is critical and where
displaying occluded information is important for safety reasons. The approach
presented here replaces 2D screens/tablets and offers distinct advantages,
particularly in scenarios demanding hands-free operation. The integration of
RTK, which provides centimeter-level accuracy of tracked objects, with OST-HMD
represents a promising solution for outdoor applications. This paper provides
valuable insights into leveraging the combined potential of RTK and OST-HMD for
outdoor tracking tasks from the perspectives of systems integration,
performance optimization, and usability. The main contributions of this paper
are: \textbf{1)} a system for seamlessly merging RTK systems with OST-HMD to
enable relatively precise and intuitive outdoor tracking, \textbf{2)} an
approach to determine a global location to achieve the position relative to the
world, \textbf{3)} an approach referred to as 'semi-dynamic' for system
assessment. Moreover, we offer insights into several relevant future research
topics aimed at improving the OST-HMD and RTK hybrid system for outdoor
tracking.

### 8. [Changing the Paradigm from Dynamic Queries to LLM-generated SQL Queries with Human Intervention](http://arxiv.org/pdf/2509.09461v1)

Authors: Ambre Assor, Hyeon Jeon, Sungbok Shin, Jean-Daniel Fekete

We propose leveraging Large Language Models (LLMs) as an interaction layer
for medical visualization systems. In domains like healthcare, where users must
navigate high-dimensional, coded, and heterogeneous datasets, LLM-generated
queries enable expert medical users to express complex analytical intents in
natural language. These intents are then translated into editable and
executable queries, replacing the dynamic query interfaces used by traditional
visualization systems built around sliders, check boxes, and drop-downs. This
interaction model reduces visual clutter and eliminates the need for users to
memorize field names or system codes, supporting fluid exploration, with the
drawback of not exposing all the filtering criteria. We also reintroduce
dynamic queries on demand to better support interactive exploration. We posit
that medical users are trained to know the possible filtering options but
challenged to remember the details of the attribute names and code values. We
demonstrate this paradigm in ParcoursVis, our scalable EventFlow-inspired
patient care pathway visualization system powered by the French National Health
Data System, one of the largest health data repositories in the world.

### 9. [Cognitive Affordances in Visualization: Related Constructs, Design Factors, and Framework](http://arxiv.org/pdf/2509.09510v1)

Authors: Racquel Fygenson, Lace Padilla, Enrico Bertini

Classically, affordance research investigates how the shape of objects
communicates actions to potential users. Cognitive affordances, a subset of
this research, characterize how the design of objects influences cognitive
actions, such as information processing. Within visualization, cognitive
affordances inform how graphs' design decisions communicate information to
their readers. Although several related concepts exist in visualization, a
formal translation of affordance theory to visualization is still lacking. In
this paper, we review and translate affordance theory to visualization by
formalizing how cognitive affordances operate within a visualization context.
We also review common methods and terms, and compare related constructs to
cognitive affordances in visualization. Based on a synthesis of research from
psychology, human computer interaction, and visualization, we propose a
framework of cognitive affordances in visualization that enumerates design
decisions and reader characteristics that influence a visualization's hierarchy
of communicated information. Finally, we demonstrate how this framework can
guide the evaluation and redesign of visualizations.

### 10. [Content Moderation Futures](http://arxiv.org/pdf/2509.09076v1)

Authors: Lindsay Blackwell

This study examines the failures and possibilities of contemporary social
media governance through the lived experiences of various content moderation
professionals. Drawing on participatory design workshops with 33 practitioners
in both the technology industry and broader civil society, this research
identifies significant structural misalignments between corporate incentives
and public interests. While experts agree that successful content moderation is
principled, consistent, contextual, proactive, transparent, and accountable,
current technology companies fail to achieve these goals, due in part to
exploitative labor practices, chronic underinvestment in user safety, and
pressures of global scale. I argue that successful governance is undermined by
the pursuit of technological novelty and rapid growth, resulting in platforms
that necessarily prioritize innovation and expansion over public trust and
safety. To counter this dynamic, I revisit the computational history of care
work, to motivate present-day solidarity amongst platform governance workers
and inspire systemic change.

### Information Retrieval

### 1. [Modality Alignment with Multi-scale Bilateral Attention for Multimodal Recommendation](http://arxiv.org/pdf/2509.09114v1)

Authors: Kelin Ren, Chan-Yang Ju, Dong-Ho Lee

Multimodal recommendation systems are increasingly becoming foundational
technologies for e-commerce and content platforms, enabling personalized
services by jointly modeling users' historical behaviors and the multimodal
features of items (e.g., visual and textual). However, most existing methods
rely on either static fusion strategies or graph-based local interaction
modeling, facing two critical limitations: (1) insufficient ability to model
fine-grained cross-modal associations, leading to suboptimal fusion quality;
and (2) a lack of global distribution-level consistency, causing
representational bias. To address these, we propose MambaRec, a novel framework
that integrates local feature alignment and global distribution regularization
via attention-guided learning. At its core, we introduce the Dilated Refinement
Attention Module (DREAM), which uses multi-scale dilated convolutions with
channel-wise and spatial attention to align fine-grained semantic patterns
between visual and textual modalities. This module captures hierarchical
relationships and context-aware associations, improving cross-modal semantic
modeling. Additionally, we apply Maximum Mean Discrepancy (MMD) and contrastive
loss functions to constrain global modality alignment, enhancing semantic
consistency. This dual regularization reduces mode-specific deviations and
boosts robustness. To improve scalability, MambaRec employs a dimensionality
reduction strategy to lower the computational cost of high-dimensional
multimodal features. Extensive experiments on real-world e-commerce datasets
show that MambaRec outperforms existing methods in fusion quality,
generalization, and efficiency. Our code has been made publicly available at
https://github.com/rkl71/MambaRec.

### 2. [CESRec: Constructing Pseudo Interactions for Sequential Recommendation via Conversational Feedback](http://arxiv.org/pdf/2509.09342v1)

Authors: Yifan Wang, Shen Gao, Jiabao Fang, Rui Yan, Billy Chiu, Shuo Shang

Sequential Recommendation Systems (SRS) have become essential in many
real-world applications. However, existing SRS methods often rely on
collaborative filtering signals and fail to capture real-time user preferences,
while Conversational Recommendation Systems (CRS) excel at eliciting immediate
interests through natural language interactions but neglect historical
behavior. To bridge this gap, we propose CESRec, a novel framework that
integrates the long-term preference modeling of SRS with the real-time
preference elicitation of CRS. We introduce semantic-based pseudo interaction
construction, which dynamically updates users'historical interaction sequences
by analyzing conversational feedback, generating a pseudo-interaction sequence
that seamlessly combines long-term and real-time preferences. Additionally, we
reduce the impact of outliers in historical items that deviate from users'core
preferences by proposing dual alignment outlier items masking, which identifies
and masks such items using semantic-collaborative aligned representations.
Extensive experiments demonstrate that CESRec achieves state-of-the-art
performance by boosting strong SRS models, validating its effectiveness in
integrating conversational feedback into SRS.

### 3. [Boosting Data Utilization for Multilingual Dense Retrieval](http://arxiv.org/pdf/2509.09459v1)

Authors: Chao Huang, Fengran Mo, Yufeng Chen, Changhao Guan, Zhenrui Yue, Xinyu Wang, Jinan Xu, Kaiyu Huang

Multilingual dense retrieval aims to retrieve relevant documents across
different languages based on a unified retriever model. The challenge lies in
aligning representations of different languages in a shared vector space. The
common practice is to fine-tune the dense retriever via contrastive learning,
whose effectiveness highly relies on the quality of the negative sample and the
efficacy of mini-batch data. Different from the existing studies that focus on
developing sophisticated model architecture, we propose a method to boost data
utilization for multilingual dense retrieval by obtaining high-quality hard
negative samples and effective mini-batch data. The extensive experimental
results on a multilingual retrieval benchmark, MIRACL, with 16 languages
demonstrate the effectiveness of our method by outperforming several existing
strong baselines.

### 4. [AskDoc -- Identifying Hidden Healthcare Disparities](http://arxiv.org/pdf/2509.09622v1)

Authors: Shashank Gupta

The objective of this study is to understand the online Ask the Doctor
services medical advice on internet platforms via AskDoc, a Reddit community
that serves as a public AtD platform and study if platforms mirror existing
hurdles and partiality in healthcare across various demographic groups. We
downloaded data from January 2020 to May 2022 from AskDoc -- a subreddit, and
created regular expressions to identify self-reported demographics (Gender,
Race, and Age) from the posts, and performed statistical analysis to understand
the interaction between peers and physicians with the posters. Half of the
posts did not receive comments from peers or physicians. At least 90% of the
people disclose their gender and age, and 80% of the people do not disclose
their race. It was observed that the subreddit is dominated by adult (age group
20-39) white males. Some disparities were observed in the engagement between
the users and the posters with certain demographics. Beyond the confines of
clinics and hospitals, social media could bring patients and providers closer
together, however, as observed, current physicians participation is low
compared to posters.

### 5. [Constructing a Question-Answering Simulator through the Distillation of LLMs](http://arxiv.org/pdf/2509.09226v1)

Authors: Haipeng Liu, Ting Long, Jing Fu

The question-answering (QA) simulator is a model that mimics real student
learning behaviors and predicts their correctness of their responses to
questions. QA simulators enable educational recommender systems (ERS) to
collect large amounts of training data without interacting with real students,
thereby preventing harmful recommendations made by an undertrained ERS from
undermining actual student learning. Given the QA history, there are two
categories of solutions to predict the correctness, conducting the simulation:
(1) LLM-free methods, which apply a traditional sequential model to transfer
the QA history into a vector representation first, and make predictions based
on the representation; (2) LLM-based methods, which leverage the domain
knowledge and reasoning capability of LLM to enhence the prediction. LLM-free
methods offer fast inference but generally yield suboptimal performance. In
contrast, most LLM-based methods achieve better results, but at the cost of
slower inference speed and higher GPU memory consumption. In this paper, we
propose a method named LLM Distillation based Simulator (LDSim), which distills
domain knowledge and reasoning capability from an LLM to better assist
prediction, thereby improving simulation performance. Extensive experiments
demonstrate that our LDSim achieves strong results on both the simulation task
and the knowledge tracing (KT) task. Our code is publicly available at
https://anonymous.4open.science/r/LDSim-05A9.

### 6. [We're Still Doing It (All) Wrong: Recommender Systems, Fifteen Years Later](http://arxiv.org/pdf/2509.09414v1)

Authors: Alan Said, Maria Soledad Pera, Michael D. Ekstrand

In 2011, Xavier Amatriain sounded the alarm: recommender systems research was
"doing it all wrong" [1]. His critique, rooted in statistical misinterpretation
and methodological shortcuts, remains as relevant today as it was then. But
rather than correcting course, we added new layers of sophistication on top of
the same broken foundations. This paper revisits Amatriain's diagnosis and
argues that many of the conceptual, epistemological, and infrastructural
failures he identified still persist, in more subtle or systemic forms. Drawing
on recent work in reproducibility, evaluation methodology, environmental
impact, and participatory design, we showcase how the field's accelerating
complexity has outpaced its introspection. We highlight ongoing community-led
initiatives that attempt to shift the paradigm, including workshops, evaluation
frameworks, and calls for value-sensitive and participatory research. At the
same time, we contend that meaningful change will require not only new metrics
or better tooling, but a fundamental reframing of what recommender systems
research is for, who it serves, and how knowledge is produced and validated.
Our call is not just for technical reform, but for a recommender systems
research agenda grounded in epistemic humility, human impact, and sustainable
practice.

### 7. [Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](http://arxiv.org/pdf/2509.09651v1)

Authors: Zakaria El Kassimi, Fares Fourati, Mohamed-Slim Alouini

We study question answering in the domain of radio regulations, a legally
sensitive and high-stakes area. We propose a telecom-specific
Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge,
the first multiple-choice evaluation set for this domain, constructed from
authoritative sources using automated filtering and human validation. To assess
retrieval quality, we define a domain-specific retrieval metric, under which
our retriever achieves approximately 97% accuracy. Beyond retrieval, our
approach consistently improves generation accuracy across all tested models. In
particular, while naively inserting documents without structured retrieval
yields only marginal gains for GPT-4o (less than 1%), applying our pipeline
results in nearly a 12% relative improvement. These findings demonstrate that
carefully targeted grounding provides a simple yet strong baseline and an
effective domain-specific solution for regulatory question answering. All code
and evaluation scripts, along with our derived question-answer dataset, are
available at https://github.com/Zakaria010/Radio-RAG.

### Machine Learning

### 1. ["A 6 or a 9?": Ensemble Learning Through the Multiplicity of Performant Models and Explanations](http://arxiv.org/pdf/2509.09073v1)

Authors: Gianlucca Zuin, Adriano Veloso

Creating models from past observations and ensuring their effectiveness on
new data is the essence of machine learning. However, selecting models that
generalize well remains a challenging task. Related to this topic, the Rashomon
Effect refers to cases where multiple models perform similarly well for a given
learning problem. This often occurs in real-world scenarios, like the
manufacturing process or medical diagnosis, where diverse patterns in data lead
to multiple high-performing solutions. We propose the Rashomon Ensemble, a
method that strategically selects models from these diverse high-performing
solutions to improve generalization. By grouping models based on both their
performance and explanations, we construct ensembles that maximize diversity
while maintaining predictive accuracy. This selection ensures that each model
covers a distinct region of the solution space, making the ensemble more robust
to distribution shifts and variations in unseen data. We validate our approach
on both open and proprietary collaborative real-world datasets, demonstrating
up to 0.20+ AUROC improvements in scenarios where the Rashomon ratio is large.
Additionally, we demonstrate tangible benefits for businesses in various
real-world applications, highlighting the robustness, practicality, and
effectiveness of our approach.

### 2. [Sensitivity-LoRA: Low-Load Sensitivity-Based Fine-Tuning for Large Language Models](http://arxiv.org/pdf/2509.09119v1)

Authors: Hao Zhang, Bo Huang, Zhenjia Li, Xi Xiao, Hui Yi Leong, Zumeng Zhang, Xinwei Long, Tianyang Wang, Hao Xu

Large Language Models (LLMs) have transformed both everyday life and
scientific research. However, adapting LLMs from general-purpose models to
specialized tasks remains challenging, particularly in resource-constrained
environments. Low-Rank Adaptation (LoRA), a prominent method within
Parameter-Efficient Fine-Tuning (PEFT), has emerged as a promising approach to
LLMs by approximating model weight updates using low-rank decomposition.
However, LoRA is limited by its uniform rank ( r ) allocation to each
incremental matrix, and existing rank allocation techniques aimed at addressing
this issue remain computationally inefficient, complex, and unstable, hindering
practical applications. To address these limitations, we propose
Sensitivity-LoRA, an efficient fine-tuning method that dynamically allocates
ranks to weight matrices based on both their global and local sensitivities. It
leverages the second-order derivatives (Hessian Matrix) of the loss function to
effectively capture weight sensitivity, enabling optimal rank allocation with
minimal computational overhead. Our experimental results have demonstrated
robust effectiveness, efficiency and stability of Sensitivity-LoRA across
diverse tasks and benchmarks.

### 3. [Learning What Matters: Causal Time Series Modeling for Arctic Sea Ice Prediction](http://arxiv.org/pdf/2509.09128v1)

Authors: Emam Hossain, Md Osman Gani

Conventional machine learning and deep learning models typically rely on
correlation-based learning, which often fails to distinguish genuine causal
relationships from spurious associations, limiting their robustness,
interpretability, and ability to generalize. To overcome these limitations, we
introduce a causality-aware deep learning framework that integrates
Multivariate Granger Causality (MVGC) and PCMCI+ for causal feature selection
within a hybrid neural architecture. Leveraging 43 years (1979-2021) of Arctic
Sea Ice Extent (SIE) data and associated ocean-atmospheric variables at daily
and monthly resolutions, the proposed method identifies causally influential
predictors, prioritizes direct causes of SIE dynamics, reduces unnecessary
features, and enhances computational efficiency. Experimental results show that
incorporating causal inputs leads to improved prediction accuracy and
interpretability across varying lead times. While demonstrated on Arctic SIE
forecasting, the framework is broadly applicable to other dynamic,
high-dimensional domains, offering a scalable approach that advances both the
theoretical foundations and practical performance of causality-informed
predictive modeling.

### 4. [Peering Partner Recommendation for ISPs using Machine Learning](http://arxiv.org/pdf/2509.09146v1)

Authors: Md Ibrahim Ibne Alam, Ankur Senapati, Anindo Mahmood, Murat Yuksel, Koushik Kar

Internet service providers (ISPs) need to connect with other ISPs to provide
global connectivity services to their users. To ensure global connectivity,
ISPs can either use transit service(s) or establish direct peering
relationships between themselves via Internet exchange points (IXPs). Peering
offers more room for ISP-specific optimizations and is preferred, but it often
involves a lengthy and complex process. Automating peering partner selection
can enhance efficiency in the global Internet ecosystem. We explore the use of
publicly available data on ISPs to develop a machine learning (ML) model that
can predict whether an ISP pair should peer or not. At first, we explore public
databases, e.g., PeeringDB, CAIDA, etc., to gather data on ISPs. Then, we
evaluate the performance of three broad types of ML models for predicting
peering relationships: tree-based, neural network-based, and transformer-based.
Among these, we observe that tree-based models achieve the highest accuracy and
efficiency in our experiments. The XGBoost model trained with publicly
available data showed promising performance, with a 98% accuracy rate in
predicting peering partners. In addition, the model demonstrated great
resilience to variations in time, space, and missing data. We envision that
ISPs can adopt our method to fully automate the peering partner selection
process, thus transitioning to a more efficient and optimized Internet
ecosystem.

### 5. [Clip Your Sequences Fairly: Enforcing Length Fairness for Sequence-Level RL](http://arxiv.org/pdf/2509.09177v1)

Authors: Hanyi Mao, Quanjia Xiao, Lei Pang, Haixiao Liu

We propose FSPO (Fair Sequence Policy Optimization), a sequence-level
reinforcement learning method for LLMs that enforces length-fair clipping
directly in the importance-sampling (IS) weight space. We revisit
sequence-level RL methods and identify a mismatch when PPO/GRPO-style clipping
is transplanted to sequences: a fixed clip range systematically reweights short
vs. long responses, distorting the effective objective. Theoretically, we
formalize length fairness via a Length Reweighting Error (LRE) and prove that
small LRE yields a directional cosine guarantee between the clipped and true
updates. FSPO introduces a simple, Gaussian-motivated remedy: we clip the
sequence log-IS ratio with a band that applies a KL-corrected drift term and
scales as $\sqrt{L}$. Empirically, FSPO flattens clip rates across length bins,
stabilizes training, and outperforms all baselines across multiple evaluation
datasets.

### 6. [Unsupervised Multi-Attention Meta Transformer for Rotating Machinery Fault Diagnosis](http://arxiv.org/pdf/2509.09251v1)

Authors: Hanyang Wang, Yuxuan Yang, Hongjun Wang, Lihui Wang

The intelligent fault diagnosis of rotating mechanical equipment usually
requires a large amount of labeled sample data. However, in practical
industrial applications, acquiring enough data is both challenging and
expensive in terms of time and cost. Moreover, different types of rotating
mechanical equipment with different unique mechanical properties, require
separate training of diagnostic models for each case. To address the challenges
of limited fault samples and the lack of generalizability in prediction models
for practical engineering applications, we propose a Multi-Attention Meta
Transformer method for few-shot unsupervised rotating machinery fault diagnosis
(MMT-FD). This framework extracts potential fault representations from
unlabeled data and demonstrates strong generalization capabilities, making it
suitable for diagnosing faults across various types of mechanical equipment.
The MMT-FD framework integrates a time-frequency domain encoder and a
meta-learning generalization model. The time-frequency domain encoder predicts
status representations generated through random augmentations in the
time-frequency domain. These enhanced data are then fed into a meta-learning
network for classification and generalization training, followed by fine-tuning
using a limited amount of labeled data. The model is iteratively optimized
using a small number of contrastive learning iterations, resulting in high
efficiency. To validate the framework, we conducted experiments on a bearing
fault dataset and rotor test bench data. The results demonstrate that the
MMT-FD model achieves 99\% fault diagnosis accuracy with only 1\% of labeled
sample data, exhibiting robust generalization capabilities.

### 7. [Kriging prior Regression: A Case for Kriging-Based Spatial Features with TabPFN in Soil Mapping](http://arxiv.org/pdf/2509.09408v1)

Authors: Jonas Schmidinger, Viacheslav Barkov, Sebastian Vogel, Martin Atzmueller, Gerard B M Heuvelink

Machine learning and geostatistics are two fundamentally different frameworks
for predicting and spatially mapping soil properties. Geostatistics leverages
the spatial structure of soil properties, while machine learning captures the
relationship between available environmental features and soil properties. We
propose a hybrid framework that enriches ML with spatial context through
engineering of 'spatial lag' features from ordinary kriging. We call this
approach 'kriging prior regression' (KpR), as it follows the inverse logic of
regression kriging. To evaluate this approach, we assessed both the point and
probabilistic prediction performance of KpR, using the TabPFN model across six
fieldscale datasets from LimeSoDa. These datasets included soil organic carbon,
clay content, and pH, along with features derived from remote sensing and
in-situ proximal soil sensing. KpR with TabPFN demonstrated reliable
uncertainty estimates and more accurate predictions in comparison to several
other spatial techniques (e.g., regression/residual kriging with TabPFN), as
well as to established non-spatial machine learning algorithms (e.g., random
forest). Most notably, it significantly improved the average R2 by around 30%
compared to machine learning algorithms without spatial context. This
improvement was due to the strong prediction performance of the TabPFN
algorithm itself and the complementary spatial information provided by KpR
features. TabPFN is particularly effective for prediction tasks with small
sample sizes, common in precision agriculture, whereas KpR can compensate for
weak relationships between sensing features and soil properties when proximal
soil sensing data are limited. Hence, we conclude that KpR with TabPFN is a
very robust and versatile modelling framework for digital soil mapping in
precision agriculture.

### 8. [Composable Score-based Graph Diffusion Model for Multi-Conditional Molecular Generation](http://arxiv.org/pdf/2509.09451v1)

Authors: Anjie Qiao, Zhen Wang, Chuan Chen, DeFu Lian, Enhong Chen

Controllable molecular graph generation is essential for material and drug
discovery, where generated molecules must satisfy diverse property constraints.
While recent advances in graph diffusion models have improved generation
quality, their effectiveness in multi-conditional settings remains limited due
to reliance on joint conditioning or continuous relaxations that compromise
fidelity. To address these limitations, we propose Composable Score-based Graph
Diffusion model (CSGD), the first model that extends score matching to discrete
graphs via concrete scores, enabling flexible and principled manipulation of
conditional guidance. Building on this foundation, we introduce two score-based
techniques: Composable Guidance (CoG), which allows fine-grained control over
arbitrary subsets of conditions during sampling, and Probability Calibration
(PC), which adjusts estimated transition probabilities to mitigate train-test
mismatches. Empirical results on four molecular datasets show that CSGD
achieves state-of-the-art performance, with a 15.3% average improvement in
controllability over prior methods, while maintaining high validity and
distributional fidelity. Our findings highlight the practical advantages of
score-based modeling for discrete graph generation and its capacity for
flexible, multi-property molecular design.

### 9. [AquaCast: Urban Water Dynamics Forecasting with Precipitation-Informed Multi-Input Transformer](http://arxiv.org/pdf/2509.09458v1)

Authors: Golnoosh Abdollahinejad, Saleh Baghersalimi, Denisa-Andreea Constantinescu, Sergey Shevchik, David Atienza

This work addresses the challenge of forecasting urban water dynamics by
developing a multi-input, multi-output deep learning model that incorporates
both endogenous variables (e.g., water height or discharge) and exogenous
factors (e.g., precipitation history and forecast reports). Unlike conventional
forecasting, the proposed model, AquaCast, captures both inter-variable and
temporal dependencies across all inputs, while focusing forecast solely on
endogenous variables. Exogenous inputs are fused via an embedding layer,
eliminating the need to forecast them and enabling the model to attend to their
short-term influences more effectively. We evaluate our approach on the
LausanneCity dataset, which includes measurements from four urban drainage
sensors, and demonstrate state-of-the-art performance when using only
endogenous variables. Performance also improves with the inclusion of exogenous
variables and forecast reports. To assess generalization and scalability, we
additionally test the model on three large-scale synthesized datasets,
generated from MeteoSwiss records, the Lorenz Attractors model, and the Random
Fields model, each representing a different level of temporal complexity across
100 nodes. The results confirm that our model consistently outperforms existing
baselines and maintains a robust and accurate forecast across both real and
synthetic datasets.

### 10. [AEGIS: An Agent for Extraction and Geographic Identification in Scholarly Proceedings](http://arxiv.org/pdf/2509.09470v1)

Authors: Om Vishesh, Harshad Khadilkar, Deepak Akkil

Keeping pace with the rapid growth of academia literature presents a
significant challenge for researchers, funding bodies, and academic societies.
To address the time-consuming manual effort required for scholarly discovery,
we present a novel, fully automated system that transitions from data discovery
to direct action. Our pipeline demonstrates how a specialized AI agent,
'Agent-E', can be tasked with identifying papers from specific geographic
regions within conference proceedings and then executing a Robotic Process
Automation (RPA) to complete a predefined action, such as submitting a
nomination form. We validated our system on 586 papers from five different
conferences, where it successfully identified every target paper with a recall
of 100% and a near perfect accuracy of 99.4%. This demonstration highlights the
potential of task-oriented AI agents to not only filter information but also to
actively participate in and accelerate the workflows of the academic community.

### Neural and Evolutionary Computing

### 1. [A modified RIME algorithm with covariance learning and diversity enhancement for numerical optimization](http://arxiv.org/pdf/2509.09529v1)

Authors: Shangqing Shi, Luoxiao Zhang, Yuchen Yin, Xiong Yang, Hoileong Lee

Metaheuristics are widely applied for their ability to provide more efficient
solutions. The RIME algorithm is a recently proposed physical-based
metaheuristic algorithm with certain advantages. However, it suffers from rapid
loss of population diversity during optimization and is prone to fall into
local optima, leading to unbalanced exploitation and exploration. To address
the shortcomings of RIME, this paper proposes a modified RIME with covariance
learning and diversity enhancement (MRIME-CD). The algorithm applies three
strategies to improve the optimization capability. First, a covariance learning
strategy is introduced in the soft-rime search stage to increase the population
diversity and balance the over-exploitation ability of RIME through the
bootstrapping effect of dominant populations. Second, in order to moderate the
tendency of RIME population to approach the optimal individual in the early
search stage, an average bootstrapping strategy is introduced into the
hard-rime puncture mechanism, which guides the population search through the
weighted position of the dominant populations, thus enhancing the global search
ability of RIME in the early stage. Finally, a new stagnation indicator is
proposed, and a stochastic covariance learning strategy is used to update the
stagnant individuals in the population when the algorithm gets stagnant, thus
enhancing the ability to jump out of the local optimal solution. The proposed
MRIME-CD algorithm is subjected to a series of validations on the CEC2017 test
set, the CEC2022 test set, and the experimental results are analyzed using the
Friedman test, the Wilcoxon rank sum test, and the Kruskal Wallis test. The
results show that MRIME-CD can effectively improve the performance of basic
RIME and has obvious superiorities in terms of solution accuracy, convergence
speed and stability.

### 2. [An improved educational competition optimizer with multi-covariance learning operators for global optimization problems](http://arxiv.org/pdf/2509.09552v1)

Authors: Baoqi Zhao, Xiong Yang, Hoileong Lee, Bowen Dong

The educational competition optimizer is a recently introduced metaheuristic
algorithm inspired by human behavior, originating from the dynamics of
educational competition within society. Nonetheless, ECO faces constraints due
to an imbalance between exploitation and exploration, rendering it susceptible
to local optima and demonstrating restricted effectiveness in addressing
complex optimization problems. To address these limitations, this study
presents an enhanced educational competition optimizer (IECO-MCO) utilizing
multi-covariance learning operators. In IECO, three distinct covariance
learning operators are introduced to improve the performance of ECO. Each
operator effectively balances exploitation and exploration while preventing
premature convergence of the population. The effectiveness of IECO is assessed
through benchmark functions derived from the CEC 2017 and CEC 2022 test suites,
and its performance is compared with various basic and improved algorithms
across different categories. The results demonstrate that IECO-MCO surpasses
the basic ECO and other competing algorithms in convergence speed, stability,
and the capability to avoid local optima. Furthermore, statistical analyses,
including the Friedman test, Kruskal-Wallis test, and Wilcoxon rank-sum test,
are conducted to validate the superiority of IECO-MCO over the compared
algorithms. Compared with the basic algorithm (improved algorithm), IECO-MCO
achieved an average ranking of 2.213 (2.488) on the CE2017 and CEC2022 test
suites. Additionally, the practical applicability of the proposed IECO-MCO
algorithm is verified by solving constrained optimization problems. The
experimental outcomes demonstrate the superior performance of IECO-MCO in
tackling intricate optimization problems, underscoring its robustness and
practical effectiveness in real-world scenarios.

### Networking and Internet Architecture

### 1. [AI Reasoning for Wireless Communications and Networking: A Survey and Perspectives](http://arxiv.org/pdf/2509.09193v1)

Authors: Haoxiang Luo, Yu Yan, Yanhui Bian, Wenjiao Feng, Ruichen Zhang, Yinqiu Liu, Jiacheng Wang, Gang Sun, Dusit Niyato, Hongfang Yu, Abbas Jamalipour, Shiwen Mao

Artificial Intelligence (AI) techniques play a pivotal role in optimizing
wireless communication networks. However, traditional deep learning approaches
often act as closed boxes, lacking the structured reasoning abilities needed to
tackle complex, multi-step decision problems. This survey provides a
comprehensive review and outlook of reasoning-enabled AI in wireless
communication networks, with a focus on Large Language Models (LLMs) and other
advanced reasoning paradigms. In particular, LLM-based agents can combine
reasoning with long-term planning, memory, tool utilization, and autonomous
cross-layer control to dynamically optimize network operations with minimal
human intervention. We begin by outlining the evolution of intelligent wireless
networking and the limitations of conventional AI methods. We then introduce
emerging AI reasoning techniques. Furthermore, we establish a classification
system applicable to wireless network tasks. We also present a layer-by-layer
examination for AI reasoning, covering the physical, data link, network,
transport, and application layers. For each part, we identify key challenges
and illustrate how AI reasoning methods can improve AI-based wireless
communication performance. Finally, we discuss key research directions for AI
reasoning toward future wireless communication networks. By combining insights
from both communications and AI, this survey aims to chart a path for
integrating reasoning techniques into the next-generation wireless networks.

### 2. [Toward quantum-safe scalable networks: an open, standards-aware key management framework](http://arxiv.org/pdf/2509.09453v1)

Authors: Ane Sanz, Asier Atutxa, David Franco, Jasone Astorga, Eduardo Jacob, Diego López

With the advent of quantum computing, the increasing threats to security
poses a great challenge to communication networks. Recent innovations in this
field resulted in promising technologies such as Quantum Key Distribution
(QKD), which enables the generation of unconditionally secure keys,
establishing secure communications between remote nodes. Additionally, QKD
networks enable the interconnection of multinode architectures, extending the
point-to-point nature of QKD. However, due to the limitations of the current
state of technology, the scalability of QKD networks remains a challenge toward
feasible implementations. When it comes to long-distance implementations,
trusted relay nodes partially solve the distance issue through the forwarding
of the distributed keys, allowing applications that do not have a direct QKD
link to securely share key material. Even though the relay procedure itself has
been extensively studied, the establishment of the relaying node path still
lacks a solution. This paper proposes an innovative network architecture that
solves the challenges of Key Management System (KMS) identification, relay path
discovery, and scalability of QKD networks by integrating Software-Defined
Networking (SDN) principles, and establishing high-level virtual KMSs (vKMS) in
each node and creating a new entity called the Quantum Security Controller
(QuSeC). The vKMS serves the end-user key requests, managing the multiple KMSs
within the node and abstracting the user from discovering the correct KMS.
Additionally, based on the high-level view of the network topology and status,
the QuSeC serves the path discovery requests from vKMSs, computing the
end-to-end (E2E) relay path and applying security policies. The paper also
provides a security analysis of the proposal, identifying the security levels
of the architecture and analyzing the core networking security properties.

### 3. [PARROT: Portable Android Reproducible traffic Observation Tool](http://arxiv.org/pdf/2509.09537v1)

Authors: Andrea Jimenez-Berenguel, Celeste Campo, Marta Moure-Garrido, Carlos Garcia-Rubio, Daniel Díaz-Sanchez, Florina Almenares

The rapid evolution of mobile security protocols and limited availability of
current datasets constrains research in app traffic analysis. This paper
presents PARROT, a reproducible and portable traffic capture system for
systematic app traffic collection using Android Virtual Devices. The system
provides automated environment setup, configurable Android versions, traffic
recording management, and labeled captures extraction with human-in-the-loop
app interaction. PARROT integrates mitmproxy for optional traffic decryption
with automated SSL/TLS key extraction, supporting flexible capture modes with
or without traffic interception. We collected a dataset of 80 apps selected
from the MAppGraph dataset list, providing traffic captures with corresponding
SSL keys for decryption analysis. Our comparative analysis between the
MAppGraph dataset (2021) and our dataset (2025) reveals app traffic pattern
evolution across 50 common apps. Key findings include migration from TLSv1.2 to
TLSv1.3 protocol, with TLSv1.3 comprising 90.0\% of TCP encrypted traffic in
2025 compared to 6.7\% in 2021. QUIC protocol adoption increased substantially,
with all 50 common apps generating QUIC traffic under normal network conditions
compared to 30 apps in 2021. DNS communications evolved from predominantly
unencrypted Do53 protocol (91.0\% in 2021) to encrypted DoT protocol (81.1\% in
2025). The open-source PARROT system enables reproducible app traffic capture
for research community adoption and provides insights into app security
protocol evolution.

### 4. [Fingerprinting Deep Packet Inspection Devices by Their Ambiguities](http://arxiv.org/pdf/2509.09081v1)

Authors: Diwen Xue, Armin Huremagic, Wayne Wang, Ram Sundara Raman, Roya Ensafi

Users around the world face escalating network interference such as
censorship, throttling, and interception, largely driven by the commoditization
and growing availability of Deep Packet Inspection (DPI) devices. Once reserved
for a few well-resourced nation-state actors, the ability to interfere with
traffic at scale is now within reach of nearly any network operator. Despite
this proliferation, our understanding of DPIs and their deployments on the
Internet remains limited -- being network intermediary leaves DPI unresponsive
to conventional host-based scanning tools, and DPI vendors actively obscuring
their products further complicates measurement efforts.
  In this work, we present a remote measurement framework, dMAP (DPI Mapper),
that derives behavioral fingerprints for DPIs to differentiate and cluster
these otherwise indistinguishable middleboxes at scale, as a first step toward
active reconnaissance of DPIs on the Internet. Our key insight is that parsing
and interpreting traffic as network intermediaries inherently involves
ambiguities -- from under-specified protocol behaviors to differing RFC
interpretations -- forcing DPI vendors into independent implementation choices
that create measurable variance among DPIs. Based on differential fuzzing, dMAP
systematically discovers, selects, and deploys specialized probes that
translate DPI internal parsing behaviors into externally observable
fingerprints. Applying dMAP to DPI deployments globally, we demonstrate its
practical feasibility, showing that even a modest set of 20-40 discriminative
probes reliably differentiates a wide range of DPI implementations, including
major nation-state censorship infrastructures and commercial DPI products. We
discuss how our fingerprinting methodology generalizes beyond censorship to
other forms of targeted interference.

### 5. [A Cyber-Twin Based Honeypot for Gathering Threat Intelligence](http://arxiv.org/pdf/2509.09222v1)

Authors: Muhammad Azmi Umer, Zhan Xuna, Yan Lin Aung, Aditya P. Mathur, Jianying Zhou

Critical Infrastructure (CI) is prone to cyberattacks. Several techniques
have been developed to protect CI against such attacks. In this work, we
describe a honeypot based on a cyber twin for a water treatment plant. The
honeypot is intended to serve as a realistic replica of a water treatment plant
that attracts potential attackers. The attacks launched on the honeypot are
recorded and analyzed for threat intelligence. The intelligence so obtained is
shared with the management of water treatment plants, who in turn may use it to
improve plant protection systems. The honeypot used here is operational and has
been attacked on several occasions using, for example, a ransomware attack that
is described in detail.

### 6. [What You Code Is What We Prove: Translating BLE App Logic into Formal Models with LLMs for Vulnerability Detection](http://arxiv.org/pdf/2509.09291v1)

Authors: Biwei Yan, Yue Zhang, Minghui Xu, Runyu Pan, Jinku Li, Xiuzhen Cheng

The application layer of Bluetooth Low Energy (BLE) is a growing source of
security vulnerabilities, as developers often neglect to implement critical
protections such as encryption, authentication, and freshness. While formal
verification offers a principled way to check these properties, the manual
effort of constructing formal models makes it impractical for large-scale
analysis. This paper introduces a key insight: BLE application security
analysis can be reframed as a semantic translation problem, i.e., from
real-world code to formal models. We leverage large language models (LLMs) not
to directly detect vulnerabilities, but to serve as translators that convert
BLE-specific code into process models verifiable by tools like ProVerif. We
implement this idea in VerifiaBLE, a system that combines static analysis,
prompt-guided LLM translation, and symbolic verification to check three core
security features: encryption, randomness, and authentication. Applied to 1,050
Android BLE apps, VerifiaBLE uncovers systemic weaknesses: only 10.2\% of apps
implement all three protections, while 53.9\% omit them entirely. Our work
demonstrates that using LLMs as structured translators can lower the barrier to
formal methods, unlocking scalable verification across security-critical
domains.

### 7. [Joint Optimisation of Load Balancing and Energy Efficiency for O-RAN Deployments](http://arxiv.org/pdf/2509.09343v1)

Authors: Mohammed M. H. Qazzaz, Abdelaziz Salama, Maryam Hafeez, Syed A. R. Zaidi

Open Radio Access Network (O-RAN) architecture provides an intrinsic
capability to exploit key performance monitoring (KPM) within Radio
Intelligence Controller (RIC) to derive network optimisation through xApps.
These xApps can leverage KPM knowledge to dynamically switch on/off the
associated RUs where such a function is supported over the E2 interface.
Several existing studies employ artificial intelligence (AI)/Machine Learning
(ML) based approaches to realise such dynamic sleeping for increased energy
efficiency (EE). Nevertheless, most of these approaches rely upon offloading
user equipment (UE) to carve out a sleeping opportunity. Such an approach
inherently creates load imbalance across the network. Such load imbalance may
impact the throughput performance of offloaded UEs as they might be allocated a
lower number of physical resource blocks (PRBs). Maintaining the same PRB
allocation while addressing the EE at the network level is a challenging task.
To that end, in this article, we present a comprehensive ML-based framework for
joint optimisation of load balancing and EE for ORAN deployments. We formulate
the problem as a multi-class classification system that predictively evaluates
potential RU configurations before optimising the EE, mapping network
conditions to three load balance categories (Well Balanced, Moderately
Balanced, Imbalanced). Our multi-threshold approach (Conservative, Moderate,
Aggressive) accommodates different operational priorities between energy
savings and performance assurance. Experimental evaluation using 4.26 million
real network measurements from simulations demonstrates that our Random Forest
model achieves 98.3% F1-macro performance, representing 195% improvement over
traditional baseline strategies.

### 8. [Towards A High-Performance Quantum Data Center Network Architecture](http://arxiv.org/pdf/2509.09653v1)

Authors: Yufeng Xin, Liang Zhang

Quantum Data Centers (QDCs) are needed to support large-scale quantum
processing for both academic and commercial applications. While large-scale
quantum computers are constrained by technological and financial barriers, a
modular approach that clusters small quantum computers offers an alternative.
This approach, however, introduces new challenges in network scalability,
entanglement generation, and quantum memory management. In this paper, we
propose a three-layer fat-tree network architecture for QDCs, designed to
address these challenges. Our architecture features a unique leaf switch and an
advanced swapping spine switch design, optimized to handle high volumes of
entanglement requests as well as a queue scheduling mechanism that efficiently
manages quantum memory to prevent decoherence. Through queuing-theoretical
models and simulations in NetSquid, we demonstrate the proposed architecture's
scalability and effectiveness in maintaining high entanglement fidelity,
offering a practical path forward for modular QDC networks.

### Robotics

### 1. [Kinetostatics and Particle-Swarm Optimization of Vehicle-Mounted Underactuated Metamorphic Loading Manipulators](http://arxiv.org/pdf/2509.09093v1)

Authors: Nan Mao, Guanglu Jia, Junpeng Chen, Emmanouil Spyrakos-Papastavridis, Jian S. Dai

Fixed degree-of-freedom (DoF) loading mechanisms often suffer from excessive
actuators, complex control, and limited adaptability to dynamic tasks. This
study proposes an innovative mechanism of underactuated metamorphic loading
manipulators (UMLM), integrating a metamorphic arm with a passively adaptive
gripper. The metamorphic arm exploits geometric constraints, enabling the
topology reconfiguration and flexible motion trajectories without additional
actuators. The adaptive gripper, driven entirely by the arm, conforms to
diverse objects through passive compliance. A structural model is developed,
and a kinetostatics analysis is conducted to investigate isomorphic grasping
configurations. To optimize performance, Particle-Swarm Optimization (PSO) is
utilized to refine the gripper's dimensional parameters, ensuring robust
adaptability across various applications. Simulation results validate the
UMLM's easily implemented control strategy, operational versatility, and
effectiveness in grasping diverse objects in dynamic environments. This work
underscores the practical potential of underactuated metamorphic mechanisms in
applications requiring efficient and adaptable loading solutions. Beyond the
specific design, this generalized modeling and optimization framework extends
to a broader class of manipulators, offering a scalable approach to the
development of robotic systems that require efficiency, flexibility, and robust
performance.

### 2. [LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots](http://arxiv.org/pdf/2509.09106v1)

Authors: Haokai Su, Haoxiang Luo, Shunpeng Yang, Kaiwen Jiang, Wei Zhang, Hua Chen

Achieving stable and robust perceptive locomotion for bipedal robots in
unstructured outdoor environments remains a critical challenge due to complex
terrain geometry and susceptibility to external disturbances. In this work, we
propose a novel reward design inspired by the Linear Inverted Pendulum Model
(LIPM) to enable perceptive and stable locomotion in the wild. The LIPM
provides theoretical guidance for dynamic balance by regulating the center of
mass (CoM) height and the torso orientation. These are key factors for
terrain-aware locomotion, as they help ensure a stable viewpoint for the
robot's camera. Building on this insight, we design a reward function that
promotes balance and dynamic stability while encouraging accurate CoM
trajectory tracking. To adaptively trade off between velocity tracking and
stability, we leverage the Reward Fusion Module (RFM) approach that prioritizes
stability when needed. A double-critic architecture is adopted to separately
evaluate stability and locomotion objectives, improving training efficiency and
robustness. We validate our approach through extensive experiments on a bipedal
robot in both simulation and real-world outdoor environments. The results
demonstrate superior terrain adaptability, disturbance rejection, and
consistent performance across a wide range of speeds and perceptual conditions.

### 3. [AEOS: Active Environment-aware Optimal Scanning Control for UAV LiDAR-Inertial Odometry in Complex Scenes](http://arxiv.org/pdf/2509.09141v1)

Authors: Jianping Li, Xinhang Xu, Zhongyuan Liu, Shenghai Yuan, Muqing Cao, Lihua Xie

LiDAR-based 3D perception and localization on unmanned aerial vehicles (UAVs)
are fundamentally limited by the narrow field of view (FoV) of compact LiDAR
sensors and the payload constraints that preclude multi-sensor configurations.
Traditional motorized scanning systems with fixed-speed rotations lack scene
awareness and task-level adaptability, leading to degraded odometry and mapping
performance in complex, occluded environments. Inspired by the active sensing
behavior of owls, we propose AEOS (Active Environment-aware Optimal Scanning),
a biologically inspired and computationally efficient framework for adaptive
LiDAR control in UAV-based LiDAR-Inertial Odometry (LIO). AEOS combines model
predictive control (MPC) and reinforcement learning (RL) in a hybrid
architecture: an analytical uncertainty model predicts future pose
observability for exploitation, while a lightweight neural network learns an
implicit cost map from panoramic depth representations to guide exploration. To
support scalable training and generalization, we develop a point cloud-based
simulation environment with real-world LiDAR maps across diverse scenes,
enabling sim-to-real transfer. Extensive experiments in both simulation and
real-world environments demonstrate that AEOS significantly improves odometry
accuracy compared to fixed-rate, optimization-only, and fully learned
baselines, while maintaining real-time performance under onboard computational
constraints. The project page can be found at
https://kafeiyin00.github.io/AEOS/.

### 4. [RENet: Fault-Tolerant Motion Control for Quadruped Robots via Redundant Estimator Networks under Visual Collapse](http://arxiv.org/pdf/2509.09283v1)

Authors: Yueqi Zhang, Quancheng Qian, Taixian Hou, Peng Zhai, Xiaoyi Wei, Kangmai Hu, Jiafu Yi, Lihua Zhang

Vision-based locomotion in outdoor environments presents significant
challenges for quadruped robots. Accurate environmental prediction and
effective handling of depth sensor noise during real-world deployment remain
difficult, severely restricting the outdoor applications of such algorithms. To
address these deployment challenges in vision-based motion control, this letter
proposes the Redundant Estimator Network (RENet) framework. The framework
employs a dual-estimator architecture that ensures robust motion performance
while maintaining deployment stability during onboard vision failures. Through
an online estimator adaptation, our method enables seamless transitions between
estimation modules when handling visual perception uncertainties. Experimental
validation on a real-world robot demonstrates the framework's effectiveness in
complex outdoor environments, showing particular advantages in scenarios with
degraded visual perception. This framework demonstrates its potential as a
practical solution for reliable robotic deployment in challenging field
conditions. Project website: https://RENet-Loco.github.io/

### 5. [AGILOped: Agile Open-Source Humanoid Robot for Research](http://arxiv.org/pdf/2509.09364v1)

Authors: Grzegorz Ficht, Luis Denninger, Sven Behnke

With academic and commercial interest for humanoid robots peaking, multiple
platforms are being developed. Through a high level of customization, they
showcase impressive performance. Most of these systems remain closed-source or
have high acquisition and maintenance costs, however. In this work, we present
AGILOped - an open-source humanoid robot that closes the gap between high
performance and accessibility. Our robot is driven by off-the-shelf
backdrivable actuators with high power density and uses standard electronic
components. With a height of 110 cm and weighing only 14.5 kg, AGILOped can be
operated without a gantry by a single person. Experiments in walking, jumping,
impact mitigation and getting-up demonstrate its viability for use in research.

### 6. [VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model](http://arxiv.org/pdf/2509.09372v1)

Authors: Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu, Donglin Wang

Vision-Language-Action (VLA) models typically bridge the gap between
perceptual and action spaces by pre-training a large-scale Vision-Language
Model (VLM) on robotic data. While this approach greatly enhances performance,
it also incurs significant training costs. In this paper, we investigate how to
effectively bridge vision-language (VL) representations to action (A). We
introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA
models on large-scale VLMs and extensive pre-training. To this end, we first
systematically analyze the effectiveness of various VL conditions and present
key findings on which conditions are essential for bridging perception and
action spaces. Based on these insights, we propose a lightweight Policy module
with Bridge Attention, which autonomously injects the optimal condition into
the action space. In this way, our method achieves high performance using only
a 0.5B-parameter backbone, without any robotic data pre-training. Extensive
experiments on both simulated and real-world robotic benchmarks demonstrate
that VLA-Adapter not only achieves state-of-the-art level performance, but also
offers the fast inference speed reported to date. Furthermore, thanks to the
proposed advanced bridging paradigm, VLA-Adapter enables the training of a
powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly
lowering the barrier to deploying the VLA model. Project page:
https://vla-adapter.github.io/.

### 7. [A Hybrid Hinge-Beam Continuum Robot with Passive Safety Capping for Real-Time Fatigue Awareness](http://arxiv.org/pdf/2509.09404v1)

Authors: Tongshun Chen, Zezhou Sun, Yanhan Sun, Yuhao Wang, Dezhen Song, Ke Wu

Cable-driven continuum robots offer high flexibility and lightweight design,
making them well-suited for tasks in constrained and unstructured environments.
However, prolonged use can induce mechanical fatigue from plastic deformation
and material degradation, compromising performance and risking structural
failure. In the state of the art, fatigue estimation of continuum robots
remains underexplored, limiting long-term operation. To address this, we
propose a fatigue-aware continuum robot with three key innovations: (1) a
Hybrid Hinge-Beam structure where TwistBeam and BendBeam decouple torsion and
bending: passive revolute joints in the BendBeam mitigate stress concentration,
while TwistBeam's limited torsional deformation reduces BendBeam stress
magnitude, enhancing durability; (2) a Passive Stopper that safely constrains
motion via mechanical constraints and employs motor torque sensing to detect
corresponding limit torque, ensuring safety and enabling data collection; and
(3) a real-time fatigue-awareness method that estimates stiffness from motor
torque at the limit pose, enabling online fatigue estimation without additional
sensors. Experiments show that the proposed design reduces fatigue accumulation
by about 49% compared with a conventional design, while passive mechanical
limiting combined with motor-side sensing allows accurate estimation of
structural fatigue and damage. These results confirm the effectiveness of the
proposed architecture for safe and reliable long-term operation.

### 8. [SMapper: A Multi-Modal Data Acquisition Platform for SLAM Benchmarking](http://arxiv.org/pdf/2509.09509v1)

Authors: Pedro Miguel Bastos Soares, Ali Tourani, Miguel Fernandez-Cortizas, Asier Bikandi Noya, Jose Luis Sanchez-Lopez, Holger Voos

Advancing research in fields like Simultaneous Localization and Mapping
(SLAM) and autonomous navigation critically depends on reliable and
reproducible multimodal datasets. While several influential datasets have
driven progress in these domains, they often suffer from limitations in sensing
modalities, environmental diversity, and the reproducibility of the underlying
hardware setups. To address these challenges, this paper introduces SMapper, a
novel open-hardware, multi-sensor platform designed explicitly for, though not
limited to, SLAM research. The device integrates synchronized LiDAR,
multi-camera, and inertial sensing, supported by a robust calibration and
synchronization pipeline that ensures precise spatio-temporal alignment across
modalities. Its open and replicable design allows researchers to extend its
capabilities and reproduce experiments across both handheld and robot-mounted
scenarios. To demonstrate its practicality, we additionally release
SMapper-light, a publicly available SLAM dataset containing representative
indoor and outdoor sequences. The dataset includes tightly synchronized
multimodal data and ground-truth trajectories derived from offline LiDAR-based
SLAM with sub-centimeter accuracy, alongside dense 3D reconstructions.
Furthermore, the paper contains benchmarking results on state-of-the-art LiDAR
and visual SLAM frameworks using the SMapper-light dataset. By combining
open-hardware design, reproducible data collection, and comprehensive
benchmarking, SMapper establishes a robust foundation for advancing SLAM
algorithm development, evaluation, and reproducibility.

### 9. [A Neuromorphic Incipient Slip Detection System using Papillae Morphology](http://arxiv.org/pdf/2509.09546v1)

Authors: Yanhui Lu, Zeyu Deng, Stephen J. Redmond, Efi Psomopoulou, Benjamin Ward-Cherrier

Detecting incipient slip enables early intervention to prevent object
slippage and enhance robotic manipulation safety. However, deploying such
systems on edge platforms remains challenging, particularly due to energy
constraints. This work presents a neuromorphic tactile sensing system based on
the NeuroTac sensor with an extruding papillae-based skin and a spiking
convolutional neural network (SCNN) for slip-state classification. The SCNN
model achieves 94.33% classification accuracy across three classes (no slip,
incipient slip, and gross slip) in slip conditions induced by sensor motion.
Under the dynamic gravity-induced slip validation conditions, after temporal
smoothing of the SCNN's final-layer spike counts, the system detects incipient
slip at least 360 ms prior to gross slip across all trials, consistently
identifying incipient slip before gross slip occurs. These results demonstrate
that this neuromorphic system has stable and responsive incipient slip
detection capability.

### 10. [MOFU: Development of a MOrphing Fluffy Unit with Expansion and Contraction Capabilities and Evaluation of the Animacy of Its Movements](http://arxiv.org/pdf/2509.09613v1)

Authors: Taisei Mogi, Mari Saito, Yoshihiro Nakata

Robots for therapy and social interaction are often intended to evoke
"animacy" in humans. While many robots imitate appearance and joint movements,
little attention has been given to whole-body expansion-contraction,
volume-changing movements observed in living organisms, and their effect on
animacy perception. We developed a mobile robot called "MOFU (Morphing Fluffy
Unit)," capable of whole-body expansion-contraction with a single motor and
covered with a fluffy exterior. MOFU employs a "Jitterbug" structure, a
geometric transformation mechanism that enables smooth volume change in
diameter from 210 to 280 mm using one actuator. It is also equipped with a
differential two-wheel drive mechanism for locomotion. To evaluate the effect
of expansion-contraction movements, we conducted an online survey using videos
of MOFU's behavior. Participants rated impressions with the Godspeed
Questionnaire Series. First, we compared videos of MOFU in a stationary state
with and without expansion-contraction and turning, finding that
expansion-contraction significantly increased perceived animacy. Second, we
hypothesized that presenting two MOFUs would increase animacy compared with a
single robot; however, this was not supported, as no significant difference
emerged. Exploratory analyses further compared four dual-robot motion
conditions. Third, when expansion-contraction was combined with locomotion,
animacy ratings were higher than locomotion alone. These results suggest that
volume-changing movements such as expansion and contraction enhance perceived
animacy in robots and should be considered an important design element in
future robot development aimed at shaping human impressions.

### Software Engineering

### 1. [CLARA: A Developer's Companion for Code Comprehension and Analysis](http://arxiv.org/pdf/2509.09072v1)

Authors: Ahmed Adnan, Mushfiqur Rahman, Saad Sakib Noor, Kazi Sakib

Code comprehension and analysis of open-source project codebases is a task
frequently performed by developers and researchers. However, existing tools
that practitioners use for assistance with such tasks often require prior
project setup, lack context-awareness, and involve significant manual effort.
To address this, we present CLARA, a browser extension that utilizes a
state-of-the-art inference model to assist developers and researchers in: (i)
comprehending code files and code fragments, (ii) code refactoring, and (iii)
code quality attribute detection. We qualitatively evaluated CLARA's inference
model using existing datasets and methodology, and performed a comprehensive
user study with 10 developers and academic researchers to assess its usability
and usefulness. The results show that CLARA is useful, accurate, and practical
in code comprehension and analysis tasks. CLARA is an open-source tool
available at https://github.com/SaadNoor555/CLARA_tool_demo. A video showing
the full capabilities of CLARA can be found at
https://youtu.be/VDKVXvIH41Q?si=qBFsmS_Y4m_9x3YH.

### 2. [Altered Histories in Version Control System Repositories: Evidence from the Trenches](http://arxiv.org/pdf/2509.09294v1)

Authors: Solal Rapaport, Laurent Pautet, Samuel Tardieu, Stefano Zacchiroli

Version Control Systems (VCS) like Git allow developers to locally rewrite
recorded history, e.g., to reorder and suppress commits or specific data in
them. These alterations have legitimate use cases, but become problematic when
performed on public branches that have downstream users: they break push/pull
workflows, challenge the integrity and reproducibility of repositories, and
create opportunities for supply chain attackers to sneak into them nefarious
changes. We conduct the first large-scale investigation of Git history
alterations in public code repositories. We analyze 111 M (millions)
repositories archived by Software Heritage, which preserves VCS histories even
across alterations. We find history alterations in 1.22 M repositories, for a
total of 8.7 M rewritten histories. We categorize changes by where they happen
(which repositories, which branches) and what is changed in them (files or
commit metadata). Conducting two targeted case studies we show that altered
histories recurrently change licenses retroactively, or are used to remove
''secrets'' (e.g., private keys) committed by mistake. As these behaviors
correspond to bad practices-in terms of project governance or security
management, respectively-that software recipients might want to avoid, we
introduce GitHistorian, an automated tool, that developers can use to spot and
describe history alterations in public Git repositories.

### 3. [Cross-Domain Evaluation of Transformer-Based Vulnerability Detection on Open & Industry Data](http://arxiv.org/pdf/2509.09313v1)

Authors: Moritz Mock, Thomas Forrer, Barbara Russo

Deep learning solutions for vulnerability detection proposed in academic
research are not always accessible to developers, and their applicability in
industrial settings is rarely addressed. Transferring such technologies from
academia to industry presents challenges related to trustworthiness, legacy
systems, limited digital literacy, and the gap between academic and industrial
expertise. For deep learning in particular, performance and integration into
existing workflows are additional concerns. In this work, we first evaluate the
performance of CodeBERT for detecting vulnerable functions in industrial and
open-source software. We analyse its cross-domain generalisation when
fine-tuned on open-source data and tested on industrial data, and vice versa,
also exploring strategies for handling class imbalance. Based on these results,
we develop AI-DO(Automating vulnerability detection Integration for Developers'
Operations), a Continuous Integration-Continuous Deployment (CI/CD)-integrated
recommender system that uses fine-tuned CodeBERT to detect and localise
vulnerabilities during code review without disrupting workflows. Finally, we
assess the tool's perceived usefulness through a survey with the company's IT
professionals. Our results show that models trained on industrial data detect
vulnerabilities accurately within the same domain but lose performance on
open-source code, while a deep learner fine-tuned on open data, with
appropriate undersampling techniques, improves the detection of
vulnerabilities.

### 4. [Probing Pre-trained Language Models on Code Changes: Insights from ReDef, a High-Confidence Just-in-Time Defect Prediction Dataset](http://arxiv.org/pdf/2509.09192v1)

Authors: Doha Nam, Taehyoun Kim, Duksan Ryu, Jongmoon Baik

Just-in-Time software defect prediction (JIT-SDP) plays a critical role in
prioritizing risky code changes during code review and continuous integration.
However, existing datasets often suffer from noisy labels and low precision in
identifying bug-inducing commits. To address this, we present ReDef
(Revert-based Defect dataset), a high-confidence benchmark of function-level
modifications curated from 22 large-scale C/C++ projects. Defective cases are
anchored by revert commits, while clean cases are validated through post-hoc
history checks. Ambiguous instances are conservatively filtered out via a
GPT-assisted triage process involving multiple votes and audits. This pipeline
yields 3,164 defective and 10,268 clean modifications, offering substantially
more reliable labels than prior existing resources. Beyond dataset
construction, we provide the first systematic evaluation of how pre-trained
language models (PLMs) reason about code modifications -- specifically, which
input encodings most effectively expose change information, and whether models
genuinely capture edit semantics. We fine-tune CodeBERT, CodeT5+, and UniXcoder
under five encoding strategies, and further probe their sensitivity through
counterfactual perturbations that swap added/deleted blocks, invert diff
polarity, or inject spurious markers. Our results show that compact diff-style
encodings consistently outperform whole-function formats across all PLMs, with
statistical tests confirming large, model-independent effects. However, under
counterfactual tests, performance degrades little or not at all -- revealing
that what appears to be robustness in fact reflects reliance on superficial
cues rather than true semantic understanding. These findings indicate that,
unlike in snapshot-based tasks, current PLMs remain limited in their ability to
genuinely comprehend code modifications.

### 5. [On Integrating Large Language Models and Scenario-Based Programming for Improving Software Reliability](http://arxiv.org/pdf/2509.09194v1)

Authors: Ayelet Berzack, Guy Katz

Large Language Models (LLMs) are fast becoming indispensable tools for
software developers, assisting or even partnering with them in crafting complex
programs. The advantages are evident -- LLMs can significantly reduce
development time, generate well-organized and comprehensible code, and
occasionally suggest innovative ideas that developers might not conceive on
their own. However, despite their strengths, LLMs will often introduce
significant errors and present incorrect code with persuasive confidence,
potentially misleading developers into accepting flawed solutions.
  In order to bring LLMs into the software development cycle in a more reliable
manner, we propose a methodology for combining them with ``traditional''
software engineering techniques in a structured way, with the goal of
streamlining the development process, reducing errors, and enabling users to
verify crucial program properties with increased confidence. Specifically, we
focus on the Scenario-Based Programming (SBP) paradigm -- an event-driven,
scenario-based approach for software engineering -- to allow human developers
to pour their expert knowledge into the LLM, as well as to inspect and verify
its outputs.
  To evaluate our methodology, we conducted a significant case study, and used
it to design and implement the Connect4 game. By combining LLMs and SBP we were
able to create a highly-capable agent, which could defeat various strong
existing agents. Further, in some cases, we were able to formally verify the
correctness of our agent. Finally, our experience reveals interesting insights
regarding the ease-of-use of our proposed approach. The full code of our
case-study will be made publicly available with the final version of this
paper.

### 6. [ORCA: Unveiling Obscure Containers In The Wild](http://arxiv.org/pdf/2509.09322v1)

Authors: Jacopo Bufalino, Agathe Blaise, Stefano Secci

Modern software development increasingly depends on open-source libraries and
third-party components, which are often encapsulated into containerized
environments. While improving the development and deployment of applications,
this approach introduces security risks, particularly when outdated or
vulnerable components are inadvertently included in production environments.
Software Composition Analysis (SCA) is a critical process that helps identify
and manage packages and dependencies inside a container. However, unintentional
modifications to the container filesystem can lead to incomplete container
images, which compromise the reliability of SCA tools. In this paper, we
examine the limitations of both cloud-based and open-source SCA tools when
faced with such obscure images. An analysis of 600 popular containers revealed
that obscure containers exist in well-known registries and trusted images and
that many tools fail to analyze such containers. To mitigate these issues, we
propose an obscuration-resilient methodology for container analysis and
introduce ORCA (Obscuration-Resilient Container Analyzer), its open-source
implementation. We reported our findings to all vendors using their appropriate
channels. Our results demonstrate that ORCA effectively detects the content of
obscure containers and achieves a median 40% improvement in file coverage
compared to Docker Scout and Syft.

### 7. [An Integrated Open Source Software System for the Generation and Analysis of Subject-Specific Blood Flow Simulation Ensembles](http://arxiv.org/pdf/2509.09392v1)

Authors: Simon Leistikow, Thomas Miro, Adrian Kummerländer, Ali Nahardani, Katja Grün, Markus Franz, Verena Hoerr, Mathias J. Krause, Lars Linsen

Background and Objective: Hemodynamic analysis of blood flow through arteries
and veins is critical for diagnosing cardiovascular diseases, such as aneurysms
and stenoses, and for investigating cardiovascular parameters, such as
turbulence and wall shear stress. For subject-specific analyses, the anatomy
and blood flow of the subject can be captured non-invasively using structural
and 4D Magnetic Resonance Imaging (MRI). Computational Fluid Dynamics (CFD), on
the other hand, can be used to generate blood flow simulations by solving the
Navier-Stokes equations. To generate and analyze subject-specific blood flow
simulations, MRI and CFD have to be brought together.
  Methods: We present an interactive, customizable, and user-oriented visual
analysis tool that assists researchers in both medicine and numerical analysis.
Our open-source tool is applicable to domains such as CFD and MRI, and it
facilitates the analysis of simulation results and medical data, especially in
hemodynamic studies. It enables the creation of simulation ensembles with a
high variety of parameters. Furthermore, it allows for the visual and
analytical examination of simulations and measurements through 2D embeddings of
the similarity space.
  Results: To demonstrate the effectiveness of our tool, we applied it to three
real-world use cases, showcasing its ability to configure simulation ensembles
and analyse blood flow dynamics. We evaluated our example cases together with
MRI and CFD experts to further enhance features and increase the usability.
  Conclusions: By combining the strengths of both CFD and MRI, our tool
provides a more comprehensive understanding of hemodynamic parameters,
facilitating more accurate analysis of hemodynamic biomarkers.

### 8. [LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering](http://arxiv.org/pdf/2509.09614v1)

Authors: Jielin Qiu, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Jianguo Zhang, Haolin Chen, Shiyu Wang, Ming Zhu, Liangwei Yang, Juntao Tan, Zhepeng Cen, Cheng Qian, Shelby Heinecke, Weiran Yao, Silvio Savarese, Caiming Xiong, Huan Wang

The emergence of long-context language models with context windows extending
to millions of tokens has created new opportunities for sophisticated code
understanding and software development evaluation. We propose LoCoBench, a
comprehensive benchmark specifically designed to evaluate long-context LLMs in
realistic, complex software development scenarios. Unlike existing code
evaluation benchmarks that focus on single-function completion or short-context
tasks, LoCoBench addresses the critical evaluation gap for long-context
capabilities that require understanding entire codebases, reasoning across
multiple files, and maintaining architectural consistency across large-scale
software systems. Our benchmark provides 8,000 evaluation scenarios
systematically generated across 10 programming languages, with context lengths
spanning 10K to 1M tokens, a 100x variation that enables precise assessment of
long-context performance degradation in realistic software development
settings. LoCoBench introduces 8 task categories that capture essential
long-context capabilities: architectural understanding, cross-file refactoring,
multi-session development, bug investigation, feature implementation, code
comprehension, integration testing, and security analysis. Through a 5-phase
pipeline, we create diverse, high-quality scenarios that challenge LLMs to
reason about complex codebases at unprecedented scale. We introduce a
comprehensive evaluation framework with 17 metrics across 4 dimensions,
including 8 new evaluation metrics, combined in a LoCoBench Score (LCBS). Our
evaluation of state-of-the-art long-context models reveals substantial
performance gaps, demonstrating that long-context understanding in complex
software development represents a significant unsolved challenge that demands
more attention. LoCoBench is released at:
https://github.com/SalesforceAIResearch/LoCoBench.

### 9. [I Know Who Clones Your Code: Interpretable Smart Contract Similarity Detection](http://arxiv.org/pdf/2509.09630v1)

Authors: Zhenguang Liu, Lixun Ma, Zhongzheng Mu, Chengkun Wei, Xiaojun Xu, Yingying Jiao, Kui Ren

Widespread reuse of open-source code in smart contract development boosts
programming efficiency but significantly amplifies bug propagation across
contracts, while dedicated methods for detecting similar smart contract
functions remain very limited. Conventional abstract-syntax-tree (AST) based
methods for smart contract similarity detection face challenges in handling
intricate tree structures, which impedes detailed semantic comparison of code.
Recent deep-learning based approaches tend to overlook code syntax and
detection interpretability, resulting in suboptimal performance.
  To fill this research gap, we introduce SmartDetector, a novel approach for
computing similarity between smart contract functions, explainable at the
fine-grained statement level. Technically, SmartDetector decomposes the AST of
a smart contract function into a series of smaller statement trees, each
reflecting a structural element of the source code. Then, SmartDetector uses a
classifier to compute the similarity score of two functions by comparing each
pair of their statement trees. To address the infinite hyperparameter space of
the classifier, we mathematically derive a cosine-wise diffusion process to
efficiently search optimal hyperparameters. Extensive experiments conducted on
three large real-world datasets demonstrate that SmartDetector outperforms
current state-of-the-art methods by an average improvement of 14.01% in
F1-score, achieving an overall average F1-score of 95.88%.

### Social and Information Networks

### 1. [Human-in-the-loop Learning Through Decentralized Communication Mechanisms](http://arxiv.org/pdf/2509.09574v1)

Authors: Yiting Hu, Lingjie Duan

Information sharing platforms like TripAdvisor and Waze involve human agents
as both information producers and consumers. All these platforms operate in a
centralized way to collect agents' latest observations of new options (e.g.,
restaurants, hotels, travel routes) and share such information with all in real
time. However, after hearing the central platforms' live updates, many human
agents are found selfish and unwilling to further explore unknown options for
the benefit of others in the long run. To regulate the human-in-the-loop
learning (HILL) game against selfish agents' free-riding, this paper proposes a
paradigm shift from centralized to decentralized way of operation that forces
agents' local explorations through restricting information sharing. When game
theory meets distributed learning, we formulate our decentralized communication
mechanism's design as a new multi-agent Markov decision process (MA-MDP), and
derive its analytical condition to outperform today's centralized operation. As
the optimal decentralized communication mechanism in MA-MDP is NP-hard to
solve, we present an asymptotically optimal algorithm with linear complexity to
determine the mechanism's timing of intermittent information sharing. Then we
turn to non-myopic agents who may revert to even over-explore, and adapt our
mechanism design to work. Simulation experiments using real-world dataset
demonstrate the effectiveness of our decentralized mechanisms for various
scenarios.

### 2. [Digital Iran Reloaded: Gamer Mitigation Tactics of IRI Information Controls](http://arxiv.org/pdf/2509.09063v1)

Authors: Melinda Cohoon

Internet censorship in the Islamic Republic of Iran restricts access to
global platforms and services, forcing users to rely on circumvention
technologies such as VPNs, proxies, and tunneling tools. This report presents
findings from a mixed-methods study of 660 Iranian internet users, with a focus
on gamers as a digitally literate and socially networked community. Survey data
are combined with network measurements of latency and VPN performance to
identify both technical and social strategies of circumvention. Results show
that while younger users report higher confidence with circumvention, peer
networks, rather than formal training, are the strongest predictors of
resilience. Gaming communities, particularly those active on platforms such as
Discord and Telegram, serve as hubs for sharing tactics and lowering barriers
to adoption. These findings extend existing work on usable security and
censorship circumvention by highlighting the intersection of infrastructural
conditions and social learning. The study concludes with design and policy
implications for developers, researchers, and funders working on digital rights
and information controls.

### 3. [Personality-Enhanced Social Recommendations in SAMI: Exploring the Role of Personality Detection in Matchmaking](http://arxiv.org/pdf/2509.09583v1)

Authors: Brittany Harbison, Samuel Taubman, Travis Taylor, Ashok. K. Goel

Social connection is a vital part of learning, yet online course environments
present barriers to the organic formation of social groups. SAMI offers one
solution by facilitating student connections, but its effectiveness is
constrained by an incomplete Theory of Mind, limiting its ability to create an
effective mental model of a student. One facet of this is its inability to
intuit personality, which may influence the relevance of its recommendations.
To explore this, we propose a personality detection model utilizing GPTs
zero-shot capability to infer Big-Five personality traits from forum
introduction posts, often encouraged in online courses. We benchmark its
performance against established models, demonstrating its efficacy in this
task. Furthermore, we integrate this model into SAMIs entity-based matchmaking
system, enabling personality-informed social recommendations. Initial
integration suggests personality traits can complement existing matching
factors, though additional evaluation is required to determine their full
impact on student engagement and match quality.

### Systems and Control

### 1. [Optimal Control of an SIR Model with Noncompliance as a Social Contagion](http://arxiv.org/pdf/2509.09075v1)

Authors: Chloe Ngo, Christian Parkinson, Weinan Wang

We propose and study a compartmental model for epidemiology with human
behavioral effects. Specifically, our model incorporates governmental
prevention measures aimed at lowering the disease infection rate, but we split
the population into those who comply with the measures and those who do not
comply and therefore do not receive the reduction in infectivity. We then allow
the attitude of noncompliance to spread as a social contagion parallel to the
disease. We derive the reproductive ratio for our model and provide stability
analysis for the disease-free equilibria. We then propose a control scenario
wherein a policy-maker with access to control variables representing disease
prevention mandates, treatment efforts, and educational campaigns aimed at
encouraging compliance minimizes a cost functional incorporating several cost
concerns. We characterize optimal controls via the Pontryagin optimality
principle and present simulations which demonstrate the behavior of the control
maps in several different parameter regimes.

### 2. [KAN-Therm: A Lightweight Battery Thermal Model Using Kolmogorov-Arnold Network](http://arxiv.org/pdf/2509.09145v1)

Authors: Soumyoraj Mallick, Sanchita Ghosh, Tanushree Roy

Battery management systems (BMSs) rely on real-time estimation of battery
temperature distribution in battery cells to ensure safe and optimal operation
of Lithium-ion batteries (LIBs). However, physical BMS often suffers from
memory and computational resource limitations required by highfidelity models.
Temperature prediction using physics-based models becomes challenging due to
their higher computational time. In contrast, machine learning based approaches
offer faster predictions but demand larger memory overhead. In this work, we
develop a lightweight and efficient Kolmogorov-Arnold networks (KAN) based
thermal model, KAN-Therm, to predict the core temperature of a cylindrical
battery. We have compared the memory overhead and computation costs of our
method with Multi-layer perceptron (MLP), recurrent neural network (RNN), and
long shortterm memory (LSTM) network. Our results show that the proposed
KAN-Therm model exhibit the best prediction accuracy with the least memory
overhead and computation time.

### 3. [Voltage Synchronization and Proportional Current Sharing of Grid-Forming Inverters](http://arxiv.org/pdf/2509.09277v1)

Authors: Qianxi Tang, Li Peng

Most previously proposed controllers are analyzed in the
small-signal/quasi-steady regime rather than large-signal or transient
stability for grid-forming inverters (GFMI). Additionally, methods that presume
system-wide data--global measurements and complete grid-model knowledge--are
challenging to realize in practice and unsuitable for large-scale operation.
Moreover, proportional current sharing is rarely embedded into them. The whole
system is a high-order, nonlinear differential system, making analysis
intractable without principled simplifications. Hence, contraction stability
analysis in GFMI is proposed to guarantee the large-signal stability.
Furthermore, a contraction-based controller is proposed to synchronize GFMI.
Additionally, this paper proposes integrating an auxiliary virtual-impedance
layer into the contraction-based controller to achieve proportional current
sharing, while the GFMI retains global stability and voltage synchronization. A
dispatchable virtual oscillator control (dVOC), also known as the
Andronov--Hopf oscillator (AHO) is used to validate the proposed contraction
stability analysis and contraction-based controller with virtual-impedance. It
is proved that the complex multi-converter system can achieve output-feedback
contraction under large-signal operation. Therefore, without requiring
system-wide data, the proposed method offers voltage synchronization,
decentralized stability conditions for the transient stability of AHO and
proportional current sharing, beyond prior small-signal, quasi-steady analysis.

### 4. [Towards Efficient and Secure Cloud Control Systems: Advances, Challenges, and Future Directions](http://arxiv.org/pdf/2509.09299v1)

Authors: Yasir Ali, Tayyab Manzoor, Huan Yang, Asif Ali, Yuanqing Xia

Networked Control Systems (NCSs) have been instrumental in realizing fully
connected and responsive intelligent environments within the context of
real-time virtual control and management. However, traditional NCSs face
considerable challenges in handling the vast amounts of data generated by
large-scale control applications, particularly in terms of data acquisition,
storage, and computational processing. To address these challenges, the
emergence of cloud computing and advancements in control theory have empowered
the new paradigm known as Cloud Control Systems (CCSs). Recently, CCSs have
received substantial attention from industries for their potential properties,
such as large-scale data management, complex computations, and data-centric
optimized decisions. This study presents an extensive review of recent progress
in CCSs spanning over multiple studies published between 2012 and 2025.
Specifically, the focus is on providing a taxonomy of the current findings in
CCS research, encompassing various perspectives, such as its efficient
implementations in industrial automation, security and privacy considerations,
and cloud-based control techniques. Each category is examined in depth through
selected state-of-the-art analyses of different approaches and contrasting
methodologies. Furthermore, we discuss future directions aimed at designing
more efficient and practical CCSs. The insights gained from this study can help
researchers, practitioners, and decision-makers in their domain for effective
CCS design and deployment.

### 5. [A Comparative Analysis of Robust and Reliable Designs Using the Compromised Design Support Problem: A Case Study in Hot Rod Rolling Processes](http://arxiv.org/pdf/2509.09422v1)

Authors: Maryam Ghasemzadeh, H M Dilshad Alam Digonta, Anand Balu Nellippallil, Anton van Beek

Design under uncertainty is a challenging problem, as a systems performance
can be highly sensitive to variations in input parameters and model
uncertainty. A conventional approach to addressing such problems is robust
optimization, which seeks to enhance design performance by reducing sensitivity
to uncertainty. Alternatively, reliability-based design focuses on optimizing
performance while ensuring that failure constraints are satisfied with a
specified probability. While both methods are well established, their
integration into multi-objective and multi-stakeholder decision-making
frameworks remains a challenging problem. In this study, we extend the
Compromise Decision Support Problem (cDSP) framework to incorporate
reliability-based design considerations and evaluate its performance in
comparison to the conventional robust-based cDSP formulation. The developed
framework has been validated on a multidisciplinary hot rod rolling process
including parametric and model uncertainties. The results compare the predicted
performance under robust and reliable scenarios, validating the efficiency of
the approach in managing uncertainties for complex, multidisciplinary systems.
Specifically, we found that the two methods exhibit markedly different
performance when the predicted performance follows a non-normal distribution, a
situation that arises in non-linear systems with parametric uncertainty. Based
on this insight, we offer guidance to designers on the conditions under which
each method is most appropriate.

### 6. [Taming Spontaneous Stop-and-Go Traffic Waves: A Computational Mechanism Design Perspective](http://arxiv.org/pdf/2509.09441v1)

Authors: Di Shen, Qi Dai, Suzhou Huang, Dimitar Filev

It is well known that stop-and-go waves can be generated spontaneously in
traffic even without bottlenecks. Can such undesirable traffic patterns,
induced by intrinsic human driving behaviors, be tamed effectively and
inexpensively? Taking advantage of emerging connectivity and autonomy
technologies, we envision a simple yet realistic traffic control system to
achieve this goal. To prove the concept, we design such a system to suppress
these waves while maximizing traffic throughput in the Tadaki setting: a
circular road with varying number of vehicles. We first introduce our driver
behavior model and demonstrate how our calibrated human driving agents can
closely reproduce the observed human driving patterns in the original Tadaki
experiment. We then propose a simple control system mediated via connected
automated vehicles (CAV) whose ideal speed parameter is treated as a
system-level control variable adapted to the local vehicle density of the
traffic. The objective of the control system is set up as a tradeoff:
maximizing throughput while minimizing traffic oscillation. Following
computational mechanism design, we search for the optimal control policy as a
function of vehicle density and the tradeoff attitude parameter. This can be
done by letting all vehicles play a simulated game of CAV-modulated traffic
under such a control system. Our simulation results show that the improvements
in traffic efficiency and smoothness are substantial. Finally, we envision how
such a traffic control system can be realized in an environment with smart
vehicles connected to a smart infrastructure or via a scheme of variable speed
advisory.

### 7. [Taming Spontaneous Stop-and-Go Traffic Waves: A Bifurcation Perspective of A Dynamical Map](http://arxiv.org/pdf/2509.09466v1)

Authors: Suzhou Huang, Jian Hu

We consider a discrete-time dynamical system in a car-following context. The
system was recently introduced to parsimoniously model human driving behavior
based on utility maximization. The parameters of the model were calibrated
using vehicle trajectory data from the Sugiyama experiment. It was shown that
such a system can accurately reproduce the observed collective phenomena of a
more elaborate experiment by Tadaki et al. Once the heterogeneity and noise are
switched off, the model defines a map of the corresponding discrete-time
dynamical system. We first perform a bifurcation analysis of the map by
studying the stability of its limit solutions: a free-flow fixed point and a
stop-and-go quasi-periodic orbit. When the vehicle density is varied, our model
displays a bifurcation diagram qualitatively similar to those found in a class
of optimal velocity models based on an ordinary differential equation approach,
including regimes where one or both of the limit solutions are stable. In a 2D
bifurcation diagram we further demonstrate that imposing a vehicle
density-dependent speed advisory can dissipate the stop-and-go quasi-periodic
orbit. This in turn lays the mathematical foundation for a simple, yet
effective proposal [1] to tame stop-and-go waves, improving traffic flow and
smoothness simultaneously via variable speed advisory.

### 8. [Learning-Based Data-Assisted Port-Hamiltonian Control for Free-Floating Space Manipulators](http://arxiv.org/pdf/2509.09563v1)

Authors: Mostafa Eslami, Maryam Babazadeh

A generic data-assisted control architecture within the port-Hamiltonian
framework is proposed, introducing a physically meaningful observable that
links conservative dynamics to all actuation, dissipation, and disturbance
channels. A robust, model-based controller combined with a high-gain
decentralized integrator establishes large robustness margins and strict
time-scale separation, ensuring that subsequent learning cannot destabilize the
primary dynamics. Learning, selected for its generalizability, is then applied
to capture complex, unmodeled effects, despite inherent delay and transient
error during adaptation. Formal Lyapunov analysis with explicit stability
bounds guarantees convergence under bounded learning errors. The structured
design confines learning to the simplest part of the dynamics, enhancing data
efficiency while preserving physical interpretability. The approach is generic,
with a free-floating space manipulator orientation control task, including
integrated null-space collision avoidance, serving as a case study to
demonstrate robust tracking performance and applicability to broader robotic
domains.

### 9. [A neural drift-plus-penalty algorithm for network power allocation and routing](http://arxiv.org/pdf/2509.09637v1)

Authors: Ahmed Rashwan, Keith Briggs, Chris Budd

The drift-plus-penalty method is a Lyapunov optimisation technique commonly
applied to network routing problems. It reduces the original stochastic
planning task to a sequence of greedy optimizations, enabling the design of
distributed routing algorithms which stabilize data queues while simultaneously
optimizing a specified penalty function. While drift-plus-penalty methods have
desirable asymptotic properties, they tend to incur higher network delay than
alternative control methods, especially under light network load. In this work,
we propose a learned variant of the drift-plus-penalty method that can preserve
its theoretical guarantees, while being flexible enough to learn routing
strategies directly from a model of the problem. Our approach introduces a
novel mechanism for learning routing decisions and employs an optimal
transport-based method for link scheduling. Applied to the joint task of
transmit-power allocation and data routing, the method achieves consistent
improvements over common baselines under a broad set of scenarios.

### 10. [Implementation of a 8-bit Wallace Tree Multiplier](http://arxiv.org/pdf/2509.09178v1)

Authors: Ayan Biswas, Jimmy Jin

Wallace tree multipliers are a parallel digital multiplier architecture
designed to minimize the worst-case time complexity of the circuit depth
relative to the input size [1]. In particular, it seeks to perform long
multiplication in the binary sense, reducing as many partial products per stage
as possible through full and half adders circuits, achieving O(log(n)) where n
= bit length of input. This paper provides an overview of the design, progress
and methodology in the final project of ECE 55900, consisting of the schematic
and layout of a Wallace tree 8-bit input multiplier on the gpdk45 technology in
Cadence Virtuoso, as well as any design attempts prior to the final product.
This also includes our endeavors in designing the final MAC (Multiply
Accumulate) unit with undefined targets, which we chose to implement as a 16
bit combinational multiply-add.

### Machine Learning (Statistics Category)

### 1. [Low-degree lower bounds via almost orthonormal bases](http://arxiv.org/pdf/2509.09353v1)

Authors: Alexandra Carpentier, Simone Maria Giancola, Christophe Giraud, Nicolas Verzelen

Low-degree polynomials have emerged as a powerful paradigm for providing
evidence of statistical-computational gaps across a variety of high-dimensional
statistical models [Wein25]. For detection problems -- where the goal is to
test a planted distribution $\mathbb{P}'$ against a null distribution
$\mathbb{P}$ with independent components -- the standard approach is to bound
the advantage using an $\mathbb{L}^2(\mathbb{P})$-orthonormal family of
polynomials. However, this method breaks down for estimation tasks or more
complex testing problems where $\mathbb{P}$ has some planted structures, so
that no simple $\mathbb{L}^2(\mathbb{P})$-orthogonal polynomial family is
available. To address this challenge, several technical workarounds have been
proposed [SW22,SW25], though their implementation can be delicate. In this
work, we propose a more direct proof strategy. Focusing on random graph models,
we construct a basis of polynomials that is almost orthonormal under
$\mathbb{P}$, in precisely those regimes where statistical-computational gaps
arise. This almost orthonormal basis not only yields a direct route to
establishing low-degree lower bounds, but also allows us to explicitly identify
the polynomials that optimize the low-degree criterion. This, in turn, provides
insights into the design of optimal polynomial-time algorithms. We illustrate
the effectiveness of our approach by recovering known low-degree lower bounds,
and establishing new ones for problems such as hidden subcliques, stochastic
block models, and seriation models.

### 2. [STRIDE: Scalable and Interpretable XAI via Subset-Free Functional Decomposition](http://arxiv.org/pdf/2509.09070v1)

Authors: Chaeyun Ko

Most explainable AI (XAI) frameworks face two practical limitations: the
exponential cost of reasoning over feature subsets and the reduced
expressiveness of summarizing effects as single scalar values. We present
STRIDE, a scalable framework that aims to mitigate both issues by framing
explanation as a subset-enumeration-free, orthogonal functional decomposition
in a Reproducing Kernel Hilbert Space (RKHS). Rather than focusing only on
scalar attributions, STRIDE computes functional components f_S(x_S) via an
analytical projection scheme based on a recursive kernel-centering procedure,
avoiding explicit subset enumeration. In the tabular setups we study, the
approach is model-agnostic, provides both local and global views, and is
supported by theoretical results on orthogonality and L^2 convergence under
stated assumptions. On public tabular benchmarks in our environment, we
observed speedups ranging from 0.6 times (slower than TreeSHAP on a small
dataset) to 9.7 times (California), with a median approximate 3.0 times across
10 datasets, while maintaining high fidelity (R^2 between 0.81 and 0.999) and
substantial rank agreement on most datasets. Overall, STRIDE complements scalar
attribution methods by offering a structured functional perspective, enabling
novel diagnostics like 'component surgery' to quantitatively measure the impact
of specific interactions within our experimental scope.

### 3. [Scalable extensions to given-data Sobol' index estimators](http://arxiv.org/pdf/2509.09078v1)

Authors: Teresa Portone, Bert Debusschere, Samantha Yang, Emiliano Islas-Quinones, T. Patrick Xiao

Given-data methods for variance-based sensitivity analysis have significantly
advanced the feasibility of Sobol' index computation for computationally
expensive models and models with many inputs. However, the limitations of
existing methods still preclude their application to models with an extremely
large number of inputs. In this work, we present practical extensions to the
existing given-data Sobol' index method, which allow variance-based sensitivity
analysis to be efficiently performed on large models such as neural networks,
which have $>10^4$ parameterizable inputs. For models of this size, holding all
input-output evaluations simultaneously in memory -- as required by existing
methods -- can quickly become impractical. These extensions also support
nonstandard input distributions with many repeated values, which are not
amenable to equiprobable partitions employed by existing given-data methods.
  Our extensions include a general definition of the given-data Sobol' index
estimator with arbitrary partition, a streaming algorithm to process
input-output samples in batches, and a heuristic to filter out small indices
that are indistinguishable from zero indices due to statistical noise. We show
that the equiprobable partition employed in existing given-data methods can
introduce significant bias into Sobol' index estimates even at large sample
sizes and provide numerical analyses that demonstrate why this can occur. We
also show that our streaming algorithm can achieve comparable accuracy and
runtimes with lower memory requirements, relative to current methods which
process all samples at once. We demonstrate our novel developments on two
application problems in neural network modeling.

### 4. [Global Optimization of Stochastic Black-Box Functions with Arbitrary Noise Distributions using Wilson Score Kernel Density Estimation](http://arxiv.org/pdf/2509.09238v1)

Authors: Thorbjørn Mosekjær Iversen, Lars Carøe Sørensen, Simon Faarvang Mathiesen, Henrik Gordon Petersen

Many optimization problems in robotics involve the optimization of
time-expensive black-box functions, such as those involving complex simulations
or evaluation of real-world experiments. Furthermore, these functions are often
stochastic as repeated experiments are subject to unmeasurable disturbances.
Bayesian optimization can be used to optimize such methods in an efficient
manner by deploying a probabilistic function estimator to estimate with a given
confidence so that regions of the search space can be pruned away.
Consequently, the success of the Bayesian optimization depends on the function
estimator's ability to provide informative confidence bounds. Existing function
estimators require many function evaluations to infer the underlying confidence
or depend on modeling of the disturbances. In this paper, it is shown that the
confidence bounds provided by the Wilson Score Kernel Density Estimator
(WS-KDE) are applicable as excellent bounds to any stochastic function with an
output confined to the closed interval [0;1] regardless of the distribution of
the output. This finding opens up the use of WS-KDE for stable global
optimization on a wider range of cost functions. The properties of WS-KDE in
the context of Bayesian optimization are demonstrated in simulation and applied
to the problem of automated trap design for vibrational part feeders.

### 5. [Expressive Power of Deep Networks on Manifolds: Simultaneous Approximation](http://arxiv.org/pdf/2509.09362v1)

Authors: Hanfei Zhou, Lei Shi

A key challenge in scientific machine learning is solving partial
differential equations (PDEs) on complex domains, where the curved geometry
complicates the approximation of functions and their derivatives required by
differential operators. This paper establishes the first simultaneous
approximation theory for deep neural networks on manifolds. We prove that a
constant-depth $\mathrm{ReLU}^{k-1}$ network with bounded weights--a property
that plays a crucial role in controlling generalization error--can approximate
any function in the Sobolev space $\mathcal{W}_p^{k}(\mathcal{M}^d)$ to an
error of $\varepsilon$ in the $\mathcal{W}_p^{s}(\mathcal{M}^d)$ norm, for
$k\geq 3$ and $s<k$, using $\mathcal{O}(\varepsilon^{-d/(k-s)})$ nonzero
parameters, a rate that overcomes the curse of dimensionality by depending only
on the intrinsic dimension $d$. These results readily extend to functions in
H\"older-Zygmund spaces. We complement this result with a matching lower bound,
proving our construction is nearly optimal by showing the required number of
parameters matches up to a logarithmic factor. Our proof of the lower bound
introduces novel estimates for the Vapnik-Chervonenkis dimension and
pseudo-dimension of the network's high-order derivative classes. These
complexity bounds provide a theoretical cornerstone for learning PDEs on
manifolds involving derivatives. Our analysis reveals that the network
architecture leverages a sparse structure to efficiently exploit the manifold's
low-dimensional geometry.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-09-12 PST.

### 1. [Robot movement planning for obstacle avoidance using reinforcement learning](https://www.nature.com/articles/s41598-025-17740-5)

Authors: Linda-Sophie Schneider et al.

### 2. [The effects of biophilic design on steering performance in virtual reality](https://www.nature.com/articles/s41598-025-19113-4)

Authors: Fariba Mostajeran et al.

### 3. [Bi-objective jellyfish algorithm for team formation problem](https://www.nature.com/articles/s41598-025-11566-x)

Authors: Mustafa Abdul Salam et al.

