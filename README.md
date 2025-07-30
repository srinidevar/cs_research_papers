# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-29 17:00:25.973190 PST.

### Artificial Intelligence

### 1. [Enhancing QoS in Edge Computing through Federated Layering Techniques: A Pathway to Resilient AI Lifelong Learning Systems](http://arxiv.org/pdf/2507.20444v1)

Authors: Chengzhuo Han

In the context of the rapidly evolving information technology landscape,
marked by the advent of 6G communication networks, we face an increased data
volume and complexity in network environments. This paper addresses these
challenges by focusing on Quality of Service (QoS) in edge computing
frameworks. We propose a novel approach to enhance QoS through the development
of General Artificial Intelligence Lifelong Learning Systems, with a special
emphasis on Federated Layering Techniques (FLT). Our work introduces a
federated layering-based small model collaborative mechanism aimed at improving
AI models' operational efficiency and response time in environments where
resources are limited. This innovative method leverages the strengths of cloud
and edge computing, incorporating a negotiation and debate mechanism among
small AI models to enhance reasoning and decision-making processes. By
integrating model layering techniques with privacy protection measures, our
approach ensures the secure transmission of model parameters while maintaining
high efficiency in learning and reasoning capabilities. The experimental
results demonstrate that our strategy not only enhances learning efficiency and
reasoning accuracy but also effectively protects the privacy of edge nodes.
This presents a viable solution for achieving resilient large model lifelong
learning systems, with a significant improvement in QoS for edge computing
environments.

### 2. [STARN-GAT: A Multi-Modal Spatio-Temporal Graph Attention Network for Accident Severity Prediction](http://arxiv.org/pdf/2507.20451v1)

Authors: Pritom Ray Nobin, Imran Ahammad Rifat

Accurate prediction of traffic accident severity is critical for improving
road safety, optimizing emergency response strategies, and informing the design
of safer transportation infrastructure. However, existing approaches often
struggle to effectively model the intricate interdependencies among spatial,
temporal, and contextual variables that govern accident outcomes. In this
study, we introduce STARN-GAT, a Multi-Modal Spatio-Temporal Graph Attention
Network, which leverages adaptive graph construction and modality-aware
attention mechanisms to capture these complex relationships. Unlike
conventional methods, STARN-GAT integrates road network topology, temporal
traffic patterns, and environmental context within a unified attention-based
framework. The model is evaluated on the Fatality Analysis Reporting System
(FARS) dataset, achieving a Macro F1-score of 85 percent, ROC-AUC of 0.91, and
recall of 81 percent for severe incidents. To ensure generalizability within
the South Asian context, STARN-GAT is further validated on the ARI-BUET traffic
accident dataset, where it attains a Macro F1-score of 0.84, recall of 0.78,
and ROC-AUC of 0.89. These results demonstrate the model's effectiveness in
identifying high-risk cases and its potential for deployment in real-time,
safety-critical traffic management systems. Furthermore, the attention-based
architecture enhances interpretability, offering insights into contributing
factors and supporting trust in AI-assisted decision-making. Overall, STARN-GAT
bridges the gap between advanced graph neural network techniques and practical
applications in road safety analytics.

### 3. [MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design](http://arxiv.org/pdf/2507.20541v1)

Authors: Zishang Qiu, Xinan Chen, Long Chen, Ruibin Bai

This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that
presents a new paradigm for Automatic Heuristic Design (AHD). Traditional
evolutionary methods operate directly on heuristic code; in contrast, MeLA
evolves the instructional prompts used to guide a Large Language Model (LLM) in
generating these heuristics. This process of "prompt evolution" is driven by a
novel metacognitive framework where the system analyzes performance feedback to
systematically refine its generative strategy. MeLA's architecture integrates a
problem analyzer to construct an initial strategic prompt, an error diagnosis
system to repair faulty code, and a metacognitive search engine that
iteratively optimizes the prompt based on heuristic effectiveness. In
comprehensive experiments across both benchmark and real-world problems, MeLA
consistently generates more effective and robust heuristics, significantly
outperforming state-of-the-art methods. Ultimately, this research demonstrates
the profound potential of using cognitive science as a blueprint for AI
architecture, revealing that by enabling an LLM to metacognitively regulate its
problem-solving process, we unlock a more robust and interpretable path to AHD.

### 4. [Unlearning of Knowledge Graph Embedding via Preference Optimization](http://arxiv.org/pdf/2507.20566v1)

Authors: Jiajun Liu, Wenjun Ke, Peng Wang, Yao He, Ziyu Shang, Guozheng Li, Zijie Xu, Ke Ji

Existing knowledge graphs (KGs) inevitably contain outdated or erroneous
knowledge that needs to be removed from knowledge graph embedding (KGE) models.
To address this challenge, knowledge unlearning can be applied to eliminate
specific information while preserving the integrity of the remaining knowledge
in KGs. Existing unlearning methods can generally be categorized into exact
unlearning and approximate unlearning. However, exact unlearning requires high
training costs while approximate unlearning faces two issues when applied to
KGs due to the inherent connectivity of triples: (1) It fails to fully remove
targeted information, as forgetting triples can still be inferred from
remaining ones. (2) It focuses on local data for specific removal, which
weakens the remaining knowledge in the forgetting boundary. To address these
issues, we propose GraphDPO, a novel approximate unlearning framework based on
direct preference optimization (DPO). Firstly, to effectively remove forgetting
triples, we reframe unlearning as a preference optimization problem, where the
model is trained by DPO to prefer reconstructed alternatives over the original
forgetting triples. This formulation penalizes reliance on forgettable
knowledge, mitigating incomplete forgetting caused by KG connectivity.
Moreover, we introduce an out-boundary sampling strategy to construct
preference pairs with minimal semantic overlap, weakening the connection
between forgetting and retained knowledge. Secondly, to preserve boundary
knowledge, we introduce a boundary recall mechanism that replays and distills
relevant information both within and across time steps. We construct eight
unlearning datasets across four popular KGs with varying unlearning rates.
Experiments show that GraphDPO outperforms state-of-the-art baselines by up to
10.1% in MRR_Avg and 14.0% in MRR_F1.

### 5. [A General Framework for Dynamic MAPF using Multi-Shot ASP and Tunnels](http://arxiv.org/pdf/2507.20703v1)

Authors: Aysu Bogatarkan, Esra Erdem

MAPF problem aims to find plans for multiple agents in an environment within
a given time, such that the agents do not collide with each other or obstacles.
Motivated by the execution and monitoring of these plans, we study Dynamic MAPF
(D-MAPF) problem, which allows changes such as agents entering/leaving the
environment or obstacles being removed/moved. Considering the requirements of
real-world applications in warehouses with the presence of humans, we introduce
1) a general definition for D-MAPF (applicable to variations of D-MAPF), 2) a
new framework to solve D-MAPF (utilizing multi-shot computation, and allowing
different methods to solve D-MAPF), and 3) a new ASP-based method to solve
D-MAPF (combining advantages of replanning and repairing methods, with a novel
concept of tunnels to specify where agents can move). We have illustrated the
strengths and weaknesses of this method by experimental evaluations, from the
perspectives of computational performance and quality of solutions.

### 6. [Algorithmic Fairness: A Runtime Perspective](http://arxiv.org/pdf/2507.20711v1)

Authors: Filip Cano, Thomas A. Henzinger, Konstantin Kueffner

Fairness in AI is traditionally studied as a static property evaluated once,
over a fixed dataset. However, real-world AI systems operate sequentially, with
outcomes and environments evolving over time. This paper proposes a framework
for analysing fairness as a runtime property. Using a minimal yet expressive
model based on sequences of coin tosses with possibly evolving biases, we study
the problems of monitoring and enforcing fairness expressed in either toss
outcomes or coin biases. Since there is no one-size-fits-all solution for
either problem, we provide a summary of monitoring and enforcement strategies,
parametrised by environment dynamics, prediction horizon, and confidence
thresholds. For both problems, we present general results under simple or
minimal assumptions. We survey existing solutions for the monitoring problem
for Markovian and additive dynamics, and existing solutions for the enforcement
problem in static settings with known dynamics.

### 7. [Beyond Listenership: AI-Predicted Interventions Drive Improvements in Maternal Health Behaviours](http://arxiv.org/pdf/2507.20755v1)

Authors: Arpan Dasgupta, Sarvesh Gharat, Neha Madhiwalla, Aparna Hegde, Milind Tambe, Aparna Taneja

Automated voice calls with health information are a proven method for
disseminating maternal and child health information among beneficiaries and are
deployed in several programs around the world. However, these programs often
suffer from beneficiary dropoffs and poor engagement. In previous work, through
real-world trials, we showed that an AI model, specifically a restless bandit
model, could identify beneficiaries who would benefit most from live service
call interventions, preventing dropoffs and boosting engagement. However, one
key question has remained open so far: does such improved listenership via
AI-targeted interventions translate into beneficiaries' improved knowledge and
health behaviors? We present a first study that shows not only listenership
improvements due to AI interventions, but also simultaneously links these
improvements to health behavior changes. Specifically, we demonstrate that
AI-scheduled interventions, which enhance listenership, lead to statistically
significant improvements in beneficiaries' health behaviors such as taking iron
or calcium supplements in the postnatal period, as well as understanding of
critical health topics during pregnancy and infancy. This underscores the
potential of AI to drive meaningful improvements in maternal and child health.

### 8. [How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation](http://arxiv.org/pdf/2507.20758v1)

Authors: Hao Yang, Qinghua Zhao, Lei Li

Chain-of-Thought (CoT) prompting significantly enhances model reasoning, yet
its internal mechanisms remain poorly understood. We analyze CoT's operational
principles by reversely tracing information flow across decoding, projection,
and activation phases. Our quantitative analysis suggests that CoT may serve as
a decoding space pruner, leveraging answer templates to guide output
generation, with higher template adherence strongly correlating with improved
performance. Furthermore, we surprisingly find that CoT modulates neuron
engagement in a task-dependent manner: reducing neuron activation in
open-domain tasks, yet increasing it in closed-domain scenarios. These findings
offer a novel mechanistic interpretability framework and critical insights for
enabling targeted CoT interventions to design more efficient and robust
prompts. We released our code and data at
https://anonymous.4open.science/r/cot-D247.

### 9. [evalSmarT: An LLM-Based Framework for Evaluating Smart Contract Generated Comments](http://arxiv.org/pdf/2507.20774v1)

Authors: Fatou Ndiaye Mbodji

Smart contract comment generation has gained traction as a means to improve
code comprehension and maintainability in blockchain systems. However,
evaluating the quality of generated comments remains a challenge. Traditional
metrics such as BLEU and ROUGE fail to capture domain-specific nuances, while
human evaluation is costly and unscalable. In this paper, we present
\texttt{evalSmarT}, a modular and extensible framework that leverages large
language models (LLMs) as evaluators. The system supports over 400 evaluator
configurations by combining approximately 40 LLMs with 10 prompting strategies.
We demonstrate its application in benchmarking comment generation tools and
selecting the most informative outputs. Our results show that prompt design
significantly impacts alignment with human judgment, and that LLM-based
evaluation offers a scalable and semantically rich alternative to existing
methods.

### 10. [MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs](http://arxiv.org/pdf/2507.20804v1)

Authors: Xueyao Wan, Hang Yu

Retrieval-Augmented Generation (RAG) enhances language model generation by
retrieving relevant information from external knowledge bases. However,
conventional RAG methods face the issue of missing multimodal information.
Multimodal RAG methods address this by fusing images and text through mapping
them into a shared embedding space, but they fail to capture the structure of
knowledge and logical chains between modalities. Moreover, they also require
large-scale training for specific tasks, resulting in limited generalizing
ability. To address these limitations, we propose MMGraphRAG, which refines
visual content through scene graphs and constructs a multimodal knowledge graph
(MMKG) in conjunction with text-based KG. It employs spectral clustering to
achieve cross-modal entity linking and retrieves context along reasoning paths
to guide the generative process. Experimental results show that MMGraphRAG
achieves state-of-the-art performance on the DocBench and MMLongBench datasets,
demonstrating strong domain adaptability and clear reasoning paths.

### Computational Complexity

### 1. [Core Safety Values for Provably Corrigible Agents](http://arxiv.org/pdf/2507.20964v1)

Authors: Aran Nayebi

We introduce the first implementable framework for corrigibility, with
provable guarantees in multi-step, partially observed environments. Our
framework replaces a single opaque reward with five *structurally separate*
utility heads -- deference, switch-access preservation, truthfulness,
low-impact behavior via a belief-based extension of Attainable Utility
Preservation, and bounded task reward -- combined lexicographically by strict
weight gaps. Theorem 1 proves exact single-round corrigibility in the partially
observable off-switch game; Theorem 3 extends the guarantee to multi-step,
self-spawning agents, showing that even if each head is \emph{learned} to
mean-squared error $\varepsilon$ and the planner is $\varepsilon$-sub-optimal,
the probability of violating \emph{any} safety property is bounded while still
ensuring net human benefit. In contrast to Constitutional AI or RLHF/RLAIF,
which merge all norms into one learned scalar, our separation makes obedience
and impact-limits dominate even when incentives conflict. For open-ended
settings where adversaries can modify the agent, we prove that deciding whether
an arbitrary post-hack agent will ever violate corrigibility is undecidable by
reduction to the halting problem, then carve out a finite-horizon ``decidable
island'' where safety can be certified in randomized polynomial time and
verified with privacy-preserving, constant-round zero-knowledge proofs.
Consequently, the remaining challenge is the ordinary ML task of data coverage
and generalization: reward-hacking risk is pushed into evaluation quality
rather than hidden incentive leak-through, giving clearer implementation
guidance for today's LLM assistants and future autonomous systems.

### Computational Engineering

### 1. [Learning Explainable Stock Predictions with Tweets Using Mixture of Experts](http://arxiv.org/pdf/2507.20535v1)

Authors: Wenyan Xu, Dawei Xiang, Rundong Wang, Yonghong Hu, Liang Zhang, Jiayu Chen, Zhonghua Lu

Stock price movements are influenced by many factors, and alongside
historical price data, tex-tual information is a key source. Public news and
social media offer valuable insights into market sentiment and emerging events.
These sources are fast-paced, diverse, and significantly impact future stock
trends. Recently, LLMs have enhanced financial analysis, but prompt-based
methods still have limitations, such as input length restrictions and
difficulties in predicting sequences of varying lengths. Additionally, most
models rely on dense computational layers, which are resource-intensive. To
address these challenges, we propose the FTS- Text-MoE model, which combines
numerical data with key summaries from news and tweets using point embeddings,
boosting prediction accuracy through the integration of factual textual data.
The model uses a Mixture of Experts (MoE) Transformer decoder to process both
data types. By activating only a subset of model parameters, it reduces
computational costs. Furthermore, the model features multi-resolution
prediction heads, enabling flexible forecasting of financial time series at
different scales. Experimental results show that FTS-Text-MoE outperforms
baseline methods in terms of investment returns and Sharpe ratio, demonstrating
its superior accuracy and ability to predict future market trends.

### 2. [Exascale Implicit Kinetic Plasma Simulations on El~Capitan for Solving the Micro-Macro Coupling in Magnetospheric Physics](http://arxiv.org/pdf/2507.20719v1)

Authors: Stefano Markidis, Andong Hu, Ivy Peng, Luca Pennati, Ian Lumsden, Dewi Yokelson, Stephanie Brink, Olga Pearce, Thomas R. W. Scogland, Bronis R. de Supinski, Gian Luca Delzanno, Michela Taufer

Our fully kinetic, implicit Particle-in-Cell (PIC) simulations of global
magnetospheres on up to 32,768 of El Capitan's AMD Instinct MI300A Accelerated
Processing Units (APUs) represent an unprecedented computational capability
that addresses a fundamental challenge in space physics: resolving the
multi-scale coupling between microscopic (electron-scale) and macroscopic
(global-scale) dynamics in planetary magnetospheres. The implicit scheme of
iPIC3D supports time steps and grid spacing that are up to 10 times larger than
those of explicit methods, without sacrificing physical accuracy. This enables
the simulation of magnetospheres while preserving fine-scale electron physics,
which is critical for key processes such as magnetic reconnection and plasma
turbulence. Our algorithmic and technological innovations include GPU-optimized
kernels, particle control, and physics-aware data compression using Gaussian
Mixture Models. With simulation domains spanning 100-1,000 ion skin depths, we
reach the global scale of small-to-medium planetary magnetospheres, such as
those of Mercury and Ganymede, which supports fully kinetic treatment of
global-scale dynamics in systems previously out of reach for fully kinetic PIC
codes.

### 3. [PySHRED: A Python package for SHallow REcurrent Decoding for sparse sensing, model reduction and scientific discovery](http://arxiv.org/pdf/2507.20954v1)

Authors: David Ye, Jan Williams, Mars Gao, Stefano Riva, Matteo Tomasetto, David Zoro, J. Nathan Kutz

SHallow REcurrent Decoders (SHRED) provide a deep learning strategy for
modeling high-dimensional dynamical systems and/or spatiotemporal data from
dynamical system snapshot observations. PySHRED is a Python package that
implements SHRED and several of its major extensions, including for robust
sensing, reduced order modeling and physics discovery. In this paper, we
introduce the version 1.0 release of PySHRED, which includes data preprocessors
and a number of cutting-edge SHRED methods specifically designed to handle
real-world data that may be noisy, multi-scale, parameterized, prohibitively
high-dimensional, and strongly nonlinear. The package is easy to install,
thoroughly-documented, supplemented with extensive code examples, and
modularly-structured to support future additions. The entire codebase is
released under the MIT license and is available at
https://github.com/pyshred-dev/pyshred.

### 4. [AQUA: A Large Language Model for Aquaculture & Fisheries](http://arxiv.org/pdf/2507.20520v1)

Authors: Praneeth Narisetty, Uday Kumar Reddy Kattamanchi, Lohit Akshant Nimma, Sri Ram Kaushik Karnati, Shiva Nagendra Babu Kore, Mounika Golamari, Tejashree Nageshreddy

Aquaculture plays a vital role in global food security and coastal economies
by providing sustainable protein sources. As the industry expands to meet
rising demand, it faces growing challenges such as disease outbreaks,
inefficient feeding practices, rising labor costs, logistical inefficiencies,
and critical hatchery issues, including high mortality rates and poor water
quality control. Although artificial intelligence has made significant
progress, existing machine learning methods fall short of addressing the
domain-specific complexities of aquaculture. To bridge this gap, we introduce
AQUA, the first large language model (LLM) tailored for aquaculture, designed
to support farmers, researchers, and industry practitioners. Central to this
effort is AQUADAPT (Data Acquisition, Processing and Tuning), an Agentic
Framework for generating and refining high-quality synthetic data using a
combination of expert knowledge, largescale language models, and automated
evaluation techniques. Our work lays the foundation for LLM-driven innovations
in aquaculture research, advisory systems, and decision-making tools.

### Computational Geometry

### 1. [General Strong Bound on the Uncrossed Number which is Tight for the Edge Crossing Number](http://arxiv.org/pdf/2507.20937v1)

Authors: Gaspard Charvy, Tomáš Masařík

We investigate a very recent concept for visualizing various aspects of a
graph in the plane using a collection of drawings introduced by Hlin\v{e}n\'y
and Masa\v{r}\'ik [GD 2023]. Formally, given a graph $G$, we aim to find an
uncrossed collection containing drawings of $G$ in the plane such that each
edge of $G$ is not crossed in at least one drawing in the collection. The
uncrossed number of $G$ ($unc(G)$) is the smallest integer $k$ such that an
uncrossed collection for $G$ of size $k$ exists. The uncrossed number is
lower-bounded by the well-known thickness, which is an edge-decomposition of
$G$ into planar graphs. This connection gives a trivial lower-bound
$\lceil\frac{|E(G)|}{3|V(G)|-6}\rceil \le unc(G)$. In a recent paper, Balko,
Hlin\v{e}n\'y, Masa\v{r}\'ik, Orthaber, Vogtenhuber, and Wagner [GD 2024]
presented the first non-trivial and general lower-bound on the uncrossed
number. We summarize it in terms of dense graphs (where
$|E(G)|=\epsilon(|V(G)|)^2$ for some $\epsilon>0$):
$\lceil\frac{|E(G)|}{c_\epsilon|V(G)|}\rceil \le unc(G)$, where $c_\epsilon\ge
2.82$ is a constant depending on $\epsilon$.
  We improve the lower-bound to state that
$\lceil\frac{|E(G)|}{3|V(G)|-6-\sqrt{2|E(G)|}+\sqrt{6(|V(G)|-2)}}\rceil \le
unc(G)$. Translated to dense graphs regime, the bound yields a multiplicative
constant $c'_\epsilon=3-\sqrt{(2-\epsilon)}$ in the expression
$\lceil\frac{|E(G)|}{c'_\epsilon|V(G)|+o(|V(G)|)}\rceil \le unc(G)$. Hence, it
is tight (up to low-order terms) for $\epsilon \approx \frac{1}{2}$ as
warranted by complete graphs.
  In fact, we formulate our result in the language of the maximum uncrossed
subgraph number, that is, the maximum number of edges of $G$ that are not
crossed in a drawing of $G$ in the plane. In that case, we also provide a
construction certifying that our bound is asymptotically tight (up to low-order
terms) on dense graphs for all $\epsilon>0$.

### Computation and Language

### 1. [SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers](http://arxiv.org/pdf/2507.20527v1)

Authors: Chaitanya Manem, Pratik Prabhanjan Brahma, Prakamya Mishra, Zicheng Liu, Emad Barsoum

The demand for Large Language Models (LLMs) capable of sophisticated
mathematical reasoning is growing across industries. However, the development
of performant mathematical LLMs is critically bottlenecked by the scarcity of
difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic
Augmented Novel and Difficult Mathematics problems and solutions), a pipeline
that addresses this by first generating high-quality problems from scratch and
then systematically elevating their complexity via a new \textbf{Difficulty
Hiking} step. We demonstrate the effectiveness of our approach through two key
findings. First, augmenting a strong baseline with SAND-Math data significantly
boosts performance, outperforming the next-best synthetic dataset by
\textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a
dedicated ablation study, we show our Difficulty Hiking process is highly
effective: by increasing average problem difficulty from 5.02 to 5.98, this
step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation
pipeline, final dataset, and a fine-tuned model form a practical and scalable
toolkit for building more capable and efficient mathematical reasoning LLMs.
SAND-Math dataset is released here:
\href{https://huggingface.co/datasets/amd/SAND-MATH}{https://huggingface.co/datasets/amd/SAND-MATH}

### 2. [Dialogues of Dissent: Thematic and Rhetorical Dimensions of Hate and Counter-Hate Speech in Social Media Conversations](http://arxiv.org/pdf/2507.20528v1)

Authors: Effi Levi, Gal Ron, Odelia Oshri, Shaul R. Shenhav

We introduce a novel multi-labeled scheme for joint annotation of hate and
counter-hate speech in social media conversations, categorizing hate and
counter-hate messages into thematic and rhetorical dimensions. The thematic
categories outline different discursive aspects of each type of speech, while
the rhetorical dimension captures how hate and counter messages are
communicated, drawing on Aristotle's Logos, Ethos and Pathos. We annotate a
sample of 92 conversations, consisting of 720 tweets, and conduct statistical
analyses, incorporating public metrics, to explore patterns of interaction
between the thematic and rhetorical dimensions within and between hate and
counter-hate speech. Our findings provide insights into the spread of hate
messages on social media, the strategies used to counter them, and their
potential impact on online behavior.

### 3. [Before the Outrage: Challenges and Advances in Predicting Online Antisocial Behavior](http://arxiv.org/pdf/2507.20614v1)

Authors: Anaïs Ollagnier

Antisocial behavior (ASB) on social media-including hate speech, harassment,
and trolling-poses growing challenges for platform safety and societal
wellbeing. While prior work has primarily focused on detecting harmful content
after it appears, predictive approaches aim to forecast future harmful
behaviors-such as hate speech propagation, conversation derailment, or user
recidivism-before they fully unfold. Despite increasing interest, the field
remains fragmented, lacking a unified taxonomy or clear synthesis of existing
methods. This paper presents a systematic review of over 49 studies on ASB
prediction, offering a structured taxonomy of five core task types: early harm
detection, harm emergence prediction, harm propagation prediction, behavioral
risk prediction, and proactive moderation support. We analyze how these tasks
differ by temporal framing, prediction granularity, and operational goals. In
addition, we examine trends in modeling techniques-from classical machine
learning to pre-trained language models-and assess the influence of dataset
characteristics on task feasibility and generalization. Our review highlights
methodological challenges, such as dataset scarcity, temporal drift, and
limited benchmarks, while outlining emerging research directions including
multilingual modeling, cross-platform generalization, and human-in-the-loop
systems. By organizing the field around a coherent framework, this survey aims
to guide future work toward more robust and socially responsible ASB
prediction.

### 4. [Geometric-Mean Policy Optimization](http://arxiv.org/pdf/2507.20673v1)

Authors: Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, Furu Wei

Recent advancements, such as Group Relative Policy Optimization (GRPO), have
enhanced the reasoning capabilities of large language models by optimizing the
arithmetic mean of token-level rewards. However, GRPO suffers from unstable
policy updates when processing tokens with outlier importance-weighted rewards,
which manifests as extreme importance sampling ratios during training, i.e.,
the ratio between the sampling probabilities assigned to a token by the current
and old policies. In this work, we propose Geometric-Mean Policy Optimization
(GMPO), a stabilized variant of GRPO. Instead of optimizing the arithmetic
mean, GMPO maximizes the geometric mean of token-level rewards, which is
inherently less sensitive to outliers and maintains a more stable range of
importance sampling ratio. In addition, we provide comprehensive theoretical
and experimental analysis to justify the design and stability benefits of GMPO.
Beyond improved stability, GMPO-7B outperforms GRPO by an average of 4.1% on
multiple mathematical benchmarks and 1.4% on multimodal reasoning benchmark,
including AIME24, AMC, MATH500, OlympiadBench, Minerva, and Geometry3K. Code is
available at https://github.com/callsys/GMPO.

### 5. [When Scale Meets Diversity: Evaluating Language Models on Fine-Grained Multilingual Claim Verification](http://arxiv.org/pdf/2507.20700v1)

Authors: Hanna Shcharbakova, Tatiana Anikina, Natalia Skachkova, Josef van Genabith

The rapid spread of multilingual misinformation requires robust automated
fact verification systems capable of handling fine-grained veracity assessments
across diverse languages. While large language models have shown remarkable
capabilities across many NLP tasks, their effectiveness for multilingual claim
verification with nuanced classification schemes remains understudied. We
conduct a comprehensive evaluation of five state-of-the-art language models on
the X-Fact dataset, which spans 25 languages with seven distinct veracity
categories. Our experiments compare small language models (encoder-based XLM-R
and mT5) with recent decoder-only LLMs (Llama 3.1, Qwen 2.5, Mistral Nemo)
using both prompting and fine-tuning approaches. Surprisingly, we find that
XLM-R (270M parameters) substantially outperforms all tested LLMs (7-12B
parameters), achieving 57.7% macro-F1 compared to the best LLM performance of
16.9%. This represents a 15.8% improvement over the previous state-of-the-art
(41.9%), establishing new performance benchmarks for multilingual fact
verification. Our analysis reveals problematic patterns in LLM behavior,
including systematic difficulties in leveraging evidence and pronounced biases
toward frequent categories in imbalanced data settings. These findings suggest
that for fine-grained multilingual fact verification, smaller specialized
models may be more effective than general-purpose large models, with important
implications for practical deployment of fact-checking systems.

### 6. [On The Role of Pretrained Language Models in General-Purpose Text Embeddings: A Survey](http://arxiv.org/pdf/2507.20783v1)

Authors: Meishan Zhang, Xin Zhang, Xinping Zhao, Shouzheng Huang, Baotian Hu, Min Zhang

Text embeddings have attracted growing interest due to their effectiveness
across a wide range of natural language processing (NLP) tasks, such as
retrieval, classification, clustering, bitext mining, and summarization. With
the emergence of pretrained language models (PLMs), general-purpose text
embeddings (GPTE) have gained significant traction for their ability to produce
rich, transferable representations. The general architecture of GPTE typically
leverages PLMs to derive dense text representations, which are then optimized
through contrastive learning on large-scale pairwise datasets. In this survey,
we provide a comprehensive overview of GPTE in the era of PLMs, focusing on the
roles PLMs play in driving its development. We first examine the fundamental
architecture and describe the basic roles of PLMs in GPTE, i.e., embedding
extraction, expressivity enhancement, training strategies, learning objectives,
and data construction. Then, we describe advanced roles enabled by PLMs, such
as multilingual support, multimodal integration, code understanding, and
scenario-specific adaptation. Finally, we highlight potential future research
directions that move beyond traditional improvement goals, including ranking
integration, safety considerations, bias mitigation, structural information
incorporation, and the cognitive extension of embeddings. This survey aims to
serve as a valuable reference for both newcomers and established researchers
seeking to understand the current state and future potential of GPTE.

### 7. [Automating Thematic Review of Prevention of Future Deaths Reports: Replicating the ONS Child Suicide Study using Large Language Models](http://arxiv.org/pdf/2507.20786v1)

Authors: Sam Osian, Arpan Dutta, Sahil Bhandari, Iain E. Buchan, Dan W. Joyce

Prevention of Future Deaths (PFD) reports, issued by coroners in England and
Wales, flag systemic hazards that may lead to further loss of life. Analysis of
these reports has previously been constrained by the manual effort required to
identify and code relevant cases. In 2025, the Office for National Statistics
(ONS) published a national thematic review of child-suicide PFD reports ($\leq$
18 years), identifying 37 cases from January 2015 to November 2023 - a process
based entirely on manual curation and coding. We evaluated whether a fully
automated, open source "text-to-table" language-model pipeline (PFD Toolkit)
could reproduce the ONS's identification and thematic analysis of child-suicide
PFD reports, and assessed gains in efficiency and reliability. All 4,249 PFD
reports published from July 2013 to November 2023 were processed via PFD
Toolkit's large language model pipelines. Automated screening identified cases
where the coroner attributed death to suicide in individuals aged 18 or
younger, and eligible reports were coded for recipient category and 23 concern
sub-themes, replicating the ONS coding frame. PFD Toolkit identified 72
child-suicide PFD reports - almost twice the ONS count. Three blinded
clinicians adjudicated a stratified sample of 144 reports to validate the
child-suicide screening. Against the post-consensus clinical annotations, the
LLM-based workflow showed substantial to almost-perfect agreement (Cohen's
$\kappa$ = 0.82, 95% CI: 0.66-0.98, raw agreement = 91%). The end-to-end script
runtime was 8m 16s, transforming a process that previously took months into one
that can be completed in minutes. This demonstrates that automated LLM analysis
can reliably and efficiently replicate manual thematic reviews of coronial
data, enabling scalable, reproducible, and timely insights for public health
and safety. The PFD Toolkit is openly available for future research.

### 8. [Latent Inter-User Difference Modeling for LLM Personalization](http://arxiv.org/pdf/2507.20849v1)

Authors: Yilun Qiu, Tianhao Shi, Xiaoyan Zhao, Fengbin Zhu, Yang Zhang, Fuli Feng

Large language models (LLMs) are increasingly integrated into users' daily
lives, leading to a growing demand for personalized outputs. Previous work
focuses on leveraging a user's own history, overlooking inter-user differences
that are crucial for effective personalization. While recent work has attempted
to model such differences, the reliance on language-based prompts often hampers
the effective extraction of meaningful distinctions. To address these issues,
we propose Difference-aware Embedding-based Personalization (DEP), a framework
that models inter-user differences in the latent space instead of relying on
language prompts. DEP constructs soft prompts by contrasting a user's embedding
with those of peers who engaged with similar content, highlighting relative
behavioral signals. A sparse autoencoder then filters and compresses both
user-specific and difference-aware embeddings, preserving only task-relevant
features before injecting them into a frozen LLM. Experiments on personalized
review generation show that DEP consistently outperforms baseline methods
across multiple metrics. Our code is available at
https://github.com/SnowCharmQ/DEP.

### 9. [A survey of diversity quantification in natural language processing: The why, what, where and how](http://arxiv.org/pdf/2507.20858v1)

Authors: Louis Estève, Marie-Catherine de Marneffe, Nurit Melnik, Agata Savary, Olha Kanishcheva

The concept of diversity has received increased consideration in Natural
Language Processing (NLP) in recent years. This is due to various motivations
like promoting and inclusion, approximating human linguistic behavior, and
increasing systems' performance. Diversity has however often been addressed in
an ad hoc manner in NLP, and with few explicit links to other domains where
this notion is better theorized. We survey articles in the ACL Anthology from
the past 6 years, with "diversity" or "diverse" in their title. We find a wide
range of settings in which diversity is quantified, often highly specialized
and using inconsistent terminology. We put forward a unified taxonomy of why,
what on, where, and how diversity is measured in NLP. Diversity measures are
cast upon a unified framework from ecology and economy (Stirling, 2007) with 3
dimensions of diversity: variety, balance and disparity. We discuss the trends
which emerge due to this systematized approach. We believe that this study
paves the way towards a better formalization of diversity in NLP, which should
bring a better understanding of this notion and a better comparability between
various approaches.

### 10. [Leveraging Open-Source Large Language Models for Clinical Information Extraction in Resource-Constrained Settings](http://arxiv.org/pdf/2507.20859v1)

Authors: Luc Builtjes, Joeran Bosma, Mathias Prokop, Bram van Ginneken, Alessa Hering

Medical reports contain rich clinical information but are often unstructured
and written in domain-specific language, posing challenges for information
extraction. While proprietary large language models (LLMs) have shown promise
in clinical natural language processing, their lack of transparency and data
privacy concerns limit their utility in healthcare. This study therefore
evaluates nine open-source generative LLMs on the DRAGON benchmark, which
includes 28 clinical information extraction tasks in Dutch. We developed
\texttt{llm\_extractinator}, a publicly available framework for information
extraction using open-source generative LLMs, and used it to assess model
performance in a zero-shot setting. Several 14 billion parameter models,
Phi-4-14B, Qwen-2.5-14B, and DeepSeek-R1-14B, achieved competitive results,
while the bigger Llama-3.3-70B model achieved slightly higher performance at
greater computational cost. Translation to English prior to inference
consistently degraded performance, highlighting the need of native-language
processing. These findings demonstrate that open-source LLMs, when used with
our framework, offer effective, scalable, and privacy-conscious solutions for
clinical information extraction in low-resource settings.

### Cryptography and Security

### 1. [MPC-EVM: Enabling MPC Execution by Smart Contracts In An Asynchronous Manner](http://arxiv.org/pdf/2507.20554v1)

Authors: Yichen Zhou, Chenxing Li, Fan Long

This paper presents MPC-EVM, the first blockchain prototype that extends the
EVM to enable asynchronous MPC invocations by smart contracts during
transaction executions without compromising consistency or throughput. MPC-EVM
uses an asynchronous execution model to process MPC-invoking transactions in a
non-blocking fashion, saving the transaction's progress when it enters an MPC
and resuming its execution upon MPC's completion. Additionally, it employs an
access control mechanism that prevents inconsistent state access and
modifications as a result of asynchronous executions. Benchmarking MPC-EVM's
throughput show that the transactions per second (TPS) decreased by less than
3% compared to the baseline when MPC-invoking transactions are executed
alongside regular transactions.

### 2. [Guard-GBDT: Efficient Privacy-Preserving Approximated GBDT Training on Vertical Dataset](http://arxiv.org/pdf/2507.20688v1)

Authors: Anxiao Song, Shujie Cui, Jianli Bai, Ke Cheng, Yulong Shen, Giovanni Russello

In light of increasing privacy concerns and stringent legal regulations,
using secure multiparty computation (MPC) to enable collaborative GBDT model
training among multiple data owners has garnered significant attention. Despite
this, existing MPC-based GBDT frameworks face efficiency challenges due to high
communication costs and the computation burden of non-linear operations, such
as division and sigmoid calculations. In this work, we introduce Guard-GBDT, an
innovative framework tailored for efficient and privacy-preserving GBDT
training on vertical datasets. Guard-GBDT bypasses MPC-unfriendly division and
sigmoid functions by using more streamlined approximations and reduces
communication overhead by compressing the messages exchanged during gradient
aggregation. We implement a prototype of Guard-GBDT and extensively evaluate
its performance and accuracy on various real-world datasets. The results show
that Guard-GBDT outperforms state-of-the-art HEP-XGB (CIKM'21) and SiGBDT (ASIA
CCS'24) by up to $2.71\times$ and $12.21 \times$ on LAN network and up to
$2.7\times$ and $8.2\times$ on WAN network. Guard-GBDT also achieves comparable
accuracy with SiGBDT and plaintext XGBoost (better than HEP-XGB ), which
exhibits a deviation of $\pm1\%$ to $\pm2\%$ only. Our implementation code is
provided at https://github.com/XidianNSS/Guard-GBDT.git.

### 3. [An Open-source Implementation and Security Analysis of Triad's TEE Trusted Time Protocol](http://arxiv.org/pdf/2507.20851v1)

Authors: Matthieu Bettinger, Sonia Ben Mokhtar, Anthony Simonet-Boulogne

The logic of many protocols relies on time measurements. However, in Trusted
Execution Environments (TEEs) like Intel SGX, the time source is outside the
Trusted Computing Base: a malicious system hosting the TEE can manipulate that
TEE's notion of time, e.g., jumping in time or affecting the perceived time
speed. Previous work like Triad propose protocols for TEEs to maintain a
trustworthy time source. However, in this paper, based on a public
implementation of Triad that we contribute, we empirically showcase
vulnerabilities to this protocol. For example, an attacker controlling the
operating system, and consequently the scheduling algorithm, may arbitrarily
manipulate their local TEE's clock speed. What is worse, in case of faster
malicious clock speeds, an attacker on a single compromised machine may
propagate the attack to honest machines participating in Triad's Trusted Time
protocol, causing them to skip to timestamps arbitrarily far in the future.
Then, infected honest machines propagate time-skips themselves to other honest
machines interacting with them. We discuss protocol changes to Triad for higher
resilience against such attacks.

### 4. [Characterizing the Sensitivity to Individual Bit Flips in Client-Side Operations of the CKKS Scheme](http://arxiv.org/pdf/2507.20891v1)

Authors: Matias Mazzanti, Augusto Vega, Esteban Mocskos

Homomorphic Encryption (HE) enables computation on encrypted data without
decryption, making it a cornerstone of privacy-preserving computation in
untrusted environments. As HE sees growing adoption in sensitive applications
such as secure machine learning and confidential data analysis ensuring its
robustness against errors becomes critical. Faults (e.g., transmission errors,
hardware malfunctions, or synchronization failures) can corrupt encrypted data
and compromise the integrity of HE operations. However, the impact of soft
errors (such as bit flips) on modern HE schemes remains unexplored.
Specifically, the CKKS scheme-one of the most widely used HE schemes for
approximate arithmetic-lacks a systematic study of how such errors propagate
across its pipeline, particularly under optimizations like the Residue Number
System (RNS) and Number Theoretic Transform (NTT). This work bridges that gap
by presenting a theoretical and empirical analysis of CKKS's fault tolerance
under single bit-flip errors. We focus on client-side operations (encoding,
encryption, decryption, and decoding) and demonstrate that while the vanilla
CKKS scheme exhibits some resilience, performance optimizations (RNS/NTT)
introduce significant fragility, amplifying error sensitivity. By
characterizing these failure modes, we lay the groundwork for error-resilient
HE designs, ensuring both performance and integrity in privacy-critical
applications.

### 5. [Development and analysis of a secured VoIP system for surveillance activities](http://arxiv.org/pdf/2507.21038v1)

Authors: M. Matsive Ali

Since the 1990s, the telephone has been the primary mode of communication.
However, Voice over Internet Protocol (VoIP), which is a highly straightforward
and affordable form of data transfer, is now becoming an important part of
daily communication. VoIP is the technology that makes it possible to send
speech and multimedia data packets across either a public or private IP
network. However, a cyberattack known as a man-in-the-middle attack poses a
serious concern in transferring data through any network. Therefore, the
authors have designed a system that sends voice over the internet within the
range of a router using encrypted data transfer. An embedded system comprising
an electret microphone, Embedded C, Node.js, Particle Photon microcontroller,
and Internet of Things (IoT) technology is developed. Due to its compact size,
this type of device may be incorporated into automobiles, surveillance systems,
or covert listening tools. The VoIP system gathers sound signals using the
MAX9814 microphone, while the Particle Photon microcontroller securely
transmits the data. Devices with access can download data from the VoIP systems
Transmission Control Protocol (TCP) server. The accessed device stores the
audio locally and uploads the corresponding data to Google Drive. This VoIP
system provides a secure method of communication while conserving the integrity
of the original signal.

### 6. [VDGraph: A Graph-Theoretic Approach to Unlock Insights from SBOM and SCA Data](http://arxiv.org/pdf/2507.20502v1)

Authors: Howell Xia, Jonah Gluck, Sevval Simsek, David Sastre Medina, David Starobinski

The high complexity of modern software supply chains necessitates tools such
as Software Bill of Materials (SBOMs) to manage component dependencies, and
Software Composition Analysis (SCA) tools to identify vulnerabilities. While
there exists limited integration between SBOMs and SCA tools, a unified view of
complex dependency-vulnerability relationships remains elusive. In this paper,
we introduce VDGraph, a novel knowledge graph-based methodology for integrating
vulnerability and dependency data into a holistic view. VDGraph consolidates
SBOM and SCA outputs into a graph representation of software projects'
dependencies and vulnerabilities. We provide a formal description and analysis
of the theoretical properties of VDGraph and present solutions to manage
possible conflicts between the SBOM and SCA data. We further introduce and
evaluate a practical, proof-of-concept implementation of VDGraph using two
popular SBOM and SCA tools, namely CycloneDX Maven plugin and Google's
OSV-Scanner. We apply VDGraph on 21 popular Java projects. Through the
formulation of appropriate queries on the graphs, we uncover the existence of
concentrated risk points (i.e., vulnerable components of high severity
reachable through numerous dependency paths). We further show that
vulnerabilities predominantly emerge at a depth of three dependency levels or
higher, indicating that direct or secondary dependencies exhibit lower
vulnerability density and tend to be more secure. Thus, VDGraph contributes a
graph-theoretic methodology that improves visibility into how vulnerabilities
propagate through complex, transitive dependencies. Moreover, our
implementation, which combines open SBOM and SCA standards with Neo4j, lays a
foundation for scalable and automated analysis across real-world projects.

### 7. [Hot-Swap MarkBoard: An Efficient Black-box Watermarking Approach for Large-scale Model Distribution](http://arxiv.org/pdf/2507.20650v1)

Authors: Zhicheng Zhang, Peizhuo Lv, Mengke Wan, Jiang Fang, Diandian Guo, Yezeng Chen, Yinlong Liu, Wei Ma, Jiyan Sun, Liru Geng

Recently, Deep Learning (DL) models have been increasingly deployed on
end-user devices as On-Device AI, offering improved efficiency and privacy.
However, this deployment trend poses more serious Intellectual Property (IP)
risks, as models are distributed on numerous local devices, making them
vulnerable to theft and redistribution. Most existing ownership protection
solutions (e.g., backdoor-based watermarking) are designed for cloud-based
AI-as-a-Service (AIaaS) and are not directly applicable to large-scale
distribution scenarios, where each user-specific model instance must carry a
unique watermark. These methods typically embed a fixed watermark, and
modifying the embedded watermark requires retraining the model. To address
these challenges, we propose Hot-Swap MarkBoard, an efficient watermarking
method. It encodes user-specific $n$-bit binary signatures by independently
embedding multiple watermarks into a multi-branch Low-Rank Adaptation (LoRA)
module, enabling efficient watermark customization without retraining through
branch swapping. A parameter obfuscation mechanism further entangles the
watermark weights with those of the base model, preventing removal without
degrading model performance. The method supports black-box verification and is
compatible with various model architectures and DL tasks, including
classification, image generation, and text generation. Extensive experiments
across three types of tasks and six backbone models demonstrate our method's
superior efficiency and adaptability compared to existing approaches, achieving
100\% verification accuracy.

### 8. [Program Analysis for High-Value Smart Contract Vulnerabilities: Techniques and Insights](http://arxiv.org/pdf/2507.20672v1)

Authors: Yannis Smaragdakis, Neville Grech, Sifis Lagouvardos, Konstantinos Triantafyllou, Ilias Tsatiris, Yannis Bollanos, Tony Rocco Valentine

A widespread belief in the blockchain security community is that automated
techniques are only good for detecting shallow bugs, typically of small value.
In this paper, we present the techniques and insights that have led us to
repeatable success in automatically discovering high-value smart contract
vulnerabilities. Our vulnerability disclosures have yielded 10 bug bounties,
for a total of over $3M, over high-profile deployed code, as well as hundreds
of bugs detected in pre-deployment or under-audit code.
  We argue that the elements of this surprising success are a) a very
high-completeness static analysis approach that manages to maintain acceptable
precision; b) domain knowledge, provided by experts or captured via statistical
inference. We present novel techniques for automatically inferring domain
knowledge from statistical analysis of a large corpus of deployed contracts, as
well as discuss insights on the ideal precision and warning rate of a promising
vulnerability detector. In contrast to academic literature in program analysis,
which routinely expects false-positive rates below 50% for publishable results,
we posit that a useful analysis for high-value real-world vulnerabilities will
likely flag very few programs (under 1%) and will do so with a high
false-positive rate (e.g., 95%, meaning that only one-of-twenty human
inspections will yield an exploitable vulnerability).

### 9. [A Novel Post-Quantum Secure Digital Signature Scheme Based on Neural Network](http://arxiv.org/pdf/2507.20676v1)

Authors: Satish Kumar, Md. Arzoo Jamal

Digital signatures are fundamental cryptographic primitives that ensure the
authenticity and integrity of digital documents. In the post-quantum era,
classical public key-based signature schemes become vulnerable to brute-force
and key-recovery attacks due to the computational power of quantum algorithms.
Multivariate polynomial based signature schemes are among the one of the
cryptographic constructions that offers strong security guarantees against such
quantum threats. With the growing capabilities of neural networks, it is
natural to explore their potential application in the design of cryptographic
primitives. Neural networks inherently captures the non-linear relationships
within the data, which are encoded in their synaptic weight matrices and bias
vectors. In this paper, we propose a novel construction of a multivariate
polynomial based digital signature scheme that leverages neural network
architectures. A neural network with binary weights is employed to define the
central structure of the signature scheme. The design introduces a recurrent
random vector, functionally analogous to an attention mechanism, which
contributes dynamic randomness based on the previous state, thereby enhancing
the scheme's security. It is demonstrated that the proposed signature scheme
provide security against Existential Unforgeability under adaptive
Chosen-Message Attacks (EUF-CMA). Furthermore, it is proven that direct attacks
aimed to recover the private keys are computationally infeasible within
polynomial time, even in the presence of quantum computing abilities. The
operational characteristics of the proposed scheme are also evaluated, with
results indicating notable efficiency and practical viability in post-quantum
cryptographic applications.

### 10. [Collusion Resistant DNS With Private Information Retrieval](http://arxiv.org/pdf/2507.20806v1)

Authors: Yunming Xiao, Peizhi Liu, Ruijie Yu, Chenkai Weng, Matteo Varvello, Aleksandar Kuzmanovic

There has been a growing interest in Internet user privacy, demonstrated by
the popularity of privacy-preserving products such as Telegram and Brave, and
the widespread adoption of HTTPS. The Domain Name System (DNS) is a key
component of Internet-based communication and its privacy has been neglected
for years. Recently, DNS over HTTPS (DoH) has improved the situation by fixing
the issue of in-path middleboxes. Further progress has been made with
proxy-based solutions such as Oblivious DoH (ODoH), which separate a user's
identity from their DNS queries. However, these solutions rely on non-collusion
assumptions between DNS resolvers and proxies -- an assumption difficult to
guarantee in practice. To address this, we explore integrating single-server
Private Information Retrieval (PIR) into DNS to enable encrypted query
processing without relying on trust assumptions. However, applying PIR to DNS
is challenging due to its hierarchical nature -- particularly, interactions
with recursive resolvers can still leak information. Navigating performance and
privacy trade-offs, we propose PDNS, a DNS extension leveraging single-server
PIR to strengthen privacy guarantees. We have implemented a prototype of PDNS
and compared its performance against state-of-the-art solutions via
trace-driven experiments. The results show that PDNS achieves acceptable
performance (2x faster than DoH over Tor with similar privacy guarantees) and
strong privacy guarantees today, mainly at the cost of its scalability, which
specialized hardware for PIR can address in the near future.

### Computer Vision and Pattern Recognition

### 1. [JOLT3D: Joint Learning of Talking Heads and 3DMM Parameters with Application to Lip-Sync](http://arxiv.org/pdf/2507.20452v1)

Authors: Sungjoon Park, Minsik Park, Haneol Lee, Jaesub Yun, Donggeon Lee

In this work, we revisit the effectiveness of 3DMM for talking head synthesis
by jointly learning a 3D face reconstruction model and a talking head synthesis
model. This enables us to obtain a FACS-based blendshape representation of
facial expressions that is optimized for talking head synthesis. This contrasts
with previous methods that either fit 3DMM parameters to 2D landmarks or rely
on pretrained face reconstruction models. Not only does our approach increase
the quality of the generated face, but it also allows us to take advantage of
the blendshape representation to modify just the mouth region for the purpose
of audio-based lip-sync. To this end, we propose a novel lip-sync pipeline
that, unlike previous methods, decouples the original chin contour from the
lip-synced chin contour, and reduces flickering near the mouth.

### 2. [Priority-Aware Pathological Hierarchy Training for Multiple Instance Learning](http://arxiv.org/pdf/2507.20469v1)

Authors: Sungrae Hong, Kyungeun Kim, Juhyeon Kim, Sol Lee, Jisu Shin, Chanjae Song, Mun Yong Yi

Multiple Instance Learning (MIL) is increasingly being used as a support tool
within clinical settings for pathological diagnosis decisions, achieving high
performance and removing the annotation burden. However, existing approaches
for clinical MIL tasks have not adequately addressed the priority issues that
exist in relation to pathological symptoms and diagnostic classes, causing MIL
models to ignore priority among classes. To overcome this clinical limitation
of MIL, we propose a new method that addresses priority issues using two
hierarchies: vertical inter-hierarchy and horizontal intra-hierarchy. The
proposed method aligns MIL predictions across each hierarchical level and
employs an implicit feature re-usability during training to facilitate
clinically more serious classes within the same level. Experiments with
real-world patient data show that the proposed method effectively reduces
misdiagnosis and prioritizes more important symptoms in multiclass scenarios.
Further analysis verifies the efficacy of the proposed components and
qualitatively confirms the MIL predictions against challenging cases with
multiple symptoms.

### 3. [Automated 3D-GS Registration and Fusion via Skeleton Alignment and Gaussian-Adaptive Features](http://arxiv.org/pdf/2507.20480v1)

Authors: Shiyang Liu, Dianyi Yang, Yu Gao, Bohan Ren, Yi Yang, Mengyin Fu

In recent years, 3D Gaussian Splatting (3D-GS)-based scene representation
demonstrates significant potential in real-time rendering and training
efficiency. However, most existing methods primarily focus on single-map
reconstruction, while the registration and fusion of multiple 3D-GS sub-maps
remain underexplored. Existing methods typically rely on manual intervention to
select a reference sub-map as a template and use point cloud matching for
registration. Moreover, hard-threshold filtering of 3D-GS primitives often
degrades rendering quality after fusion. In this paper, we present a novel
approach for automated 3D-GS sub-map alignment and fusion, eliminating the need
for manual intervention while enhancing registration accuracy and fusion
quality. First, we extract geometric skeletons across multiple scenes and
leverage ellipsoid-aware convolution to capture 3D-GS attributes, facilitating
robust scene registration. Second, we introduce a multi-factor Gaussian fusion
strategy to mitigate the scene element loss caused by rigid thresholding.
Experiments on the ScanNet-GSReg and our Coord datasets demonstrate the
effectiveness of the proposed method in registration and fusion. For
registration, it achieves a 41.9\% reduction in RRE on complex scenes, ensuring
more precise pose estimation. For fusion, it improves PSNR by 10.11 dB,
highlighting superior structural preservation. These results confirm its
ability to enhance scene alignment and reconstruction fidelity, ensuring more
consistent and accurate 3D scene representation for robotic perception and
autonomous navigation.

### 4. [An Improved YOLOv8 Approach for Small Target Detection of Rice Spikelet Flowering in Field Environments](http://arxiv.org/pdf/2507.20506v1)

Authors: Beizhang Chen, Jinming Liang, Zheng Xiong, Ming Pan, Xiangbao Meng, Qingshan Lin, Qun Ma, Yingping Zhao

Accurately detecting rice flowering time is crucial for timely pollination in
hybrid rice seed production. This not only enhances pollination efficiency but
also ensures higher yields. However, due to the complexity of field
environments and the characteristics of rice spikelets, such as their small
size and short flowering period, automated and precise recognition remains
challenging. To address this, this study proposes a rice spikelet flowering
recognition method based on an improved YOLOv8 object detection model. First, a
Bidirectional Feature Pyramid Network (BiFPN) replaces the original PANet
structure to enhance feature fusion and improve multi-scale feature
utilization. Second, to boost small object detection, a p2 small-object
detection head is added, using finer feature mapping to reduce feature loss
commonly seen in detecting small targets. Given the lack of publicly available
datasets for rice spikelet flowering in field conditions, a high-resolution RGB
camera and data augmentation techniques are used to construct a dedicated
dataset, providing reliable support for model training and testing.
Experimental results show that the improved YOLOv8s-p2 model achieves an
mAP@0.5 of 65.9%, precision of 67.6%, recall of 61.5%, and F1-score of 64.41%,
representing improvements of 3.10%, 8.40%, 10.80%, and 9.79%, respectively,
over the baseline YOLOv8. The model also runs at 69 f/s on the test set,
meeting practical application requirements. Overall, the improved YOLOv8s-p2
offers high accuracy and speed, providing an effective solution for automated
monitoring in hybrid rice seed production.

### 5. [Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification](http://arxiv.org/pdf/2507.20511v1)

Authors: Wei Zhuo, Runjie Luo, Wufeng Xue, Linlin Shen

Few-shot Learning (FSL), which endeavors to develop the generalization
ability for recognizing novel classes using only a few images, faces
significant challenges due to data scarcity. Recent CLIP-like methods based on
contrastive language-image pertaining mitigate the issue by leveraging textual
representation of the class name for unseen image discovery. Despite the
achieved success, simply aligning visual representations to class name
embeddings would compromise the visual diversity for novel class
discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method
(BCT-CLIP) that explores \textbf{dominating properties} via contrastive
learning beyond simply using class tokens. Through leveraging LLM-based prior
knowledge, our method pushes forward FSL with comprehensive structural image
representations, including both global category representation and the
patch-aware property embeddings. In particular, we presented a novel
multi-property generator (MPG) with patch-aware cross-attentions to generate
multiple visual property tokens, a Large-Language Model (LLM)-assistant
retrieval procedure with clustering-based pruning to obtain dominating property
descriptions, and a new contrastive learning strategy for property-token
learning. The superior performances on the 11 widely used datasets demonstrate
that our investigation of dominating properties advances discriminative
class-specific representation learning and few-shot classification.

### 6. [GaRe: Relightable 3D Gaussian Splatting for Outdoor Scenes from Unconstrained Photo Collections](http://arxiv.org/pdf/2507.20512v1)

Authors: Haiyang Bai, Jiaqi Zhu, Songru Jiang, Wei Huang, Tao Lu, Yuanqi Li, Jie Guo, Runze Fu, Yanwen Guo, Lijun Chen

We propose a 3D Gaussian splatting-based framework for outdoor relighting
that leverages intrinsic image decomposition to precisely integrate sunlight,
sky radiance, and indirect lighting from unconstrained photo collections.
Unlike prior methods that compress the per-image global illumination into a
single latent vector, our approach enables simultaneously diverse shading
manipulation and the generation of dynamic shadow effects. This is achieved
through three key innovations: (1) a residual-based sun visibility extraction
method to accurately separate direct sunlight effects, (2) a region-based
supervision framework with a structural consistency loss for physically
interpretable and coherent illumination decomposition, and (3) a
ray-tracing-based technique for realistic shadow simulation. Extensive
experiments demonstrate that our framework synthesizes novel views with
competitive fidelity against state-of-the-art relighting solutions and produces
more natural and multifaceted illumination and shadow effects.

### 7. [AgroBench: Vision-Language Model Benchmark in Agriculture](http://arxiv.org/pdf/2507.20519v1)

Authors: Risa Shinoda, Nakamasa Inoue, Hirokatsu Kataoka, Masaki Onishi, Yoshitaka Ushiku

Precise automated understanding of agricultural tasks such as disease
identification is essential for sustainable crop production. Recent advances in
vision-language models (VLMs) are expected to further expand the range of
agricultural tasks by facilitating human-model interaction through easy,
text-based communication. Here, we introduce AgroBench (Agronomist AI
Benchmark), a benchmark for evaluating VLM models across seven agricultural
topics, covering key areas in agricultural engineering and relevant to
real-world farming. Unlike recent agricultural VLM benchmarks, AgroBench is
annotated by expert agronomists. Our AgroBench covers a state-of-the-art range
of categories, including 203 crop categories and 682 disease categories, to
thoroughly evaluate VLM capabilities. In our evaluation on AgroBench, we reveal
that VLMs have room for improvement in fine-grained identification tasks.
Notably, in weed identification, most open-source VLMs perform close to random.
With our wide range of topics and expert-annotated categories, we analyze the
types of errors made by VLMs and suggest potential pathways for future VLM
development. Our dataset and code are available at
https://dahlian00.github.io/AgroBenchPage/ .

### 8. [Low-Cost Machine Vision System for Sorting Green Lentils (Lens Culinaris) Based on Pneumatic Ejection and Deep Learning](http://arxiv.org/pdf/2507.20531v1)

Authors: Davy Rojas Yana, Edwin Salcedo

This paper presents the design, development, and evaluation of a dynamic
grain classification system for green lentils (Lens Culinaris), which leverages
computer vision and pneumatic ejection. The system integrates a YOLOv8-based
detection model that identifies and locates grains on a conveyor belt, together
with a second YOLOv8-based classification model that categorises grains into
six classes: Good, Yellow, Broken, Peeled, Dotted, and Reject. This two-stage
YOLOv8 pipeline enables accurate, real-time, multi-class categorisation of
lentils, implemented on a low-cost, modular hardware platform. The pneumatic
ejection mechanism separates defective grains, while an Arduino-based control
system coordinates real-time interaction between the vision system and
mechanical components. The system operates effectively at a conveyor speed of
59 mm/s, achieving a grain separation accuracy of 87.2%. Despite a limited
processing rate of 8 grams per minute, the prototype demonstrates the potential
of machine vision for grain sorting and provides a modular foundation for
future enhancements.

### 9. [Annotation-Free Human Sketch Quality Assessment](http://arxiv.org/pdf/2507.20548v1)

Authors: Lan Yang, Kaiyue Pang, Honggang Zhang, Yi-Zhe Song

As lovely as bunnies are, your sketched version would probably not do them
justice (Fig.~\ref{fig:intro}). This paper recognises this very problem and
studies sketch quality assessment for the first time -- letting you find these
badly drawn ones. Our key discovery lies in exploiting the magnitude ($L_2$
norm) of a sketch feature as a quantitative quality metric. We propose
Geometry-Aware Classification Layer (GACL), a generic method that makes
feature-magnitude-as-quality-metric possible and importantly does it without
the need for specific quality annotations from humans. GACL sees feature
magnitude and recognisability learning as a dual task, which can be
simultaneously optimised under a neat cross-entropy classification loss with
theoretic guarantee. This gives GACL a nice geometric interpretation (the
better the quality, the easier the recognition), and makes it agnostic to both
network architecture changes and the underlying sketch representation. Through
a large scale human study of 160,000 \doublecheck{trials}, we confirm the
agreement between our GACL-induced metric and human quality perception. We
further demonstrate how such a quality assessment capability can for the first
time enable three practical sketch applications. Interestingly, we show GACL
not only works on abstract visual representations such as sketch but also
extends well to natural images on the problem of image quality assessment
(IQA). Last but not least, we spell out the general properties of GACL as
general-purpose data re-weighting strategy and demonstrate its applications in
vertical problems such as noisy label cleansing. Code will be made publicly
available at github.com/yanglan0225/SketchX-Quantifying-Sketch-Quality.

### 10. [FED-PsyAU: Privacy-Preserving Micro-Expression Recognition via Psychological AU Coordination and Dynamic Facial Motion Modeling](http://arxiv.org/pdf/2507.20557v1)

Authors: Jingting Li, Yu Qian, Lin Zhao, Su-Jing Wang

Micro-expressions (MEs) are brief, low-intensity, often localized facial
expressions. They could reveal genuine emotions individuals may attempt to
conceal, valuable in contexts like criminal interrogation and psychological
counseling. However, ME recognition (MER) faces challenges, such as small
sample sizes and subtle features, which hinder efficient modeling.
Additionally, real-world applications encounter ME data privacy issues, leaving
the task of enhancing recognition across settings under privacy constraints
largely unexplored. To address these issues, we propose a FED-PsyAU research
framework. We begin with a psychological study on the coordination of upper and
lower facial action units (AUs) to provide structured prior knowledge of facial
muscle dynamics. We then develop a DPK-GAT network that combines these
psychological priors with statistical AU patterns, enabling hierarchical
learning of facial motion features from regional to global levels, effectively
enhancing MER performance. Additionally, our federated learning framework
advances MER capabilities across multiple clients without data sharing,
preserving privacy and alleviating the limited-sample issue for each client.
Extensive experiments on commonly-used ME databases demonstrate the
effectiveness of our approach.

### Computers and Society

### 1. [The Xeno Sutra: Can Meaning and Value be Ascribed to an AI-Generated "Sacred" Text?](http://arxiv.org/pdf/2507.20525v1)

Authors: Murray Shanahan, Tara Das, Robert Thurman

This paper presents a case study in the use of a large language model to
generate a fictional Buddhist "sutr"', and offers a detailed analysis of the
resulting text from a philosophical and literary point of view. The conceptual
subtlety, rich imagery, and density of allusion found in the text make it hard
to causally dismiss on account of its mechanistic origin. This raises questions
about how we, as a society, should come to terms with the potentially
unsettling possibility of a technology that encroaches on human meaning-making.
We suggest that Buddhist philosophy, by its very nature, is well placed to
adapt.

### 2. [Customize Multi-modal RAI Guardrails with Precedent-based predictions](http://arxiv.org/pdf/2507.20503v1)

Authors: Cheng-Fu Yang, Thanh Tran, Christos Christodoulopoulos, Weitong Ruan, Rahul Gupta, Kai-Wei Chang

A multi-modal guardrail must effectively filter image content based on
user-defined policies, identifying material that may be hateful, reinforce
harmful stereotypes, contain explicit material, or spread misinformation.
Deploying such guardrails in real-world applications, however, poses
significant challenges. Users often require varied and highly customizable
policies and typically cannot provide abundant examples for each custom policy.
Consequently, an ideal guardrail should be scalable to the multiple policies
and adaptable to evolving user standards with minimal retraining. Existing
fine-tuning methods typically condition predictions on pre-defined policies,
restricting their generalizability to new policies or necessitating extensive
retraining to adapt. Conversely, training-free methods struggle with limited
context lengths, making it difficult to incorporate all the policies
comprehensively. To overcome these limitations, we propose to condition model's
judgment on "precedents", which are the reasoning processes of prior data
points similar to the given input. By leveraging precedents instead of fixed
policies, our approach greatly enhances the flexibility and adaptability of the
guardrail. In this paper, we introduce a critique-revise mechanism for
collecting high-quality precedents and two strategies that utilize precedents
for robust prediction. Experimental results demonstrate that our approach
outperforms previous methods across both few-shot and full-dataset scenarios
and exhibits superior generalization to novel policies.

### 3. [Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition](http://arxiv.org/pdf/2507.20526v1)

Authors: Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Alexander Grattan, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Nate Burnikell, Yarin Gal, Dan Hendrycks, J. Zico Kolter, Matt Fredrikson

Recent advances have enabled LLM-powered AI agents to autonomously execute
complex tasks by combining language model reasoning with tools, memory, and web
access. But can these systems be trusted to follow deployment policies in
realistic environments, especially under attack? To investigate, we ran the
largest public red-teaming competition to date, targeting 22 frontier AI agents
across 44 realistic deployment scenarios. Participants submitted 1.8 million
prompt-injection attacks, with over 60,000 successfully eliciting policy
violations such as unauthorized data access, illicit financial actions, and
regulatory noncompliance. We use these results to build the Agent Red Teaming
(ART) benchmark - a curated set of high-impact attacks - and evaluate it across
19 state-of-the-art models. Nearly all agents exhibit policy violations for
most behaviors within 10-100 queries, with high attack transferability across
models and tasks. Importantly, we find limited correlation between agent
robustness and model size, capability, or inference-time compute, suggesting
that additional defenses are needed against adversarial misuse. Our findings
highlight critical and persistent vulnerabilities in today's AI agents. By
releasing the ART benchmark and accompanying evaluation framework, we aim to
support more rigorous security assessment and drive progress toward safer agent
deployment.

### 4. [Learning the Value Systems of Societies from Preferences](http://arxiv.org/pdf/2507.20728v1)

Authors: Andrés Holgado-Sánchez, Holger Billhardt, Sascha Ossowski, Sara Degli-Esposti

Aligning AI systems with human values and the value-based preferences of
various stakeholders (their value systems) is key in ethical AI. In value-aware
AI systems, decision-making draws upon explicit computational representations
of individual values (groundings) and their aggregation into value systems. As
these are notoriously difficult to elicit and calibrate manually, value
learning approaches aim to automatically derive computational models of an
agent's values and value system from demonstrations of human behaviour.
Nonetheless, social science and humanities literature suggest that it is more
adequate to conceive the value system of a society as a set of value systems of
different groups, rather than as the simple aggregation of individual value
systems. Accordingly, here we formalize the problem of learning the value
systems of societies and propose a method to address it based on heuristic deep
clustering. The method learns socially shared value groundings and a set of
diverse value systems representing a given society by observing qualitative
value-based preferences from a sample of agents. We evaluate the proposal in a
use case with real data about travelling decisions.

### 5. [FHSTP@EXIST 2025 Benchmark: Sexism Detection with Transparent Speech Concept Bottleneck Models](http://arxiv.org/pdf/2507.20924v1)

Authors: Roberto Labadie-Tamayo, Adrian Jaques Böck, Djordje Slijepčević, Xihui Chen, Andreas Babic, Matthias Zeppelzauer

Sexism has become widespread on social media and in online conversation. To
help address this issue, the fifth Sexism Identification in Social Networks
(EXIST) challenge is initiated at CLEF 2025. Among this year's international
benchmarks, we concentrate on solving the first task aiming to identify and
classify sexism in social media textual posts. In this paper, we describe our
solutions and report results for three subtasks: Subtask 1.1 - Sexism
Identification in Tweets, Subtask 1.2 - Source Intention in Tweets, and Subtask
1.3 - Sexism Categorization in Tweets. We implement three models to address
each subtask which constitute three individual runs: Speech Concept Bottleneck
Model (SCBM), Speech Concept Bottleneck Model with Transformer (SCBMT), and a
fine-tuned XLM-RoBERTa transformer model. SCBM uses descriptive adjectives as
human-interpretable bottleneck concepts. SCBM leverages large language models
(LLMs) to encode input texts into a human-interpretable representation of
adjectives, then used to train a lightweight classifier for downstream tasks.
SCBMT extends SCBM by fusing adjective-based representation with contextual
embeddings from transformers to balance interpretability and classification
performance. Beyond competitive results, these two models offer fine-grained
explanations at both instance (local) and class (global) levels. We also
investigate how additional metadata, e.g., annotators' demographic profiles,
can be leveraged. For Subtask 1.1, XLM-RoBERTa, fine-tuned on provided data
augmented with prior datasets, ranks 6th for English and Spanish and 4th for
English in the Soft-Soft evaluation. Our SCBMT achieves 7th for English and
Spanish and 6th for Spanish.

### 6. [VArsity: Can Large Language Models Keep Power Engineering Students in Phase?](http://arxiv.org/pdf/2507.20995v1)

Authors: Samuel Talkington, Daniel K. Molzahn

This paper provides an educational case study regarding our experience in
deploying ChatGPT Large Language Models (LLMs) in the Spring 2025 and Fall 2023
offerings of ECE 4320: Power System Analysis and Control at Georgia Tech. As
part of course assessments, students were tasked with identifying, explaining,
and correcting errors in the ChatGPT outputs corresponding to power factor
correction problems. While most students successfully identified the errors in
the outputs from the GPT-4 version of ChatGPT used in Fall 2023, students found
the errors from the ChatGPT o1 version much more difficult to identify in
Spring 2025. As shown in this case study, the role of LLMs in pedagogy,
assessment, and learning in power engineering classrooms is an important topic
deserving further investigation.

### Databases

### 1. [A Functional Data Model and Query Language is All You Need](http://arxiv.org/pdf/2507.20671v1)

Authors: Jens Dittrich

We propose the vision of a functional data model (FDM) and an associated
functional query language (FQL). Our proposal has far-reaching consequences: we
show a path to come up with a modern QL that solves (almost if not) all
problems of SQL (NULL-values, impedance mismatch, SQL injection, missing
querying capabilities for updates, etc.). FDM and FQL are much more expressive
than the relational model and SQL. In addition, in contrast to SQL, FQL
integrates smoothly into existing programming languages. In our approach both
QL and PL become the "same thing", thus opening up some interesting holistic
optimization opportunities between compilers and databases. In FQL, we also do
not need to force application developers to switch to unfamiliar programming
paradigms (like SQL or datalog): developers can stick with the abstractions
provided by their programming language.

### 2. [MVIAnalyzer: A Holistic Approach to Analyze Missing Value Imputation](http://arxiv.org/pdf/2507.20815v1)

Authors: Valerie Restat, Kai Tejkl, Uta Störl

Missing values often limit the usage of data analysis or cause falsification
of results. Therefore, methods of missing value imputation (MVI) are of great
significance. However, in general, there is no universal, fair MVI method for
different tasks. This work thus places MVI in the overall context of data
analysis. For this purpose, we present the MVIAnalyzer, a generic framework for
a holistic analysis of MVI. It considers the overall process up to the
application and analysis of machine learning methods. The associated software
is provided and can be used by other researchers for their own analyses. To
this end, it further includes a missing value simulation with consideration of
relevant parameters. The application of the MVIAnalyzer is demonstrated on data
with different characteristics. An evaluation of the results shows the
possibilities and limitations of different MVI methods. Since MVI is a very
complex topic with different influencing variables, this paper additionally
illustrates how the analysis can be supported by visualizations.

### 3. [Data Cleaning of Data Streams](http://arxiv.org/pdf/2507.20839v1)

Authors: Valerie Restat, Niklas Rodenhausen, Carina Antonin, Uta Störl

Streaming data can arise from a variety of contexts. Important use cases are
continuous sensor measurements such as temperature, light or radiation values.
In the process, streaming data may also contain data errors that should be
cleaned before further use. Many studies from science and practice focus on
data cleaning in a static context. However, in terms of data cleaning,
streaming data has particularities that distinguish it from static data. In
this paper, we have therefore undertaken an intensive exploration of data
cleaning of data streams. We provide a detailed analysis of the applicability
of data cleaning to data streams. Our theoretical considerations are evaluated
in comprehensive experiments. Using a prototype framework, we show that
cleaning is not consistent when working with data streams. An additional
contribution is the investigation of requirements for streaming technologies in
context of data cleaning.

### 4. [Search-Based Fuzzing For RESTful APIs That Use MongoDB](http://arxiv.org/pdf/2507.20848v1)

Authors: Hernan Ghianni, Man Zhang, Juan P. Galeotti, Andrea Arcuri

In RESTful APIs, interactions with a database are a common and crucial
aspect. When generating whitebox tests, it is essential to consider the
database's state (i.e., the data contained in the database) to achieve higher
code coverage and uncover more hidden faults. This article presents novel
techniques to enhance search-based software test generation for RESTful APIs
interacting with NoSQL databases. Specifically, we target the popular MongoDB
database, by dynamically analyzing (via automated code instrumentation) the
state of the database during the test generation process. Additionally, to
achieve better results, our novel approach allows inserting NoSQL data directly
from test cases. This is particularly beneficial when generating the correct
sequence of events to set the NoSQL database in an appropriate state is
challenging or time-consuming. This method is also advantageous for testing
read-only microservices. Our novel techniques are implemented as an extension
of EvoMaster, the only open-source tool for white-box fuzzing RESTful APIs.
Experiments conducted on six RESTful APIs demonstrated significant improvements
in code coverage, with increases of up to 18% compared to existing white-box
approaches. To better highlight the improvements of our novel techniques,
comparisons are also carried out with four state-of-the-art black-box fuzzers.

### Distributed, Parallel, and Cluster Computing

### 1. [RIMMS: Runtime Integrated Memory Management System for Heterogeneous Computing](http://arxiv.org/pdf/2507.20514v1)

Authors: Serhan Gener, Aditya Ukarande, Shilpa Mysore Srinivasa Murthy, Sahil Hassan, Joshua Mack, Chaitali Chakrabarti, Umit Ogras, Ali Akoglu

Efficient memory management in heterogeneous systems is increasingly
challenging due to diverse compute architectures (e.g., CPU, GPU, FPGA) and
dynamic task mappings not known at compile time. Existing approaches often
require programmers to manage data placement and transfers explicitly, or
assume static mappings that limit portability and scalability. This paper
introduces RIMMS (Runtime Integrated Memory Management System), a lightweight,
runtime-managed, hardware-agnostic memory abstraction layer that decouples
application development from low-level memory operations. RIMMS transparently
tracks data locations, manages consistency, and supports efficient memory
allocation across heterogeneous compute elements without requiring
platform-specific tuning or code modifications. We integrate RIMMS into a
baseline runtime and evaluate with complete radar signal processing
applications across CPU+GPU and CPU+FPGA platforms. RIMMS delivers up to 2.43X
speedup on GPU-based and 1.82X on FPGA-based systems over the baseline.
Compared to IRIS, a recent heterogeneous runtime system, RIMMS achieves up to
3.08X speedup and matches the performance of native CUDA implementations while
significantly reducing programming complexity. Despite operating at a higher
abstraction level, RIMMS incurs only 1-2 cycles of overhead per memory
management call, making it a low-cost solution. These results demonstrate
RIMMS's ability to deliver high performance and enhanced programmer
productivity in dynamic, real-world heterogeneous environments.

### 2. [Accelerating Deterministic Global Optimization via GPU-parallel Interval Arithmetic](http://arxiv.org/pdf/2507.20769v1)

Authors: Hongzhen Zhang, Tim Kerkenhoff, Neil Kichler, Manuel Dahmen, Alexander Mitsos, Uwe Naumann, Dominik Bongartz

Spatial Branch and Bound (B&B) algorithms are widely used for solving
nonconvex problems to global optimality, yet they remain computationally
expensive. Though some works have been carried out to speed up B&B via CPU
parallelization, GPU parallelization is much less explored. In this work, we
investigate the design of a spatial B&B algorithm that involves an
interval-based GPU-parallel lower bounding solver: The domain of each B&B node
is temporarily partitioned into numerous subdomains, then massive GPU
parallelism is leveraged to compute interval bounds of the objective function
and constraints on each subdomain, using the Mean Value Form. The resulting
bounds are tighter than those achieved via regular interval arithmetic without
partitioning, but they remain fast to compute. We implement the method into our
open-source solver MAiNGO via CUDA in two manners: wrapping all GPU tasks
within one kernel function, or distributing the GPU tasks onto a CUDA graph.
Numerical experiments show that using more subdomains leads to significantly
tighter lower bounds and thus less B&B iterations. Regarding wall clock time,
the proposed spatial B&B framework achieves a speedup of three orders of
magnitude compared to applying interval arithmetic on the CPU without domain
partitioning. Among the two implementations, the one developed with CUDA graph
enables higher efficiency. Moreover, in some case studies, the proposed method
delivers competitive or better performance compared to MAiNGO's default solver
which is based on McCormick relaxations. These results highlight the potential
of GPU-accelerated bounding techniques to accelerate B&B algorithms.

### Discrete Mathematics

### 1. [Ternary Binomial and Trinomial Bent Functions in the Completed Maiorana-McFarland Class](http://arxiv.org/pdf/2507.20715v1)

Authors: Tor Helleseth, Alexander Kholosha, Niki Spithaki

Two classes of ternary bent functions of degree four with two and three terms
in the univariate representation that belong to the completed
Maiorana-McFarland class are found. Binomials are mappings
$\F_{3^{4k}}\mapsto\fthree$ given by $f(x)=\Tr_{4k}\big(a_1 x^{2(3^k+1)}+a_2
x^{(3^k+1)^2}\big)$, where $a_1$ is a nonsquare in $\F_{3^{4k}}$ and $a_2$ is
defined explicitly by $a_1$. Particular subclasses of the binomial bent
functions we found can be represented by exceptional polynomials over
$\fthreek$. Bent trinomials are mappings $\F_{3^{2k}}\mapsto\fthree$ given by
$f(x)=\Tr_n\big(a_1 x^{2\cdot3^k+4} + a_2 x^{3^k+5} + a_3 x^2\big)$ with
coefficients explicitly defined by the parity of $k$. The proof is based on a
new criterion that allows checking bentness by analyzing first- and
second-order derivatives of $f$ in the direction of a chosen $n/2$-dimensional
subspace.

### 2. [General Strong Bound on the Uncrossed Number which is Tight for the Edge Crossing Number](http://arxiv.org/pdf/2507.20937v1)

Authors: Gaspard Charvy, Tomáš Masařík

We investigate a very recent concept for visualizing various aspects of a
graph in the plane using a collection of drawings introduced by Hlin\v{e}n\'y
and Masa\v{r}\'ik [GD 2023]. Formally, given a graph $G$, we aim to find an
uncrossed collection containing drawings of $G$ in the plane such that each
edge of $G$ is not crossed in at least one drawing in the collection. The
uncrossed number of $G$ ($unc(G)$) is the smallest integer $k$ such that an
uncrossed collection for $G$ of size $k$ exists. The uncrossed number is
lower-bounded by the well-known thickness, which is an edge-decomposition of
$G$ into planar graphs. This connection gives a trivial lower-bound
$\lceil\frac{|E(G)|}{3|V(G)|-6}\rceil \le unc(G)$. In a recent paper, Balko,
Hlin\v{e}n\'y, Masa\v{r}\'ik, Orthaber, Vogtenhuber, and Wagner [GD 2024]
presented the first non-trivial and general lower-bound on the uncrossed
number. We summarize it in terms of dense graphs (where
$|E(G)|=\epsilon(|V(G)|)^2$ for some $\epsilon>0$):
$\lceil\frac{|E(G)|}{c_\epsilon|V(G)|}\rceil \le unc(G)$, where $c_\epsilon\ge
2.82$ is a constant depending on $\epsilon$.
  We improve the lower-bound to state that
$\lceil\frac{|E(G)|}{3|V(G)|-6-\sqrt{2|E(G)|}+\sqrt{6(|V(G)|-2)}}\rceil \le
unc(G)$. Translated to dense graphs regime, the bound yields a multiplicative
constant $c'_\epsilon=3-\sqrt{(2-\epsilon)}$ in the expression
$\lceil\frac{|E(G)|}{c'_\epsilon|V(G)|+o(|V(G)|)}\rceil \le unc(G)$. Hence, it
is tight (up to low-order terms) for $\epsilon \approx \frac{1}{2}$ as
warranted by complete graphs.
  In fact, we formulate our result in the language of the maximum uncrossed
subgraph number, that is, the maximum number of edges of $G$ that are not
crossed in a drawing of $G$ in the plane. In that case, we also provide a
construction certifying that our bound is asymptotically tight (up to low-order
terms) on dense graphs for all $\epsilon>0$.

### Emerging Technologies

### 1. [Efficient Memristive Spiking Neural Networks Architecture with Supervised In-Situ STDP Method](http://arxiv.org/pdf/2507.20998v1)

Authors: Santlal Prajapati, Susmita Sur-Kolay, Soumyadeep Dutta

Memristor-based Spiking Neural Networks (SNNs) with temporal spike encoding
enable ultra-low-energy computation, making them ideal for battery-powered
intelligent devices. This paper presents a circuit-level memristive spiking
neural network (SNN) architecture trained using a proposed novel supervised
in-situ learning algorithm inspired by spike-timing-dependent plasticity
(STDP). The proposed architecture efficiently implements lateral inhibition and
the refractory period, eliminating the need for external microcontrollers or
ancillary control hardware. All synapses of the winning neurons are updated in
parallel, enhancing training efficiency. The modular design ensures scalability
with respect to input data dimensions and output class count. The SNN is
evaluated in LTspice for pattern recognition (using 5x3 binary images) and
classification tasks using the Iris and Breast Cancer Wisconsin (BCW) datasets.
During testing, the system achieved perfect pattern recognition and high
classification accuracies of 99.11\% (Iris) and 97.9\% (BCW). Additionally, it
has demonstrated robustness, maintaining an average recognition rate of 93.4\%
under 20\% input noise. The impact of stuck-at-conductance faults and memristor
device variations was also analyzed.

### 2. [Curved Apertures for Customized Wave Trajectories: Beyond Flat Aperture Limitations](http://arxiv.org/pdf/2507.20699v1)

Authors: Joan Martínez Canals, Francesco Devoti, Vincenzo Sciancalepore, Marco Di Renzo, Xavier Costa-Pérez

Beam shaping techniques enable tailored beam trajectories, offering
unprecedented connectivity opportunities in wireless communications. Current
approaches rely on flat apertures, which limit trajectory flexibility due to
inherent geometric constraints. To overcome such restrictions, we propose
adopting curved apertures as a more versatile alternative for beam shaping. We
introduce a novel formulation for wave trajectory engineering compatible with
arbitrarily shaped apertures. Theoretical and numerical analyses demonstrate
that curved apertures offer improved control over wave propagation, are more
resilient to phase control constraints, and achieve higher power density across
a wider portion of the desired beam trajectory than flat apertures.

### Graphics

### 1. [Endoscopic Depth Estimation Based on Deep Learning: A Survey](http://arxiv.org/pdf/2507.20881v1)

Authors: Ke Niu, Zeyun Liu, Xue Feng, Heng Li, Kaize Shi

Endoscopic depth estimation is a critical technology for improving the safety
and precision of minimally invasive surgery. It has attracted considerable
attention from researchers in medical imaging, computer vision, and robotics.
Over the past decade, a large number of methods have been developed. Despite
the existence of several related surveys, a comprehensive overview focusing on
recent deep learning-based techniques is still limited. This paper endeavors to
bridge this gap by systematically reviewing the state-of-the-art literature.
Specifically, we provide a thorough survey of the field from three key
perspectives: data, methods, and applications, covering a range of methods
including both monocular and stereo approaches. We describe common performance
evaluation metrics and summarize publicly available datasets. Furthermore, this
review analyzes the specific challenges of endoscopic scenes and categorizes
representative techniques based on their supervision strategies and network
architectures. The application of endoscopic depth estimation in the important
area of robot-assisted surgery is also reviewed. Finally, we outline potential
directions for future research, such as domain adaptation, real-time
implementation, and enhanced model generalization, thereby providing a valuable
starting point for researchers to engage with and advance the field.

### 2. [Methodology for intelligent injection point location based on geometric algorithms and discrete topologies for virtual digital twin environments](http://arxiv.org/pdf/2507.20922v1)

Authors: J. Mercado Colmenero, A. Torres Alba, C. Martin Donate

This article presents an innovative methodology for locating injection points
in injection-molded parts using intelligent models with geometric algorithms
for discrete topologies. The first algorithm calculates the center of mass of
the discrete model based on the center of mass of each triangular facet in the
system, ensuring uniform molten plastic distribution during mold cavity
filling. Two sub-algorithms intelligently evaluate the geometry and optimal
injection point location. The first sub-algorithm generates a geometric matrix
based on a two-dimensional nodal quadrature adapted to the part's bounding box.
The second sub-algorithm projects the nodal matrix and associated circular
areas orthogonally on the part's surface along the demolding direction. The
optimal injection point location is determined by minimizing the distance to
the center of mass from the first algorithm's result. This novel methodology
has been validated through rheological simulations in six case studies with
complex geometries. The results demonstrate uniform and homogeneous molten
plastic distribution with minimal pressure loss during the filling phase.
Importantly, this methodology does not require expert intervention, reducing
time and costs associated with manual injection mold feed system design. It is
also adaptable to various design environments and virtual twin systems, not
tied to specific CAD software. The validated results surpass the state of the
art, offering an agile alternative for digital twin applications in new product
design environments, reducing dependence on experts, facilitating designer
training, and ultimately cutting costs

### Computer Science and Game Theory

### 1. [Fairness under Equal-Sized Bundles: Impossibility Results and Approximation Guarantees](http://arxiv.org/pdf/2507.20899v1)

Authors: Alviona Mancho, Evangelos Markakis, Nicos Protopapas

We study the fair allocation of indivisible goods under cardinality
constraints, where each agent must receive a bundle of fixed size. This models
practical scenarios, such as assigning shifts or forming equally sized teams.
Recently, variants of envy-freeness up to one/any item (EF1, EFX) were
introduced for this setting, based on flips or exchanges of items. Namely, one
can define envy-freeness up to one/any flip (EFF1, EFFX), meaning that an agent
$i$ does not envy another agent $j$ after performing one or any one-item flip
between their bundles that improves the value of $i$.
  We explore algorithmic aspects of this notion, and our contribution is
twofold: we present both algorithmic and impossibility results, highlighting a
stark contrast between the classic EFX concept and its flip-based analogue.
First, we explore standard techniques used in the literature and show that they
fail to guarantee EFFX approximations. On the positive side, we show that we
can achieve a constant factor approximation guarantee when agents share a
common ranking over item values, based on the well-known envy cycle elimination
technique. This idea also leads to a generalized algorithm with approximation
guarantees when agents agree on the top $n$ items and their valuation functions
are bounded. Finally, we show that an algorithm that maximizes the Nash welfare
guarantees a 1/2-EFF1 allocation, and that this bound is tight.

### 2. [Behavioral Study of Dashboard Mechanisms](http://arxiv.org/pdf/2507.20985v1)

Authors: Paula Kayongo, Jessica Hullman, Jason Hartline

Visualization dashboards are increasingly used in strategic settings like
auctions to enhance decision-making and reduce strategic confusion. This paper
presents behavioral experiments evaluating how different dashboard designs
affect bid optimization in reverse first-price auctions. Additionally, we
assess how dashboard designs impact the auction designer's ability to
accurately infer bidders' preferences within the dashboard mechanism framework.
We compare visualizations of the bid allocation rule, commonly deployed in
practice, to alternatives that display expected utility. We find that
utility-based visualizations significantly improve bidding by reducing
cognitive demands on bidders. However, even with improved dashboards, bidders
systematically under-shade their bids, driven by an implicit preference for
certain wins in uncertain settings. As a result, dashboard-based mechanisms
that assume fully rational or risk-neutral bidder responses to dashboards can
produce significant estimation errors when inferring private preferences, which
may lead to suboptimal allocations in practice. Explicitly modeling agents'
behavioral responses to dashboards substantially improves inference accuracy,
highlighting the need to align visualization design and econometric inference
assumptions in practice.

### 3. [Core Safety Values for Provably Corrigible Agents](http://arxiv.org/pdf/2507.20964v1)

Authors: Aran Nayebi

We introduce the first implementable framework for corrigibility, with
provable guarantees in multi-step, partially observed environments. Our
framework replaces a single opaque reward with five *structurally separate*
utility heads -- deference, switch-access preservation, truthfulness,
low-impact behavior via a belief-based extension of Attainable Utility
Preservation, and bounded task reward -- combined lexicographically by strict
weight gaps. Theorem 1 proves exact single-round corrigibility in the partially
observable off-switch game; Theorem 3 extends the guarantee to multi-step,
self-spawning agents, showing that even if each head is \emph{learned} to
mean-squared error $\varepsilon$ and the planner is $\varepsilon$-sub-optimal,
the probability of violating \emph{any} safety property is bounded while still
ensuring net human benefit. In contrast to Constitutional AI or RLHF/RLAIF,
which merge all norms into one learned scalar, our separation makes obedience
and impact-limits dominate even when incentives conflict. For open-ended
settings where adversaries can modify the agent, we prove that deciding whether
an arbitrary post-hack agent will ever violate corrigibility is undecidable by
reduction to the halting problem, then carve out a finite-horizon ``decidable
island'' where safety can be certified in randomized polynomial time and
verified with privacy-preserving, constant-round zero-knowledge proofs.
Consequently, the remaining challenge is the ordinary ML task of data coverage
and generalization: reward-hacking risk is pushed into evaluation quality
rather than hidden incentive leak-through, giving clearer implementation
guidance for today's LLM assistants and future autonomous systems.

### Human-Computer Interaction

### 1. [CoGrader: Transforming Instructors' Assessment of Project Reports through Collaborative LLM Integration](http://arxiv.org/pdf/2507.20655v1)

Authors: Zixin Chen, Jiachen Wang, Yumeng Li, Haobo Li, Chuhan Shi, Rong Zhang, Huamin Qu

Grading project reports are increasingly significant in today's educational
landscape, where they serve as key assessments of students' comprehensive
problem-solving abilities. However, it remains challenging due to the
multifaceted evaluation criteria involved, such as creativity and
peer-comparative achievement. Meanwhile, instructors often struggle to maintain
fairness throughout the time-consuming grading process. Recent advances in AI,
particularly large language models, have demonstrated potential for automating
simpler grading tasks, such as assessing quizzes or basic writing quality.
However, these tools often fall short when it comes to complex metrics, like
design innovation and the practical application of knowledge, that require an
instructor's educational insights into the class situation. To address this
challenge, we conducted a formative study with six instructors and developed
CoGrader, which introduces a novel grading workflow combining human-LLM
collaborative metrics design, benchmarking, and AI-assisted feedback. CoGrader
was found effective in improving grading efficiency and consistency while
providing reliable peer-comparative feedback to students. We also discuss
design insights and ethical considerations for the development of human-AI
collaborative grading systems.

### 2. [EarXplore: An Open Research Database on Earable Interaction](http://arxiv.org/pdf/2507.20656v1)

Authors: Jonas Hummel, Tobias Röddiger, Valeria Zitz, Philipp Lepold, Michael Küttner, Marius Prill, Christopher Clarke, Hans Gellersen, Michael Beigl

Interaction with earables - earphones equipped with additional sensors - has
been identified as one of four major areas of earable research. Worn naturally
and positioned near key physiological signals, earables support a wide range of
interaction modalities and have demonstrated the ability to detect multiple
inputs simultaneously. Yet this diversity has resulted in a fragmented body of
research, making it increasingly difficult to track developments and identify
relevant studies. To address this, we introduce EarXplore, a curated,
interactive online database on earable interaction research. Designed through a
question-centered process that guided both the development of 34 criteria
applied to annotate 118 studies and the structure of the platform, EarXplore
comprises four distinct yet integrated views: a Tabular View for structured
exploration, a Graphical View for visual overviews, a Similarity View for
identifying conceptual links, and a Timeline View for analyzing trends and
scholarly lineage. We demonstrate how the platform supports tailored
exploration, targeted filtering, and interactive information retrieval,
allowing researchers to query the literature and synthesize information in the
format of their choice. We furthermore leverage the contents and capabilities
of the platform to discuss the research gaps and opportunities in the field.
With built-in mechanisms for continuous community updates, EarXplore not only
reflects the current state of the field but also evolves alongside it, serving
as a living resource to inform and accelerate future developments.

### 3. [Beyond Text: Probing K-12 Educators' Perspectives and Ideas for Learning Opportunities Leveraging Multimodal Large Language Models](http://arxiv.org/pdf/2507.20720v1)

Authors: Tiffany Tseng, Katelyn Lam, Tiffany Lin Fu, Alekhya Maram

Multimodal Large Language Models (MLLMs) are beginning to empower new user
experiences that can flexibly generate content from a range of inputs,
including images, text, speech, and video. These capabilities have the
potential to enrich learning by enabling users to capture and interact with
information using a variety of modalities, but little is known about how
educators envision how MLLMs might shape the future of learning experiences,
what challenges diverse teachers encounter when interpreting how these models
work, and what practical needs should be considered for successful
implementation in educational contexts. We investigated educator perspectives
through formative workshops with 12 K-12 educators, where participants
brainstormed learning opportunities, discussed practical concerns for effective
use, and prototyped their own MLLM-powered learning applications using Claude
3.5 and its Artifacts feature for previewing code-based output. We use case
studies to illustrate two contrasting end-user approaches (teacher-and
student-driven), and share insights about opportunities and concerns expressed
by our participants, ending with implications for leveraging MLLMs for future
learning experiences.

### 4. [Beyond QWERTY: A pressure-based text input approach for XR that enables a touch-typing like experience](http://arxiv.org/pdf/2507.20741v1)

Authors: Fabian Rücker, Torben Storch

Text input in extended reality (XR) applications remains inefficient and
tedious. Most solutions are derived from the traditional keyboard layout, yet
fail to translate its positive characteristics to the spatial digital realm.
This limits the productive use of immersive technologies. In this work, we
analyze physical keyboard input to identify key characteristics that facilitate
its comfort, touch typing and high typing speeds. Building on these findings,
we propose a novel pressure-based text input modality that transfers these
characteristics into immersive space by substituting the two-dimensional QWERTY
layout with a linear scale. This design facilitates a touch-typing-like
experience, eliminating the need for visual guidance for proficient users. Our
skill-based approach enables typing speeds of over 200 characters per minute.
Additionally, it is suitable for discreet use in public spaces and everyday
text-input tasks, since the proposed system requires virtually no hand or
finger movements and resembles smartphone-based text input in appearance.

### 5. [The Impact of Simple, Brief, and Adaptive Instructions within Virtual Reality Training: Components of Cognitive Load Theory in an Assembly Task](http://arxiv.org/pdf/2507.20943v1)

Authors: Rebecca L. Pharmer, Christopher D. Wickens, Lucas Plabst, Benjamin A. Clegg, Leanne M. Hirshfield, Joanna E. Lewis, Jalynn B. Nicoly, Cara A. Spencer, Francisco R. Ortega

Objective: The study examined the effects of varying all three core elements
of cognitive load on learning efficiency during a shape assembly task in
virtual reality (VR).
  Background: Adaptive training systems aim to improve learning efficiency and
retention by dynamically adjusting difficulty. However, design choices can
impact the cognitive workload imposed on the learner. The present experiments
examined how aspects of cognitive load impact training outcomes.
  Method: Participants learned step-by-step shape assembly in a VR environment.
Cognitive load was manipulated across three dimensions: Intrinsic Load (shape
complexity), Extraneous Load (instruction verbosity), and Germane Load
(adaptive vs. fixed training). In adaptive training (experiment 1), difficulty
increased based on individual performance. In fixed training (experiment 2),
difficulty followed a preset schedule from a yoked participant.
  Results: Higher Intrinsic Load significantly increased training times and
subjective workload but did not affect retention test accuracy. Extraneous Load
modestly impacted training time, with little impact on workload or retention.
Adaptive training shortened overall training time without increasing workload
or impairing retention. No interactions were observed between the three types
of load. Conclusion: Both Intrinsic and Extraneous Load increased training
time, but adaptive training improved efficiency without harming retention. The
lack of interaction between the elements suggests training benefits can be
worth seeking within any of the components of cognitive load. Application:
These findings support the use of VR adaptive systems in domains such as
manufacturing and military service, where efficient assembly skill acquisition
is critical. Tailoring difficulty in real-time can optimize efficiency without
compromising learning.

### 6. [Towards Effective Human Performance in XR Space Framework based on Real-time Eye Tracking Biofeedback](http://arxiv.org/pdf/2507.21000v1)

Authors: Barbara Karpowicz, Tomasz Kowalewski, Pavlo Zinevych, Adam Kuzdraliński, Grzegorz Marcin Wójcik, Wiesław Kopeć

This paper proposes an eye tracking module for the XR Space Framework aimed
at enhancing human performance in XR-based applications, specifically in
training, screening, and teleoperation. This framework provides a methodology
and components that streamline the development of adaptive real-time virtual
immersive systems. It contains multimodal measurements - declarative in the
form of in-VR questionnaires and objective, including eye tracking, body
movement, and psychophysiological data (e.g., ECG, GSR, PPG). A key focus of
this paper is the integration of real-time eye tracking data into XR
environments to facilitate a biofeedback loop, providing insight into user
attention, cognitive load, and engagement. Given the relatively high
measurement frequency of eye tracking - recognized as a noninvasive yet robust
psychophysiological measure - this technology is particularly well suited for
real-time adjustments in task difficulty and feedback to enhance learning and
operational effectiveness. Despite its established role in cognitive and
attentional studies, implementing eye tracking metrics within dynamic,
real-time XR environments poses unique challenges, particularly given the
complex moving visuals presented in head-mounted displays (HMDs). This paper
addresses these challenges by focusing on the essential aspects of integrating
eye tracking in immersive systems based on real-time engines, ultimately
facilitating more efficient, adaptive XR applications.

### 7. [User-Centered Design with AI in the Loop: A Case Study of Rapid User Interface Prototyping with "Vibe Coding"](http://arxiv.org/pdf/2507.21012v1)

Authors: Tianyi Li, Tanay Maheshwari, Alex Voelker

We present a case study of using generative user interfaces, or ``vibe
coding,'' a method leveraging large language models (LLMs) for generating code
via natural language prompts, to support rapid prototyping in user-centered
design (UCD). Extending traditional UCD practices, we propose an AI-in-the-loop
ideate-prototyping process. We share insights from an empirical experience
integrating this process to develop an interactive data analytics interface for
highway traffic engineers to effectively retrieve and analyze historical
traffic data. With generative UIs, the team was able to elicit rich user
feedback and test multiple alternative design ideas from user evaluation
interviews and real-time collaborative sessions with domain experts. We discuss
the advantages and pitfalls of vibe coding for bridging the gaps between design
expertise and domain-specific expertise.

### 8. [Self-Supervised Continuous Colormap Recovery from a 2D Scalar Field Visualization without a Legend](http://arxiv.org/pdf/2507.20632v1)

Authors: Hongxu Liu, Xinyu Chen, Haoyang Zheng, Manyi Li, Zhenfan Liu, Fumeng Yang, Yunhai Wang, Changhe Tu, Qiong Zeng

Recovering a continuous colormap from a single 2D scalar field visualization
can be quite challenging, especially in the absence of a corresponding color
legend. In this paper, we propose a novel colormap recovery approach that
extracts the colormap from a color-encoded 2D scalar field visualization by
simultaneously predicting the colormap and underlying data using a
decoupling-and-reconstruction strategy. Our approach first separates the input
visualization into colormap and data using a decoupling module, then
reconstructs the visualization with a differentiable color-mapping module. To
guide this process, we design a reconstruction loss between the input and
reconstructed visualizations, which serves both as a constraint to ensure
strong correlation between colormap and data during training, and as a
self-supervised optimizer for fine-tuning the predicted colormap of unseen
visualizations during inferencing. To ensure smoothness and correct color
ordering in the extracted colormap, we introduce a compact colormap
representation using cubic B-spline curves and an associated color order loss.
We evaluate our method quantitatively and qualitatively on a synthetic dataset
and a collection of real-world visualizations from the VIS30K dataset.
Additionally, we demonstrate its utility in two prototype applications --
colormap adjustment and colormap transfer -- and explore its generalization to
visualizations with color legends and ones encoded using discrete color
palettes.

### 9. [Vocalize: Lead Acquisition and User Engagement through Gamified Voice Competitions](http://arxiv.org/pdf/2507.20730v1)

Authors: Edvin Teskeredzic, Muamer Paric, Adna Sestic, Petra Fribert, Anamarija Lukac, Hadzem Hadzic, Kemal Altwlkany, Emanuel Lacic

This paper explores the prospect of creating engaging user experiences and
collecting leads through an interactive and gamified platform. We introduce
Vocalize, an end-to-end system for increasing user engagement and lead
acquisition through gamified voice competitions. Using audio processing
techniques and LLMs, we create engaging and interactive experiences that have
the potential to reach a wide audience, foster brand recognition, and increase
customer loyalty. We describe the system from a technical standpoint and report
results from launching Vocalize at 4 different live events. Our user study
shows that Vocalize is capable of generating significant user engagement, which
shows potential for gamified audio campaigns in marketing and similar
verticals.

### 10. [Understanding Bias in Perceiving Dimensionality Reduction Projections](http://arxiv.org/pdf/2507.20805v1)

Authors: Seoyoung Doh, Hyeon Jeon, Sungbok Shin, Ghulam Jilani Quadri, Nam Wook Kim, Jinwook Seo

Selecting the dimensionality reduction technique that faithfully represents
the structure is essential for reliable visual communication and analytics. In
reality, however, practitioners favor projections for other attractions, such
as aesthetics and visual saliency, over the projection's structural
faithfulness, a bias we define as visual interestingness. In this research, we
conduct a user study that (1) verifies the existence of such bias and (2)
explains why the bias exists. Our study suggests that visual interestingness
biases practitioners' preferences when selecting projections for analysis, and
this bias intensifies with color-encoded labels and shorter exposure time.
Based on our findings, we discuss strategies to mitigate bias in perceiving and
interpreting DR projections.

### Information Retrieval

### 1. [Watermarking Large Language Model-based Time Series Forecasting](http://arxiv.org/pdf/2507.20762v1)

Authors: Wei Yuan, Chaoqun Yang, Yu Xing, Tong Chen, Nguyen Quoc Viet Hung, Hongzhi Yin

Large Language Model-based Time Series Forecasting (LLMTS) has shown
remarkable promise in handling complex and diverse temporal data, representing
a significant step toward foundation models for time series analysis. However,
this emerging paradigm introduces two critical challenges. First, the
substantial commercial potential and resource-intensive development raise
urgent concerns about intellectual property (IP) protection. Second, their
powerful time series forecasting capabilities may be misused to produce
misleading or fabricated deepfake time series data. To address these concerns,
we explore watermarking the outputs of LLMTS models, that is, embedding
imperceptible signals into the generated time series data that remain
detectable by specialized algorithms. We propose a novel post-hoc watermarking
framework, Waltz, which is broadly compatible with existing LLMTS models. Waltz
is inspired by the empirical observation that time series patch embeddings are
rarely aligned with a specific set of LLM tokens, which we term ``cold
tokens''. Leveraging this insight, Waltz embeds watermarks by rewiring the
similarity statistics between patch embeddings and cold token embeddings, and
detects watermarks using similarity z-scores. To minimize potential side
effects, we introduce a similarity-based embedding position identification
strategy and employ projected gradient descent to constrain the watermark noise
within a defined boundary. Extensive experiments using two popular LLMTS models
across seven benchmark datasets demonstrate that Waltz achieves high watermark
detection accuracy with minimal impact on the quality of the generated time
series.

### 2. [Improving Community Detection in Academic Networks by Handling Publication Bias](http://arxiv.org/pdf/2507.20449v1)

Authors: Md Asaduzzaman Noor, John Sheppard, Jason Clark

Finding potential research collaborators is a challenging task, especially in
today's fast-growing and interdisciplinary research landscape. While
traditional methods often rely on observable relationships such as
co-authorships and citations to construct the research network, in this work,
we focus solely on publication content to build a topic-based research network
using BERTopic with a fine-tuned SciBERT model that connects and recommends
researchers across disciplines based on shared topical interests. A major
challenge we address is publication imbalance, where some researchers publish
much more than others, often across several topics. Without careful handling,
their less frequent interests are hidden under dominant topics, limiting the
network's ability to detect their full research scope. To tackle this, we
introduce a cloning strategy that clusters a researcher's publications and
treats each cluster as a separate node. This allows researchers to be part of
multiple communities, improving the detection of interdisciplinary links.
Evaluation on the proposed method shows that the cloned network structure leads
to more meaningful communities and uncovers a broader set of collaboration
opportunities.

### 3. [ZSE-Cap: A Zero-Shot Ensemble for Image Retrieval and Prompt-Guided Captioning](http://arxiv.org/pdf/2507.20564v1)

Authors: Duc-Tai Dinh, Duc Anh Khoa Dinh

We present ZSE-Cap (Zero-Shot Ensemble for Captioning), our 4th place system
in Event-Enriched Image Analysis (EVENTA) shared task on article-grounded image
retrieval and captioning. Our zero-shot approach requires no finetuning on the
competition's data. For retrieval, we ensemble similarity scores from CLIP,
SigLIP, and DINOv2. For captioning, we leverage a carefully engineered prompt
to guide the Gemma 3 model, enabling it to link high-level events from the
article to the visual content in the image. Our system achieved a final score
of 0.42002, securing a top-4 position on the private test set, demonstrating
the effectiveness of combining foundation models through ensembling and
prompting. Our code is available at https://github.com/ductai05/ZSE-Cap.

### 4. [Beyond Interactions: Node-Level Graph Generation for Knowledge-Free Augmentation in Recommender Systems](http://arxiv.org/pdf/2507.20578v1)

Authors: Zhaoyan Wang, Hyunjun Ahn, In-Young Ko

Recent advances in recommender systems rely on external resources such as
knowledge graphs or large language models to enhance recommendations, which
limit applicability in real-world settings due to data dependency and
computational overhead. Although knowledge-free models are able to bolster
recommendations by direct edge operations as well, the absence of augmentation
primitives drives them to fall short in bridging semantic and structural gaps
as high-quality paradigm substitutes. Unlike existing diffusion-based works
that remodel user-item interactions, this work proposes NodeDiffRec, a
pioneering knowledge-free augmentation framework that enables fine-grained
node-level graph generation for recommendations and expands the scope of
restricted augmentation primitives via diffusion. By synthesizing pseudo-items
and corresponding interactions that align with the underlying distribution for
injection, and further refining user preferences through a denoising preference
modeling process, NodeDiffRec dramatically enhances both semantic diversity and
structural connectivity without external knowledge. Extensive experiments
across diverse datasets and recommendation algorithms demonstrate the
superiority of NodeDiffRec, achieving State-of-the-Art (SOTA) performance, with
maximum average performance improvement 98.6% in Recall@5 and 84.0% in NDCG@5
over selected baselines.

### 5. [Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank](http://arxiv.org/pdf/2507.20753v1)

Authors: Yunus Lutz, Timo Wilm, Philipp Duwe

In e-commerce recommender and search systems, tree-based models, such as
LambdaMART, have set a strong baseline for Learning-to-Rank (LTR) tasks.
Despite their effectiveness and widespread adoption in industry, the debate
continues whether deep neural networks (DNNs) can outperform traditional
tree-based models in this domain. To contribute to this discussion, we
systematically benchmark DNNs against our production-grade LambdaMART model. We
evaluate multiple DNN architectures and loss functions on a proprietary dataset
from OTTO and validate our findings through an 8-week online A/B test. The
results show that a simple DNN architecture outperforms a strong tree-based
baseline in terms of total clicks and revenue, while achieving parity in total
units sold.

### 6. [Modeling User Behavior from Adaptive Surveys with Supplemental Context](http://arxiv.org/pdf/2507.20919v1)

Authors: Aman Shukla, Daniel Patrick Scantlebury, Rishabh Kumar

Modeling user behavior is critical across many industries where understanding
preferences, intent, or decisions informs personalization, targeting, and
strategic outcomes. Surveys have long served as a classical mechanism for
collecting such behavioral data due to their interpretability, structure, and
ease of deployment. However, surveys alone are inherently limited by user
fatigue, incomplete responses, and practical constraints on their length making
them insufficient for capturing user behavior. In this work, we present LANTERN
(Late-Attentive Network for Enriched Response Modeling), a modular architecture
for modeling user behavior by fusing adaptive survey responses with
supplemental contextual signals. We demonstrate the architectural value of
maintaining survey primacy through selective gating, residual connections and
late fusion via cross-attention, treating survey data as the primary signal
while incorporating external modalities only when relevant. LANTERN outperforms
strong survey-only baselines in multi-label prediction of survey responses. We
further investigate threshold sensitivity and the benefits of selective
modality reliance through ablation and rare/frequent attribute analysis.
LANTERN's modularity supports scalable integration of new encoders and evolving
datasets. This work provides a practical and extensible blueprint for behavior
modeling in survey-centric applications.

### Machine Learning

### 1. [Provable In-Context Learning of Nonlinear Regression with Transformers](http://arxiv.org/pdf/2507.20443v1)

Authors: Hongbo Li, Lingjie Duan, Yingbin Liang

The transformer architecture, which processes sequences of input tokens to
produce outputs for query tokens, has revolutionized numerous areas of machine
learning. A defining feature of transformers is their ability to perform
previously unseen tasks using task-specific prompts without updating
parameters, a phenomenon known as in-context learning (ICL). Recent research
has actively explored the training dynamics behind ICL, with much of the focus
on relatively simple tasks such as linear regression and binary classification.
To advance the theoretical understanding of ICL, this paper investigates more
complex nonlinear regression tasks, aiming to uncover how transformers acquire
in-context learning capabilities in these settings. We analyze the stage-wise
dynamics of attention during training: attention scores between a query token
and its target features grow rapidly in the early phase, then gradually
converge to one, while attention to irrelevant features decays more slowly and
exhibits oscillatory behavior. Our analysis introduces new proof techniques
that explicitly characterize how the nature of general non-degenerate
L-Lipschitz task functions affects attention weights. Specifically, we identify
that the Lipschitz constant L of nonlinear function classes as a key factor
governing the convergence dynamics of transformers in ICL. Leveraging these
insights, for two distinct regimes depending on whether L is below or above a
threshold, we derive different time bounds to guarantee near-zero prediction
error. Notably, despite the convergence time depending on the underlying task
functions, we prove that query tokens consistently attend to prompt tokens with
highly relevant features at convergence, demonstrating the ICL capability of
transformers for unseen functions.

### 2. [BOASF: A Unified Framework for Speeding up Automatic Machine Learning via Adaptive Successive Filtering](http://arxiv.org/pdf/2507.20446v1)

Authors: Guanghui Zhu, Xin Fang, Lei Wang, Wenzhong Chen, Rong Gu, Chunfeng Yuan, Yihua Huang

Machine learning has been making great success in many application areas.
However, for the non-expert practitioners, it is always very challenging to
address a machine learning task successfully and efficiently. Finding the
optimal machine learning model or the hyperparameter combination set from a
large number of possible alternatives usually requires considerable expert
knowledge and experience. To tackle this problem, we propose a combined
Bayesian Optimization and Adaptive Successive Filtering algorithm (BOASF) under
a unified multi-armed bandit framework to automate the model selection or the
hyperparameter optimization. Specifically, BOASF consists of multiple
evaluation rounds in each of which we select promising configurations for each
arm using the Bayesian optimization. Then, ASF can early discard the
poor-performed arms adaptively using a Gaussian UCB-based probabilistic model.
Furthermore, a Softmax model is employed to adaptively allocate available
resources for each promising arm that advances to the next round. The arm with
a higher probability of advancing will be allocated more resources.
Experimental results show that BOASF is effective for speeding up the model
selection and hyperparameter optimization processes while achieving robust and
better prediction performance than the existing state-of-the-art automatic
machine learning methods. Moreover, BOASF achieves better anytime performance
under various time budgets.

### 3. [Your Attention Matters: to Improve Model Robustness to Noise and Spurious Correlations](http://arxiv.org/pdf/2507.20453v1)

Authors: Camilo Tamayo-Rousseau, Yunjia Zhao, Yiqun Zhang, Randall Balestriero

Self-attention mechanisms are foundational to Transformer architectures,
supporting their impressive success in a wide range of tasks. While there are
many self-attention variants, their robustness to noise and spurious
correlations has not been well studied. This study evaluates Softmax, Sigmoid,
Linear, Doubly Stochastic, and Cosine attention within Vision Transformers
under different data corruption scenarios. Through testing across the CIFAR-10,
CIFAR-100, and Imagenette datasets, we show that Doubly Stochastic attention is
the most robust. Our findings inform self-attention selection in contexts with
imperfect data.

### 4. [Conditional Diffusion Models for Global Precipitation Map Inpainting](http://arxiv.org/pdf/2507.20478v1)

Authors: Daiko Kishikawa, Yuka Muto, Shunji Kotsuki

Incomplete satellite-based precipitation presents a significant challenge in
global monitoring. For example, the Global Satellite Mapping of Precipitation
(GSMaP) from JAXA suffers from substantial missing regions due to the orbital
characteristics of satellites that have microwave sensors, and its current
interpolation methods often result in spatial discontinuities. In this study,
we formulate the completion of the precipitation map as a video inpainting task
and propose a machine learning approach based on conditional diffusion models.
Our method employs a 3D U-Net with a 3D condition encoder to reconstruct
complete precipitation maps by leveraging spatio-temporal information from
infrared images, latitude-longitude grids, and physical time inputs. Training
was carried out on ERA5 hourly precipitation data from 2020 to 2023. We
generated a pseudo-GSMaP dataset by randomly applying GSMaP masks to ERA maps.
Performance was evaluated for the calendar year 2024, and our approach produces
more spatio-temporally consistent inpainted precipitation maps compared to
conventional methods. These results indicate the potential to improve global
precipitation monitoring using the conditional diffusion models.

### 5. [HIAL: A New Paradigm for Hypergraph Active Learning via Influence Maximization](http://arxiv.org/pdf/2507.20490v1)

Authors: Yanheng Hou, Xunkai Li, Zhenjun Li, Bing Zhou, Ronghua Li, Guoren Wang

In recent years, Hypergraph Neural Networks (HNNs) have demonstrated immense
potential in handling complex systems with high-order interactions. However,
acquiring large-scale, high-quality labeled data for these models is costly,
making Active Learning (AL) a critical technique. Existing Graph Active
Learning (GAL) methods, when applied to hypergraphs, often rely on techniques
like "clique expansion," which destroys the high-order structural information
crucial to a hypergraph's success, thereby leading to suboptimal performance.
To address this challenge, we introduce HIAL (Hypergraph Active Learning), a
native active learning framework designed specifically for hypergraphs. We
innovatively reformulate the Hypergraph Active Learning (HAL) problem as an
Influence Maximization task. The core of HIAL is a dual-perspective influence
function that, based on our novel "High-Order Interaction-Aware (HOI-Aware)"
propagation mechanism, synergistically evaluates a node's feature-space
coverage (via Magnitude of Influence, MoI) and its topological influence (via
Expected Diffusion Value, EDV). We prove that this objective function is
monotone and submodular, thus enabling the use of an efficient greedy algorithm
with a formal (1-1/e) approximation guarantee. Extensive experiments on seven
public datasets demonstrate that HIAL significantly outperforms
state-of-the-art baselines in terms of performance, efficiency, generality, and
robustness, establishing an efficient and powerful new paradigm for active
learning on hypergraphs.

### 6. [Mixture of Length and Pruning Experts for Knowledge Graphs Reasoning](http://arxiv.org/pdf/2507.20498v1)

Authors: Enjun Du, Siyi Liu, Yongqi Zhang

Knowledge Graph (KG) reasoning, which aims to infer new facts from structured
knowledge repositories, plays a vital role in Natural Language Processing (NLP)
systems. Its effectiveness critically depends on constructing informative and
contextually relevant reasoning paths. However, existing graph neural networks
(GNNs) often adopt rigid, query-agnostic path-exploration strategies, limiting
their ability to adapt to diverse linguistic contexts and semantic nuances. To
address these limitations, we propose \textbf{MoKGR}, a mixture-of-experts
framework that personalizes path exploration through two complementary
components: (1) a mixture of length experts that adaptively selects and weights
candidate path lengths according to query complexity, providing query-specific
reasoning depth; and (2) a mixture of pruning experts that evaluates candidate
paths from a complementary perspective, retaining the most informative paths
for each query. Through comprehensive experiments on diverse benchmark, MoKGR
demonstrates superior performance in both transductive and inductive settings,
validating the effectiveness of personalized path exploration in KGs reasoning.

### 7. [Attributed Graph Clustering with Multi-Scale Weight-Based Pairwise Coarsening and Contrastive Learning](http://arxiv.org/pdf/2507.20505v1)

Authors: Binxiong Li, Yuefei Wang, Binyu Zhao, Heyang Gao, Benhan Yang, Quanzhou Luo, Xue Li, Xu Xiang, Yujie Liu, Huijie Tang

This study introduces the Multi-Scale Weight-Based Pairwise Coarsening and
Contrastive Learning (MPCCL) model, a novel approach for attributed graph
clustering that effectively bridges critical gaps in existing methods,
including long-range dependency, feature collapse, and information loss.
Traditional methods often struggle to capture high-order graph features due to
their reliance on low-order attribute information, while contrastive learning
techniques face limitations in feature diversity by overemphasizing local
neighborhood structures. Similarly, conventional graph coarsening methods,
though reducing graph scale, frequently lose fine-grained structural details.
MPCCL addresses these challenges through an innovative multi-scale coarsening
strategy, which progressively condenses the graph while prioritizing the
merging of key edges based on global node similarity to preserve essential
structural information. It further introduces a one-to-many contrastive
learning paradigm, integrating node embeddings with augmented graph views and
cluster centroids to enhance feature diversity, while mitigating feature
masking issues caused by the accumulation of high-frequency node weights during
multi-scale coarsening. By incorporating a graph reconstruction loss and KL
divergence into its self-supervised learning framework, MPCCL ensures
cross-scale consistency of node representations. Experimental evaluations
reveal that MPCCL achieves a significant improvement in clustering performance,
including a remarkable 15.24% increase in NMI on the ACM dataset and notable
robust gains on smaller-scale datasets such as Citeseer, Cora and DBLP.

### 8. [Efficient Proxy Raytracer for Optical Systems using Implicit Neural Representations](http://arxiv.org/pdf/2507.20513v1)

Authors: Shiva Sinaei, Chuanjun Zheng, Kaan Akşit, Daisuke Iwai

Ray tracing is a widely used technique for modeling optical systems,
involving sequential surface-by-surface computations, which can be
computationally intensive. We propose Ray2Ray, a novel method that leverages
implicit neural representations to model optical systems with greater
efficiency, eliminating the need for surface-by-surface computations in a
single pass end-to-end model. Ray2Ray learns the mapping between rays emitted
from a given source and their corresponding rays after passing through a given
optical system in a physically accurate manner. We train Ray2Ray on nine
off-the-shelf optical systems, achieving positional errors on the order of
1{\mu}m and angular deviations on the order 0.01 degrees in the estimated
output rays. Our work highlights the potential of neural representations as a
proxy for optical raytracer.

### 9. [Kernel Learning for Sample Constrained Black-Box Optimization](http://arxiv.org/pdf/2507.20533v1)

Authors: Rajalaxmi Rajagopalan, Yu-Lin Wei, Romit Roy Choudhury

Black box optimization (BBO) focuses on optimizing unknown functions in
high-dimensional spaces. In many applications, sampling the unknown function is
expensive, imposing a tight sample budget. Ongoing work is making progress on
reducing the sample budget by learning the shape/structure of the function,
known as kernel learning. We propose a new method to learn the kernel of a
Gaussian Process. Our idea is to create a continuous kernel space in the latent
space of a variational autoencoder, and run an auxiliary optimization to
identify the best kernel. Results show that the proposed method, Kernel
Optimized Blackbox Optimization (KOBO), outperforms state of the art by
estimating the optimal at considerably lower sample budgets. Results hold not
only across synthetic benchmark functions but also in real applications. We
show that a hearing aid may be personalized with fewer audio queries to the
user, or a generative model could converge to desirable images from limited
user ratings.

### 10. [Reminiscence Attack on Residuals: Exploiting Approximate Machine Unlearning for Privacy](http://arxiv.org/pdf/2507.20573v1)

Authors: Yaxin Xiao, Qingqing Ye, Li Hu, Huadi Zheng, Haibo Hu, Zi Liang, Haoyang Li, Yijie Jiao

Machine unlearning enables the removal of specific data from ML models to
uphold the right to be forgotten. While approximate unlearning algorithms offer
efficient alternatives to full retraining, this work reveals that they fail to
adequately protect the privacy of unlearned data. In particular, these
algorithms introduce implicit residuals which facilitate privacy attacks
targeting at unlearned data. We observe that these residuals persist regardless
of model architectures, parameters, and unlearning algorithms, exposing a new
attack surface beyond conventional output-based leakage. Based on this insight,
we propose the Reminiscence Attack (ReA), which amplifies the correlation
between residuals and membership privacy through targeted fine-tuning
processes. ReA achieves up to 1.90x and 1.12x higher accuracy than prior
attacks when inferring class-wise and sample-wise membership, respectively. To
mitigate such residual-induced privacy risk, we develop a dual-phase
approximate unlearning framework that first eliminates deep-layer unlearned
data traces and then enforces convergence stability to prevent models from
"pseudo-convergence", where their outputs are similar to retrained models but
still preserve unlearned residuals. Our framework works for both classification
and generation tasks. Experimental evaluations confirm that our approach
maintains high unlearning efficacy, while reducing the adaptive privacy attack
accuracy to nearly random guess, at the computational cost of 2-12% of full
retraining from scratch.

### Neural and Evolutionary Computing

### 1. [Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](http://arxiv.org/pdf/2507.20923v1)

Authors: Minh Hieu Ha, Hung Phan, Tung Duy Doan, Tung Dao, Dao Tran, Huynh Thi Thanh Binh

Multi-objective combinatorial optimization problems (MOCOP) frequently arise
in practical applications that require the simultaneous optimization of
conflicting objectives. Although traditional evolutionary algorithms can be
effective, they typically depend on domain knowledge and repeated parameter
tuning, limiting flexibility when applied to unseen MOCOP instances. Recently,
integration of Large Language Models (LLMs) into evolutionary computation has
opened new avenues for automatic heuristic generation, using their advanced
language understanding and code synthesis capabilities. Nevertheless, most
existing approaches predominantly focus on single-objective tasks, often
neglecting key considerations such as runtime efficiency and heuristic
diversity in multi-objective settings. To bridge this gap, we introduce
Multi-heuristics for MOCOP via Pareto-Grid-guided Evolution of LLMs (MPaGE), a
novel enhancement of the Simple Evolutionary Multiobjective Optimization (SEMO)
framework that leverages LLMs and Pareto Front Grid (PFG) technique. By
partitioning the objective space into grids and retaining top-performing
candidates to guide heuristic generation, MPaGE utilizes LLMs to prioritize
heuristics with semantically distinct logical structures during variation, thus
promoting diversity and mitigating redundancy within the population. Through
extensive evaluations, MPaGE demonstrates superior performance over existing
LLM-based frameworks, and achieves competitive results to traditional
Multi-objective evolutionary algorithms (MOEAs), with significantly faster
runtime. Our code is available at: https://github.com/langkhachhoha/MPaGE.

### 2. [Efficient Memristive Spiking Neural Networks Architecture with Supervised In-Situ STDP Method](http://arxiv.org/pdf/2507.20998v1)

Authors: Santlal Prajapati, Susmita Sur-Kolay, Soumyadeep Dutta

Memristor-based Spiking Neural Networks (SNNs) with temporal spike encoding
enable ultra-low-energy computation, making them ideal for battery-powered
intelligent devices. This paper presents a circuit-level memristive spiking
neural network (SNN) architecture trained using a proposed novel supervised
in-situ learning algorithm inspired by spike-timing-dependent plasticity
(STDP). The proposed architecture efficiently implements lateral inhibition and
the refractory period, eliminating the need for external microcontrollers or
ancillary control hardware. All synapses of the winning neurons are updated in
parallel, enhancing training efficiency. The modular design ensures scalability
with respect to input data dimensions and output class count. The SNN is
evaluated in LTspice for pattern recognition (using 5x3 binary images) and
classification tasks using the Iris and Breast Cancer Wisconsin (BCW) datasets.
During testing, the system achieved perfect pattern recognition and high
classification accuracies of 99.11\% (Iris) and 97.9\% (BCW). Additionally, it
has demonstrated robustness, maintaining an average recognition rate of 93.4\%
under 20\% input noise. The impact of stuck-at-conductance faults and memristor
device variations was also analyzed.

### 3. [AR-LIF: Adaptive reset leaky-integrate and fire neuron for spiking neural networks](http://arxiv.org/pdf/2507.20746v1)

Authors: Zeyu Huang, Wei Meng, Quan Liu, Kun Chen, Li Ma

Spiking neural networks possess the advantage of low energy consumption due
to their event-driven nature. Compared with binary spike outputs, their
inherent floating-point dynamics are more worthy of attention. The threshold
level and reset mode of neurons play a crucial role in determining the number
and timing of spikes. The existing hard reset method causes information loss,
while the improved soft reset method adopts a uniform treatment for neurons. In
response to this, this paper designs an adaptive reset neuron, establishing the
correlation between input, output and reset, and integrating a simple yet
effective threshold adjustment strategy. It achieves excellent performance on
various datasets while maintaining the advantage of low energy consumption.

### 4. [Why Flow Matching is Particle Swarm Optimization?](http://arxiv.org/pdf/2507.20810v1)

Authors: Kaichen Ouyang

This paper preliminarily investigates the duality between flow matching in
generative models and particle swarm optimization (PSO) in evolutionary
computation. Through theoretical analysis, we reveal the intrinsic connections
between these two approaches in terms of their mathematical formulations and
optimization mechanisms: the vector field learning in flow matching shares
similar mathematical expressions with the velocity update rules in PSO; both
methods follow the fundamental framework of progressive evolution from initial
to target distributions; and both can be formulated as dynamical systems
governed by ordinary differential equations. Our study demonstrates that flow
matching can be viewed as a continuous generalization of PSO, while PSO
provides a discrete implementation of swarm intelligence principles. This
duality understanding establishes a theoretical foundation for developing novel
hybrid algorithms and creates a unified framework for analyzing both methods.
Although this paper only presents preliminary discussions, the revealed
correspondences suggest several promising research directions, including
improving swarm intelligence algorithms based on flow matching principles and
enhancing generative models using swarm intelligence concepts.

### Networking and Internet Architecture

### 1. [DD-JSCC: Dynamic Deep Joint Source-Channel Coding for Semantic Communications](http://arxiv.org/pdf/2507.20467v1)

Authors: Avi Deb Raha, Apurba Adhikary, Mrityunjoy Gain, Yumin Park, Walid Saad, Choong Seon Hong

Deep Joint Source-Channel Coding (Deep-JSCC) has emerged as a promising
semantic communication approach for wireless image transmission by jointly
optimizing source and channel coding using deep learning techniques. However,
traditional Deep-JSCC architectures employ fixed encoder-decoder structures,
limiting their adaptability to varying device capabilities, real-time
performance optimization, power constraints and channel conditions. To address
these limitations, we propose DD-JSCC: Dynamic Deep Joint Source-Channel Coding
for Semantic Communications, a novel encoder-decoder architecture designed for
semantic communication systems. Unlike traditional Deep-JSCC models, DD-JSCC is
flexible for dynamically adjusting its layer structures in real-time based on
transmitter and receiver capabilities, power constraints, compression ratios,
and current channel conditions. This adaptability is achieved through a
hierarchical layer activation mechanism combined with implicit regularization
via sequential randomized training, effectively reducing combinatorial
complexity, preventing overfitting, and ensuring consistent feature
representations across varying configurations. Simulation results demonstrate
that DD-JSCC enhances the performance of image reconstruction in semantic
communications, achieving up to 2 dB improvement in Peak Signal-to-Noise Ratio
(PSNR) over fixed Deep-JSCC architectures, while reducing training costs by
over 40%. The proposed unified framework eliminates the need for multiple
specialized models, significantly reducing training complexity and deployment
overhead.

### 2. [A Lyapunov-Guided Diffusion-Based Reinforcement Learning Approach for UAV-Assisted Vehicular Networks with Delayed CSI Feedback](http://arxiv.org/pdf/2507.20524v1)

Authors: Zhang Liu, Lianfen Huang, Zhibin Gao, Xianbin Wang, Dusit Niyato, Xuemin, Shen

Low altitude uncrewed aerial vehicles (UAVs) are expected to facilitate the
development of aerial-ground integrated intelligent transportation systems and
unlocking the potential of the emerging low-altitude economy. However, several
critical challenges persist, including the dynamic optimization of network
resources and UAV trajectories, limited UAV endurance, and imperfect channel
state information (CSI). In this paper, we offer new insights into low-altitude
economy networking by exploring intelligent UAV-assisted vehicle-to-everything
communication strategies aligned with UAV energy efficiency. Particularly, we
formulate an optimization problem of joint channel allocation, power control,
and flight altitude adjustment in UAV-assisted vehicular networks. Taking CSI
feedback delay into account, our objective is to maximize the vehicle-to-UAV
communication sum rate while satisfying the UAV's long-term energy constraint.
To this end, we first leverage Lyapunov optimization to decompose the original
long-term problem into a series of per-slot deterministic subproblems. We then
propose a diffusion-based deep deterministic policy gradient (D3PG) algorithm,
which innovatively integrates diffusion models to determine optimal channel
allocation, power control, and flight altitude adjustment decisions. Through
extensive simulations using real-world vehicle mobility traces, we demonstrate
the superior performance of the proposed D3PG algorithm compared to existing
benchmark solutions.

### 3. [Towards a Robust Transport Network With Self-adaptive Network Digital Twin](http://arxiv.org/pdf/2507.20971v1)

Authors: Cláudio Modesto, João Borges, Cleverson Nahum, Lucas Matni, Cristiano Bonato Both, Kleber Cardoso, Glauco Gonçalves, Ilan Correa, Silvia Lins, Andrey Silva, Aldebaro Klautau

The ability of the network digital twin (NDT) to remain aware of changes in
its physical counterpart, known as the physical twin (PTwin), is a fundamental
condition to enable timely synchronization, also referred to as twinning. In
this way, considering a transport network, a key requirement is to handle
unexpected traffic variability and dynamically adapt to maintain optimal
performance in the associated virtual model, known as the virtual twin (VTwin).
In this context, we propose a self-adaptive implementation of a novel NDT
architecture designed to provide accurate delay predictions, even under
fluctuating traffic conditions. This architecture addresses an essential
challenge, underexplored in the literature: improving the resilience of
data-driven NDT platforms against traffic variability and improving
synchronization between the VTwin and its physical counterpart. Therefore, the
contributions of this article rely on NDT lifecycle by focusing on the
operational phase, where telemetry modules are used to monitor incoming
traffic, and concept drift detection techniques guide retraining decisions
aimed at updating and redeploying the VTwin when necessary. We validate our
architecture with a network management use case, across various emulated
network topologies, and diverse traffic patterns to demonstrate its
effectiveness in preserving acceptable performance and predicting per-flow
delay under unexpected traffic variation. The results in all tested topologies,
using the normalized mean square error as the evaluation metric, demonstrate
that our proposed architecture, after a traffic concept drift, achieves a
performance improvement in prediction of at least 56.7% compared to a
configuration without NDT synchronization.

### 4. [Collusion Resistant DNS With Private Information Retrieval](http://arxiv.org/pdf/2507.20806v1)

Authors: Yunming Xiao, Peizhi Liu, Ruijie Yu, Chenkai Weng, Matteo Varvello, Aleksandar Kuzmanovic

There has been a growing interest in Internet user privacy, demonstrated by
the popularity of privacy-preserving products such as Telegram and Brave, and
the widespread adoption of HTTPS. The Domain Name System (DNS) is a key
component of Internet-based communication and its privacy has been neglected
for years. Recently, DNS over HTTPS (DoH) has improved the situation by fixing
the issue of in-path middleboxes. Further progress has been made with
proxy-based solutions such as Oblivious DoH (ODoH), which separate a user's
identity from their DNS queries. However, these solutions rely on non-collusion
assumptions between DNS resolvers and proxies -- an assumption difficult to
guarantee in practice. To address this, we explore integrating single-server
Private Information Retrieval (PIR) into DNS to enable encrypted query
processing without relying on trust assumptions. However, applying PIR to DNS
is challenging due to its hierarchical nature -- particularly, interactions
with recursive resolvers can still leak information. Navigating performance and
privacy trade-offs, we propose PDNS, a DNS extension leveraging single-server
PIR to strengthen privacy guarantees. We have implemented a prototype of PDNS
and compared its performance against state-of-the-art solutions via
trace-driven experiments. The results show that PDNS achieves acceptable
performance (2x faster than DoH over Tor with similar privacy guarantees) and
strong privacy guarantees today, mainly at the cost of its scalability, which
specialized hardware for PIR can address in the near future.

### 5. [\textit{FedABC}: Attention-Based Client Selection for Federated Learning with Long-Term View](http://arxiv.org/pdf/2507.20871v1)

Authors: Wenxuan Ye, Xueli An, Junfan Wang, Xueqiang Yan, Georg Carle

Native AI support is a key objective in the evolution of 6G networks, with
Federated Learning (FL) emerging as a promising paradigm. FL allows
decentralized clients to collaboratively train an AI model without directly
sharing their data, preserving privacy. Clients train local models on private
data and share model updates, which a central server aggregates to refine the
global model and redistribute it for the next iteration. However, client data
heterogeneity slows convergence and reduces model accuracy, and frequent client
participation imposes communication and computational burdens. To address these
challenges, we propose \textit{FedABC}, an innovative client selection
algorithm designed to take a long-term view in managing data heterogeneity and
optimizing client participation. Inspired by attention mechanisms,
\textit{FedABC} prioritizes informative clients by evaluating both model
similarity and each model's unique contributions to the global model. Moreover,
considering the evolving demands of the global model, we formulate an
optimization problem to guide \textit{FedABC} throughout the training process.
Following the ``later-is-better" principle, \textit{FedABC} adaptively adjusts
the client selection threshold, encouraging greater participation in later
training stages. Extensive simulations on CIFAR-10 demonstrate that
\textit{FedABC} significantly outperforms existing approaches in model accuracy
and client participation efficiency, achieving comparable performance with 32\%
fewer clients than the classical FL algorithm \textit{FedAvg}, and 3.5\% higher
accuracy with 2\% fewer clients than the state-of-the-art. This work marks a
step toward deploying FL in heterogeneous, resource-constrained environments,
thereby supporting native AI capabilities in 6G networks.

### 6. [Curved Apertures for Customized Wave Trajectories: Beyond Flat Aperture Limitations](http://arxiv.org/pdf/2507.20699v1)

Authors: Joan Martínez Canals, Francesco Devoti, Vincenzo Sciancalepore, Marco Di Renzo, Xavier Costa-Pérez

Beam shaping techniques enable tailored beam trajectories, offering
unprecedented connectivity opportunities in wireless communications. Current
approaches rely on flat apertures, which limit trajectory flexibility due to
inherent geometric constraints. To overcome such restrictions, we propose
adopting curved apertures as a more versatile alternative for beam shaping. We
introduce a novel formulation for wave trajectory engineering compatible with
arbitrarily shaped apertures. Theoretical and numerical analyses demonstrate
that curved apertures offer improved control over wave propagation, are more
resilient to phase control constraints, and achieve higher power density across
a wider portion of the desired beam trajectory than flat apertures.

### 7. [Handoff Design in User-Centric Cell-Free Massive MIMO Networks Using DRL](http://arxiv.org/pdf/2507.20966v1)

Authors: Hussein A. Ammar, Raviraj Adve, Shahram Shahbazpanahi, Gary Boudreau, Israfil Bahceci

In the user-centric cell-free massive MIMO (UC-mMIMO) network scheme, user
mobility necessitates updating the set of serving access points to maintain the
user-centric clustering. Such updates are typically performed through handoff
(HO) operations; however, frequent HOs lead to overheads associated with the
allocation and release of resources. This paper presents a deep reinforcement
learning (DRL)-based solution to predict and manage these connections for
mobile users. Our solution employs the Soft Actor-Critic algorithm, with
continuous action space representation, to train a deep neural network to serve
as the HO policy. We present a novel proposition for a reward function that
integrates a HO penalty in order to balance the attainable rate and the
associated overhead related to HOs. We develop two variants of our system; the
first one uses mobility direction-assisted (DA) observations that are based on
the user movement pattern, while the second one uses history-assisted (HA)
observations that are based on the history of the large-scale fading (LSF).
Simulation results show that our DRL-based continuous action space approach is
more scalable than discrete space counterpart, and that our derived HO policy
automatically learns to gather HOs in specific time slots to minimize the
overhead of initiating HOs. Our solution can also operate in real time with a
response time less than 0.4 ms.

### Robotics

### 1. [Learning Physical Interaction Skills from Human Demonstrations](http://arxiv.org/pdf/2507.20445v1)

Authors: Tianyu Li, Hengbo Ma, Sehoon Ha, Kwonjoon Lee

Learning physical interaction skills, such as dancing, handshaking, or
sparring, remains a fundamental challenge for agents operating in human
environments, particularly when the agent's morphology differs significantly
from that of the demonstrator. Existing approaches often rely on handcrafted
objectives or morphological similarity, limiting their capacity for
generalization. Here, we introduce a framework that enables agents with diverse
embodiments to learn wholebbody interaction behaviors directly from human
demonstrations. The framework extracts a compact, transferable representation
of interaction dynamics, called the Embedded Interaction Graph (EIG), which
captures key spatiotemporal relationships between the interacting agents. This
graph is then used as an imitation objective to train control policies in
physics-based simulations, allowing the agent to generate motions that are both
semantically meaningful and physically feasible. We demonstrate BuddyImitation
on multiple agents, such as humans, quadrupedal robots with manipulators, or
mobile manipulators and various interaction scenarios, including sparring,
handshaking, rock-paper-scissors, or dancing. Our results demonstrate a
promising path toward coordinated behaviors across morphologically distinct
characters via cross embodiment interaction learning.

### 2. [Large-Scale LiDAR-Inertial Dataset for Degradation-Robust High-Precision Mapping](http://arxiv.org/pdf/2507.20516v1)

Authors: Xiaofeng Jin, Ningbo Bu, Shijie Wang, Jianfei Ge, Jiangjian Xiao, Matteo Matteucci

This paper introduces a large-scale, high-precision LiDAR-Inertial Odometry
(LIO) dataset, aiming to address the insufficient validation of LIO systems in
complex real-world scenarios in existing research. The dataset covers four
diverse real-world environments spanning 60,000 to 750,000 square meters,
collected using a custom backpack-mounted platform equipped with multi-beam
LiDAR, an industrial-grade IMU, and RTK-GNSS modules. The dataset includes long
trajectories, complex scenes, and high-precision ground truth, generated by
fusing SLAM-based optimization with RTK-GNSS anchoring, and validated for
trajectory accuracy through the integration of oblique photogrammetry and
RTK-GNSS. This dataset provides a comprehensive benchmark for evaluating the
generalization ability of LIO systems in practical high-precision mapping
scenarios.

### 3. [Uni-Mapper: Unified Mapping Framework for Multi-modal LiDARs in Complex and Dynamic Environments](http://arxiv.org/pdf/2507.20538v1)

Authors: Gilhwan Kang, Hogyun Kim, Byunghee Choi, Seokhwan Jeong, Young-Sik Shin, Younggun Cho

The unification of disparate maps is crucial for enabling scalable robot
operation across multiple sessions and collaborative multi-robot scenarios.
However, achieving a unified map robust to sensor modalities and dynamic
environments remains a challenging problem. Variations in LiDAR types and
dynamic elements lead to differences in point cloud distribution and scene
consistency, hindering reliable descriptor generation and loop closure
detection essential for accurate map alignment. To address these challenges,
this paper presents Uni-Mapper, a dynamic-aware 3D point cloud map merging
framework for multi-modal LiDAR systems. It comprises dynamic object removal,
dynamic-aware loop closure, and multi-modal LiDAR map merging modules. A
voxel-wise free space hash map is built in a coarse-to-fine manner to identify
and reject dynamic objects via temporal occupancy inconsistencies. The removal
module is integrated with a LiDAR global descriptor, which encodes preserved
static local features to ensure robust place recognition in dynamic
environments. In the final stage, multiple pose graph optimizations are
conducted for both intra-session and inter-map loop closures. We adopt a
centralized anchor-node strategy to mitigate intra-session drift errors during
map merging. In the final stage, centralized anchor-node-based pose graph
optimization is performed to address intra- and inter-map loop closures for
globally consistent map merging. Our framework is evaluated on diverse
real-world datasets with dynamic objects and heterogeneous LiDARs, showing
superior performance in loop detection across sensor modalities, robust mapping
in dynamic environments, and accurate multi-map alignment over existing
methods. Project Page: https://sparolab.github.io/research/uni_mapper.

### 4. [FMimic: Foundation Models are Fine-grained Action Learners from Human Videos](http://arxiv.org/pdf/2507.20622v1)

Authors: Guangyan Chen, Meiling Wang, Te Cui, Yao Mu, Haoyang Lu, Zicai Peng, Mengxiao Hu, Tianxing Zhou, Mengyin Fu, Yi Yang, Yufeng Yue

Visual imitation learning (VIL) provides an efficient and intuitive strategy
for robotic systems to acquire novel skills. Recent advancements in foundation
models, particularly Vision Language Models (VLMs), have demonstrated
remarkable capabilities in visual and linguistic reasoning for VIL tasks.
Despite this progress, existing approaches primarily utilize these models for
learning high-level plans from human demonstrations, relying on pre-defined
motion primitives for executing physical interactions, which remains a major
bottleneck for robotic systems. In this work, we present FMimic, a novel
paradigm that harnesses foundation models to directly learn generalizable
skills at even fine-grained action levels, using only a limited number of human
videos. Extensive experiments demonstrate that our FMimic delivers strong
performance with a single human video, and significantly outperforms all other
methods with five videos. Furthermore, our method exhibits significant
improvements of over 39% and 29% in RLBench multi-task experiments and
real-world manipulation tasks, respectively, and exceeds baselines by more than
34% in high-precision tasks and 47% in long-horizon tasks.

### 5. [A Strawberry Harvesting Tool with Minimal Footprint](http://arxiv.org/pdf/2507.20784v1)

Authors: Mohamed Sorour, Mohamed Heshmat, Khaled Elgeneidy, Pål Johan From

In this paper, a novel prototype for harvesting table-top grown strawberries
is presented, that is minimalist in its footprint interacting with the fruit.
In our methodology, a smooth trapper manipulates the stem into a precise groove
location at which a distant laser beam is focused. The tool reaches
temperatures as high as 188{\deg} Celsius and as such killing germs and
preventing the spread of local plant diseases. The burnt stem wound preserves
water content and in turn the fruit shelf life. Cycle and cut times achieved
are 5.56 and 2.88 seconds respectively in successful in-door harvesting
demonstration. Extensive experiments are performed to optimize the laser spot
diameter and lateral speed against the cutting time.

### 6. [Hanging Around: Cognitive Inspired Reasoning for Reactive Robotics](http://arxiv.org/pdf/2507.20832v1)

Authors: Mihai Pomarlan, Stefano De Giorgis, Rachel Ringe, Maria M. Hedblom, Nikolaos Tsiogkas

Situationally-aware artificial agents operating with competence in natural
environments face several challenges: spatial awareness, object affordance
detection, dynamic changes and unpredictability. A critical challenge is the
agent's ability to identify and monitor environmental elements pertinent to its
objectives. Our research introduces a neurosymbolic modular architecture for
reactive robotics. Our system combines a neural component performing object
recognition over the environment and image processing techniques such as
optical flow, with symbolic representation and reasoning. The reasoning system
is grounded in the embodied cognition paradigm, via integrating image schematic
knowledge in an ontological structure. The ontology is operatively used to
create queries for the perception system, decide on actions, and infer
entities' capabilities derived from perceptual data. The combination of
reasoning and image processing allows the agent to focus its perception for
normal operation as well as discover new concepts for parts of objects involved
in particular interactions. The discovered concepts allow the robot to
autonomously acquire training data and adjust its subsymbolic perception to
recognize the parts, as well as making planning for more complex tasks feasible
by focusing search on those relevant object parts. We demonstrate our approach
in a simulated world, in which an agent learns to recognize parts of objects
involved in support relations. While the agent has no concept of handle
initially, by observing examples of supported objects hanging from a hook it
learns to recognize the parts involved in establishing support and becomes able
to plan the establishment/destruction of the support relation. This underscores
the agent's capability to expand its knowledge through observation in a
systematic way, and illustrates the potential of combining deep reasoning
[...].

### 7. [Uncertainty-aware Planning with Inaccurate Models for Robotized Liquid Handling](http://arxiv.org/pdf/2507.20861v1)

Authors: Marco Faroni, Carlo Odesco, Andrea Zanchettin, Paolo Rocco

Physics-based simulations and learning-based models are vital for complex
robotics tasks like deformable object manipulation and liquid handling.
However, these models often struggle with accuracy due to epistemic uncertainty
or the sim-to-real gap. For instance, accurately pouring liquid from one
container to another poses challenges, particularly when models are trained on
limited demonstrations and may perform poorly in novel situations. This paper
proposes an uncertainty-aware Monte Carlo Tree Search (MCTS) algorithm designed
to mitigate these inaccuracies. By incorporating estimates of model
uncertainty, the proposed MCTS strategy biases the search towards actions with
lower predicted uncertainty. This approach enhances the reliability of planning
under uncertain conditions. Applied to a liquid pouring task, our method
demonstrates improved success rates even with models trained on minimal data,
outperforming traditional methods and showcasing its potential for robust
decision-making in robotics.

### 8. [A Human-in-the-loop Approach to Robot Action Replanning through LLM Common-Sense Reasoning](http://arxiv.org/pdf/2507.20870v1)

Authors: Elena Merlo, Marta Lagomarsino, Arash Ajoudani

To facilitate the wider adoption of robotics, accessible programming tools
are required for non-experts. Observational learning enables intuitive human
skills transfer through hands-on demonstrations, but relying solely on visual
input can be inefficient in terms of scalability and failure mitigation,
especially when based on a single demonstration. This paper presents a
human-in-the-loop method for enhancing the robot execution plan, automatically
generated based on a single RGB video, with natural language input to a Large
Language Model (LLM). By including user-specified goals or critical task
aspects and exploiting the LLM common-sense reasoning, the system adjusts the
vision-based plan to prevent potential failures and adapts it based on the
received instructions. Experiments demonstrated the framework intuitiveness and
effectiveness in correcting vision-derived errors and adapting plans without
requiring additional demonstrations. Moreover, interactive plan refinement and
hallucination corrections promoted system robustness.

### 9. [PixelNav: Towards Model-based Vision-Only Navigation with Topological Graphs](http://arxiv.org/pdf/2507.20892v1)

Authors: Sergey Bakulin, Timur Akhtyamov, Denis Fatykhov, German Devchich, Gonzalo Ferrer

This work proposes a novel hybrid approach for vision-only navigation of
mobile robots, which combines advances of both deep learning approaches and
classical model-based planning algorithms. Today, purely data-driven end-to-end
models are dominant solutions to this problem. Despite advantages such as
flexibility and adaptability, the requirement of a large amount of training
data and limited interpretability are the main bottlenecks for their practical
applications. To address these limitations, we propose a hierarchical system
that utilizes recent advances in model predictive control, traversability
estimation, visual place recognition, and pose estimation, employing
topological graphs as a representation of the target environment. Using such a
combination, we provide a scalable system with a higher level of
interpretability compared to end-to-end approaches. Extensive real-world
experiments show the efficiency of the proposed method.

### 10. [Methods for the Segmentation of Reticular Structures Using 3D LiDAR Data: A Comparative Evaluation](http://arxiv.org/pdf/2507.20589v1)

Authors: Francisco J. Soler Mora, Adrián Peidró Vidal, Marc Fabregat-Jaén, Luis Payá Castelló, Óscar Reinoso García

Reticular structures form the backbone of major infrastructure like bridges,
pylons, and airports, but their inspection and maintenance are costly and
hazardous, often requiring human intervention. While prior research has focused
on fault detection via images or robotic platform design, the autonomous
navigation of robots within these structures is less explored. This study
addresses that gap by proposing methods to detect navigable surfaces in truss
structures, enhancing the autonomy of climbing robots. The paper introduces
several approaches for binary segmentation of navigable surfaces versus
background from 3D point clouds of metallic trusses. These methods fall into
two categories: analytical algorithms and deep learning models. The analytical
approach features a custom algorithm that segments structures by analyzing the
eigendecomposition of planar patches in the point cloud. In parallel, advanced
deep learning models PointNet, PointNet++, MinkUNet34C, and PointTransformerV3
are trained and evaluated for the same task. Comparative analysis shows that
the analytical algorithm offers easier parameter tuning and performance
comparable to deep learning models, which, while more computationally
intensive, excel in segmentation accuracy. Notably, PointTransformerV3 achieves
a Mean Intersection Over Union (mIoU) of about 97%. The study demonstrates the
promise of both analytical and deep learning methods for improving autonomous
navigation in complex truss environments. The results highlight the trade-offs
between computational efficiency and segmentation performance, providing
valuable guidance for future research and practical applications in autonomous
infrastructure inspection and maintenance.

### Software Engineering

### 1. [GeoJSEval: An Automated Evaluation Framework for Large Language Models on JavaScript-Based Geospatial Computation and Visualization Code Generation](http://arxiv.org/pdf/2507.20553v1)

Authors: Guanyu Chen, Haoyue Jiao, Shuyang Hou, Ziqi Liu, Lutong Xie, Shaowen Wu, Huayi Wu, Xuefeng Guan, Zhipeng Gui

With the widespread adoption of large language models (LLMs) in code
generation tasks, geospatial code generation has emerged as a critical frontier
in the integration of artificial intelligence and geoscientific analysis. This
trend underscores the urgent need for systematic evaluation methodologies to
assess LLMs generation capabilities in geospatial contexts. In particular,
geospatial computation and visualization tasks in JavaScript environments rely
heavily on orchestrating diverse frontend libraries and ecosystems, placing
elevated demands on a model's semantic understanding and code synthesis
abilities. To address this challenge, we propose GeoJSEval--the first
multimodal, function-level automatic evaluation framework for LLMs in
JavaScript-based geospatial code generation. GeoJSEval comprises three core
components: a standardized test suite (GeoJSEval-Bench), a code submission
engine, and an evaluation module. It includes 432 function-level tasks and
2,071 structured test cases spanning five widely used JavaScript geospatial
libraries and 25 mainstream geospatial data types. GeoJSEval enables
multidimensional quantitative evaluation across metrics such as accuracy,
output stability, execution efficiency, resource consumption, and error type
distribution, and integrates boundary testing mechanisms to enhance robustness
and coverage. We conduct a comprehensive evaluation of 18 state-of-the-art LLMs
using GeoJSEval, revealing significant performance disparities and bottlenecks
in spatial semantic understanding, code reliability, and function invocation
accuracy. GeoJSEval provides a foundational methodology, evaluation resource,
and practical toolkit for the standardized assessment and optimization of
geospatial code generation models, with strong extensibility and applicability
in real-world scenarios.

### 2. [Intention-Driven Generation of Project-Specific Test Cases](http://arxiv.org/pdf/2507.20619v1)

Authors: Binhang Qi, Yun Lin, Xinyi Weng, Yuhuan Huang, Chenyan Liu, Hailong Sun, Jin Song Dong

Test cases are valuable assets for maintaining software quality. While
numerous automated techniques have been proposed for generating tests (either
by maximizing code coverage or by translating focal code into test code),
practical tests are seldom driven by coverage alone. In real projects, each
test reflects a developer's validation intention for a specific behaviour and
embodies rich, project-specific knowledge: which specific APIs to call and what
assertions truly matter. Without considering such knowledge, tests can hardly
pass code review and be integrated into the software product.
  In this work, we propose IntentionTest, which generates project-specific
tests with validation intention as a structured description. Our design is
motivated by two insights: (1) a description of validation intention, compared
to coverage and focal code, carries more crucial information about what to
test; and (2) practical tests exhibit high code duplication, indicating that
domain knowledge is highly reusable for writing new tests. Given a focal code
and a description of validation intention (in the form of either an informal
comment or a formal test plan), IntentionTest retrieves a referable test in the
project to guide test generation. Moreover, IntentionTest reduces the test
generation problem into an editing problem on the test code regarding the
validation intention. It generates a test including both test prefix and
oracle, which aims to be executable and semantically correct.
  We evaluate IntentionTest against state-of-the-art baselines on 4,146 test
cases from 13 open-source projects. Specifically, compared to ChatTester,
IntentionTest can (1) generate significantly more semantically correct tests,
improving common mutation scores by 39.03% and coverage overlap with
ground-truth tests by 40.14%; (2) generate 21.30% more successful passing
tests.

### 3. [Client--Library Compatibility Testing with API Interaction Snapshots](http://arxiv.org/pdf/2507.20814v1)

Authors: Gustave Monce, Thomas Degueule, Jean-Rémy Falleri, Romain Robbes

Modern software development heavily relies on third-party libraries to speed
up development and enhance quality. As libraries evolve, they may break the
tacit contract established with their clients by introducing behavioral
breaking changes (BBCs) that alter run-time behavior and silently break client
applications without being detected at compile time. Traditional regression
tests on the client side often fail to detect such BBCs, either due to limited
library coverage or weak assertions that do not sufficiently exercise the
library's expected behavior. To address this issue, we propose a novel approach
to client--library compatibility testing that leverages existing client tests
in a novel way. Instead of relying on developer-written assertions, we propose
recording the actual interactions at the API boundary during the execution of
client tests (protocol, input and output values, exceptions, etc.). These
sequences of API interactions are stored as snapshots which capture the exact
contract expected by a client at a specific point in time. As the library
evolves, we compare the original and new snapshots to identify perturbations in
the contract, flag potential BBCs, and notify clients. We implement this
technique in our prototype tool Gilesi, a Java framework that automatically
instruments library APIs, records snapshots, and compares them. Through a
preliminary case study on several client--library pairs with artificially
seeded BBCs, we show that Gilesi reliably detects BBCs missed by client test
suites.

### 4. [Distinguishing Quantum Software Bugs from Hardware Noise: A Statistical Approach](http://arxiv.org/pdf/2507.20475v1)

Authors: Ahmik Virani, Devraj, Anirudh Suresh, Lei Zhang, M V Panduranga Rao

Quantum computing in the Noisy Intermediate-Scale Quantum (NISQ) era presents
significant challenges in differentiating quantum software bugs from hardware
noise. Traditional debugging techniques from classical software engineering
cannot directly resolve this issue due to the inherently stochastic nature of
quantum computation mixed with noises from NISQ computers. To address this gap,
we propose a statistical approach leveraging probabilistic metrics to
differentiate between quantum software bugs and hardware noise. We evaluate our
methodology empirically using well-known quantum algorithms, including Grover's
algorithm, Deutsch-Jozsa algorithm, and Simon's algorithm. Experimental results
demonstrate the efficacy and practical applicability of our approach, providing
quantum software developers with a reliable analytical tool to identify and
classify unexpected behavior in quantum programs.

### 5. [VDGraph: A Graph-Theoretic Approach to Unlock Insights from SBOM and SCA Data](http://arxiv.org/pdf/2507.20502v1)

Authors: Howell Xia, Jonah Gluck, Sevval Simsek, David Sastre Medina, David Starobinski

The high complexity of modern software supply chains necessitates tools such
as Software Bill of Materials (SBOMs) to manage component dependencies, and
Software Composition Analysis (SCA) tools to identify vulnerabilities. While
there exists limited integration between SBOMs and SCA tools, a unified view of
complex dependency-vulnerability relationships remains elusive. In this paper,
we introduce VDGraph, a novel knowledge graph-based methodology for integrating
vulnerability and dependency data into a holistic view. VDGraph consolidates
SBOM and SCA outputs into a graph representation of software projects'
dependencies and vulnerabilities. We provide a formal description and analysis
of the theoretical properties of VDGraph and present solutions to manage
possible conflicts between the SBOM and SCA data. We further introduce and
evaluate a practical, proof-of-concept implementation of VDGraph using two
popular SBOM and SCA tools, namely CycloneDX Maven plugin and Google's
OSV-Scanner. We apply VDGraph on 21 popular Java projects. Through the
formulation of appropriate queries on the graphs, we uncover the existence of
concentrated risk points (i.e., vulnerable components of high severity
reachable through numerous dependency paths). We further show that
vulnerabilities predominantly emerge at a depth of three dependency levels or
higher, indicating that direct or secondary dependencies exhibit lower
vulnerability density and tend to be more secure. Thus, VDGraph contributes a
graph-theoretic methodology that improves visibility into how vulnerabilities
propagate through complex, transitive dependencies. Moreover, our
implementation, which combines open SBOM and SCA standards with Neo4j, lays a
foundation for scalable and automated analysis across real-world projects.

### 6. [LLM-Based Repair of Static Nullability Errors](http://arxiv.org/pdf/2507.20674v1)

Authors: Nima Karimipour, Michael Pradel, Martin Kellogg, Manu Sridharan

Modern Java projects increasingly adopt static analysis tools that prevent
null-pointer exceptions by treating nullness as a type property. However,
integrating such tools into large, existing codebases remains a significant
challenge. While annotation inference can eliminate many errors automatically,
a subset of residual errors -- typically a mix of real bugs and false positives
-- often persist and can only be resolved via code changes. Manually addressing
these errors is tedious and error-prone. Large language models (LLMs) offer a
promising path toward automating these repairs, but naively-prompted LLMs often
generate incorrect, contextually-inappropriate edits. Resolving a nullability
error demands a deep understanding of how a symbol is used across the codebase,
often spanning methods, classes, and packages. We present NullRepair, a system
that integrates LLMs into a structured workflow for resolving the errors from a
nullability checker. NullRepair's decision process follows a flowchart derived
from manual analysis of 200 real-world errors. It leverages static analysis to
identify safe and unsafe usage regions of symbols, using error-free usage
examples to contextualize model prompts. Patches are generated through an
iterative interaction with the LLM that incorporates project-wide context and
decision logic. Our evaluation on 12 real-world Java projects shows that
NullRepair resolves an average of 72% of the errors that remain after applying
a state-of-the-art annotation inference technique. Unlike a naively-prompted
LLM, NullRepair also largely preserves program semantics, with all unit tests
passing in 10/12 projects after applying every edit proposed by NullRepair, and
98% or more tests passing in the remaining two projects.

### 7. [Search-Based Fuzzing For RESTful APIs That Use MongoDB](http://arxiv.org/pdf/2507.20848v1)

Authors: Hernan Ghianni, Man Zhang, Juan P. Galeotti, Andrea Arcuri

In RESTful APIs, interactions with a database are a common and crucial
aspect. When generating whitebox tests, it is essential to consider the
database's state (i.e., the data contained in the database) to achieve higher
code coverage and uncover more hidden faults. This article presents novel
techniques to enhance search-based software test generation for RESTful APIs
interacting with NoSQL databases. Specifically, we target the popular MongoDB
database, by dynamically analyzing (via automated code instrumentation) the
state of the database during the test generation process. Additionally, to
achieve better results, our novel approach allows inserting NoSQL data directly
from test cases. This is particularly beneficial when generating the correct
sequence of events to set the NoSQL database in an appropriate state is
challenging or time-consuming. This method is also advantageous for testing
read-only microservices. Our novel techniques are implemented as an extension
of EvoMaster, the only open-source tool for white-box fuzzing RESTful APIs.
Experiments conducted on six RESTful APIs demonstrated significant improvements
in code coverage, with increases of up to 18% compared to existing white-box
approaches. To better highlight the improvements of our novel techniques,
comparisons are also carried out with four state-of-the-art black-box fuzzers.

### 8. [Enhancing Project-Specific Code Completion by Inferring Internal API Information](http://arxiv.org/pdf/2507.20888v1)

Authors: Le Deng, Xiaoxue Ren, Chao Ni, Ming Liang, David Lo, Zhongxin Liu

Project-specific code completion is a critical task that leverages context
from a project to generate accurate code. State-of-the-art methods use
retrieval-augmented generation (RAG) with large language models (LLMs) and
project information for code completion. However, they often struggle to
incorporate internal API information, which is crucial for accuracy, especially
when APIs are not explicitly imported in the file.
  To address this, we propose a method to infer internal API information
without relying on imports. Our method extends the representation of APIs by
constructing usage examples and semantic descriptions, building a knowledge
base for LLMs to generate relevant completions. We also introduce ProjBench, a
benchmark that avoids leaked imports and consists of large-scale real-world
projects.
  Experiments on ProjBench and CrossCodeEval show that our approach
significantly outperforms existing methods, improving code exact match by
22.72% and identifier exact match by 18.31%. Additionally, integrating our
method with existing baselines boosts code match by 47.80% and identifier match
by 35.55%.

### 9. [Smart Expansion Techniques for ASP-based Interactive Configuration](http://arxiv.org/pdf/2507.21027v1)

Authors: Lucia Balážová, Richard Comploi-Taupe, Susana Hahn, Nicolas Rühling, Gottfried Schenner

Product configuration is a successful application of Answer Set Programming
(ASP). However, challenges are still open for interactive systems to
effectively guide users through the configuration process. The aim of our work
is to provide an ASP-based solver for interactive configuration that can deal
with large-scale industrial configuration problems and that supports intuitive
user interfaces via an API. In this paper, we focus on improving the
performance of automatically completing a partial configuration. Our main
contribution enhances the classical incremental approach for multi-shot solving
by four different smart expansion functions. The core idea is to determine and
add specific objects or associations to the partial configuration by exploiting
cautious and brave consequences before checking for the existence of a complete
configuration with the current objects in each iteration. This approach limits
the number of costly unsatisfiability checks and reduces the search space,
thereby improving solving performance. In addition, we present a user interface
that uses our API and is implemented in ASP.

### 10. [Repairing vulnerabilities without invisible hands. A differentiated replication study on LLMs](http://arxiv.org/pdf/2507.20977v1)

Authors: Maria Camporese, Fabio Massacci

Background: Automated Vulnerability Repair (AVR) is a fast-growing branch of
program repair. Recent studies show that large language models (LLMs)
outperform traditional techniques, extending their success beyond code
generation and fault detection.
  Hypothesis: These gains may be driven by hidden factors -- "invisible hands"
such as training-data leakage or perfect fault localization -- that let an LLM
reproduce human-authored fixes for the same code.
  Objective: We replicate prior AVR studies under controlled conditions by
deliberately adding errors to the reported vulnerability location in the
prompt. If LLMs merely regurgitate memorized fixes, both small and large
localization errors should yield the same number of correct patches, because
any offset should divert the model from the original fix.
  Method: Our pipeline repairs vulnerabilities from the Vul4J and VJTrans
benchmarks after shifting the fault location by n lines from the ground truth.
A first LLM generates a patch, a second LLM reviews it, and we validate the
result with regression and proof-of-vulnerability tests. Finally, we manually
audit a sample of patches and estimate the error rate with the
Agresti-Coull-Wilson method.

### Social and Information Networks

### 1. [Structural-Aware Key Node Identification in Hypergraphs via Representation Learning and Fine-Tuning](http://arxiv.org/pdf/2507.20682v1)

Authors: Xiaonan Ni, Guangyuan Mei, Su-Su Zhang, Yang Chen, Xin Xu, Chuang Liu, Xiu-Xiu Zhan

Evaluating node importance is a critical aspect of analyzing complex systems,
with broad applications in digital marketing, rumor suppression, and disease
control. However, existing methods typically rely on conventional network
structures and fail to capture the polyadic interactions intrinsic to many
real-world systems. To address this limitation, we study key node
identification in hypergraphs, where higher-order interactions are naturally
modeled as hyperedges. We propose a novel framework, AHGA, which integrates an
Autoencoder for extracting higher-order structural features, a HyperGraph
neural network-based pre-training module (HGNN), and an Active learning-based
fine-tuning process. This fine-tuning step plays a vital role in mitigating the
gap between synthetic and real-world data, thereby enhancing the model's
robustness and generalization across diverse hypergraph topologies. Extensive
experiments on eight empirical hypergraphs show that AHGA outperforms classical
centrality-based baselines by approximately 37.4%. Furthermore, the nodes
identified by AHGA exhibit both high influence and strong structural disruption
capability, demonstrating their superiority in detecting multifunctional nodes.

### 2. [Improving Community Detection in Academic Networks by Handling Publication Bias](http://arxiv.org/pdf/2507.20449v1)

Authors: Md Asaduzzaman Noor, John Sheppard, Jason Clark

Finding potential research collaborators is a challenging task, especially in
today's fast-growing and interdisciplinary research landscape. While
traditional methods often rely on observable relationships such as
co-authorships and citations to construct the research network, in this work,
we focus solely on publication content to build a topic-based research network
using BERTopic with a fine-tuned SciBERT model that connects and recommends
researchers across disciplines based on shared topical interests. A major
challenge we address is publication imbalance, where some researchers publish
much more than others, often across several topics. Without careful handling,
their less frequent interests are hidden under dominant topics, limiting the
network's ability to detect their full research scope. To tackle this, we
introduce a cloning strategy that clusters a researcher's publications and
treats each cluster as a separate node. This allows researchers to be part of
multiple communities, improving the detection of interdisciplinary links.
Evaluation on the proposed method shows that the cloned network structure leads
to more meaningful communities and uncovers a broader set of collaboration
opportunities.

### 3. [FHSTP@EXIST 2025 Benchmark: Sexism Detection with Transparent Speech Concept Bottleneck Models](http://arxiv.org/pdf/2507.20924v1)

Authors: Roberto Labadie-Tamayo, Adrian Jaques Böck, Djordje Slijepčević, Xihui Chen, Andreas Babic, Matthias Zeppelzauer

Sexism has become widespread on social media and in online conversation. To
help address this issue, the fifth Sexism Identification in Social Networks
(EXIST) challenge is initiated at CLEF 2025. Among this year's international
benchmarks, we concentrate on solving the first task aiming to identify and
classify sexism in social media textual posts. In this paper, we describe our
solutions and report results for three subtasks: Subtask 1.1 - Sexism
Identification in Tweets, Subtask 1.2 - Source Intention in Tweets, and Subtask
1.3 - Sexism Categorization in Tweets. We implement three models to address
each subtask which constitute three individual runs: Speech Concept Bottleneck
Model (SCBM), Speech Concept Bottleneck Model with Transformer (SCBMT), and a
fine-tuned XLM-RoBERTa transformer model. SCBM uses descriptive adjectives as
human-interpretable bottleneck concepts. SCBM leverages large language models
(LLMs) to encode input texts into a human-interpretable representation of
adjectives, then used to train a lightweight classifier for downstream tasks.
SCBMT extends SCBM by fusing adjective-based representation with contextual
embeddings from transformers to balance interpretability and classification
performance. Beyond competitive results, these two models offer fine-grained
explanations at both instance (local) and class (global) levels. We also
investigate how additional metadata, e.g., annotators' demographic profiles,
can be leveraged. For Subtask 1.1, XLM-RoBERTa, fine-tuned on provided data
augmented with prior datasets, ranks 6th for English and Spanish and 4th for
English in the Soft-Soft evaluation. Our SCBMT achieves 7th for English and
Spanish and 6th for Spanish.

### Systems and Control

### 1. [HJB-based online safety-embedded critic learning for uncertain systems with self-triggered mechanism](http://arxiv.org/pdf/2507.20545v1)

Authors: Zhanglin Shangguan, Bo Yang, Qi Li, Wei Xiao, Xingping Guan

This paper presents a learning-based optimal control framework for
safety-critical systems with parametric uncertainties, addressing both
time-triggered and self-triggered controller implementations. First, we develop
a robust control barrier function (RCBF) incorporating Lyapunov-based
compensation terms to rigorously guarantee safety despite parametric
uncertainties. Building on this safety guarantee, we formulate the constrained
optimal control problem as the minimization of a novel safety-embedded value
function, where the RCBF is involved via a Lagrange multiplier that adaptively
balances safety constraints against optimal stabilization objectives. To
enhance computational efficiency, we propose a self-triggered implementation
mechanism that reduces control updates while maintaining dual stability-safety
guarantees. The resulting self-triggered constrained Hamilton-Jacobi-Bellman
(HJB) equation is solved through an online safety-embedded critic learning
framework, with the Lagrange multiplier computed in real time to ensure safety.
Numerical simulations demonstrate the effectiveness of the proposed approach in
achieving both safety and control performance.

### 2. [A Modified Adaptive Data-Enabled Policy Optimization Control to Resolve State Perturbations](http://arxiv.org/pdf/2507.20580v1)

Authors: Mojtaba Kaheni, Niklas Persson, Vittorio De Iuliis, Costanzo Manes, Alessandro V. Papadopoulos

This paper proposes modifications to the data-enabled policy optimization
(DeePO) algorithm to mitigate state perturbations. DeePO is an adaptive,
data-driven approach designed to iteratively compute a feedback gain equivalent
to the certainty-equivalence LQR gain. Like other data-driven approaches based
on Willems' fundamental lemma, DeePO requires persistently exciting input
signals. However, linear state-feedback gains from LQR designs cannot
inherently produce such inputs. To address this, probing noise is
conventionally added to the control signal to ensure persistent excitation.
However, the added noise may induce undesirable state perturbations. We first
identify two key issues that jeopardize the desired performance of DeePO when
probing noise is not added: the convergence of states to the equilibrium point,
and the convergence of the controller to its optimal value. To address these
challenges without relying on probing noise, we propose Perturbation-Free DeePO
(PFDeePO) built on two fundamental principles. First, the algorithm pauses the
control gain updating in DeePO process when system states are near the
equilibrium point. Second, it applies a multiplicative noise, scaled by a mean
value of $1$ as a gain for the control signal, when the controller converges.
This approach minimizes the impact of noise as the system approaches
equilibrium while preserving stability. We demonstrate the effectiveness of
PFDeePO through simulations, showcasing its ability to eliminate state
perturbations while maintaining system performance and stability.

### 3. [SNR Optimization for Common Emitter Amplifier](http://arxiv.org/pdf/2507.20669v1)

Authors: Orhan Gazi

In this paper we investigate the effects of the thermal noise of the base
resistance of common emitter amplifier (CEA) on the output SNR, and we show
that a first order Butterworth filter at the output of the CEA significantly
improves output SNR significantly and supress the performances of higher order
Butterworth, Chebyshev I, II and elliptic filters. We propose a formula for the
selection of cut-off frequency of analog filters for given orders to achieve
significant SNR improvement at CEA output. Considering the filter complexity
and output SNR improvement, we can conclude that the first order Butterworth
filter outperforms Chebyshev I, II and elliptic filters.

### 4. [What's Really Different with AI? -- A Behavior-based Perspective on System Safety for Automated Driving Systems](http://arxiv.org/pdf/2507.20685v1)

Authors: Marcus Nolte, Nayel Fabian Salem, Olaf Franke, Jan Heckmann, Christoph Höhmann, Georg Stettinger, Markus Maurer

Assuring safety for ``AI-based'' systems is one of the current challenges in
safety engineering. For automated driving systems, in particular, further
assurance challenges result from the open context that the systems need to
operate in after deployment. The current standardization and regulation
landscape for ``AI-based'' systems is becoming ever more complex, as standards
and regulations are being released at high frequencies.
  This position paper seeks to provide guidance for making qualified arguments
which standards should meaningfully be applied to (``AI-based'') automated
driving systems. Furthermore, we argue for clearly differentiating sources of
risk between AI-specific and general uncertainties related to the open context.
In our view, a clear conceptual separation can help to exploit commonalities
that can close the gap between system-level and AI-specific safety analyses,
while ensuring the required rigor for engineering safe ``AI-based'' systems.

### 5. [Minimum Attention Control (MAC) in a Receding Horizon Framework with Applications](http://arxiv.org/pdf/2507.20835v1)

Authors: T. Ganesh Teja, Santhosh Kumar Varanasi, Phanindra Jampana

Minimum Attention Control (MAC) is a control technique that provides minimal
input changes to meet the control objective. Mathematically, the zero norm of
the input changes is used as a constraint for the given control objective and
minimized with respect to the process dynamics. In this paper, along with the
zero norm constraint, stage costs are also considered for reference tracking in
a receding horizon framework. For this purpose, the optimal inputs of the
previous horizons are also considered in the optimization problem of the
current horizon. An alternating minimization algorithm is applied to solve the
optimization problem (Minimum Attention Model Predictive Control (MAMPC)). The
outer step of the optimization is a quadratic program, while the inner step,
which solves for sparsity, has an analytical solution. The proposed algorithm
is implemented on two case studies: a four-tank system with slow dynamics and a
fuel cell stack with fast dynamics. A detailed comparative study of the
proposed algorithm with standard MPC indicates sparse control actions with a
tradeoff in the tracking error.

### 6. [dq Modeling for Series-Parallel Compensated Wireless Power Transfer Systems](http://arxiv.org/pdf/2507.20921v1)

Authors: Zixuan Jiang

Series-parallel (SP) compensated wireless power transfer (WPT) systems are
widely used in some specific scenarios, such as bioelectronics and portable
electronics. However, most studies are based on the phasor method and focused
on the steady-state analysis, which may overlook the transient process of
systems. Accordingly, inspired by the notion of coordinate transformation in
the field of motor drive, this work develops a dq modeling method for SP
compensated WPT systems. The proposed model effectively characterizes
first-order system dynamics, facilitating enhanced measurement precision and
control system development. One measurement application, dq model-based mutual
inductance identification, is presented to reflect the value of the dq model.
Simulation results are shown to validate the model's effectiveness, indicating
that the developed model can be a good tool for the design of SP compensated
WPT systems.

### 7. [A lightweight numerical model for predictive control of borehole thermal energy storages](http://arxiv.org/pdf/2507.20974v1)

Authors: Johannes van Randenborgh, Steffen Daniel, Moritz Schulze Darup

Borehole thermal energy storage (BTES) can reduce the operation of fossil
fuel-based heating, ventilation, and air conditioning systems for buildings.
With BTES, thermal energy is stored via a borehole heat exchanger in the
ground. Model predictive control (MPC) may maximize the use of BTES by
achieving a dynamic interaction between the building and BTES. However,
modeling BTES for MPC is challenging, and a trade-off between model accuracy
and an easy-to-solve optimal control problem (OCP) must be found. This
manuscript presents an accurate numerical model yielding an easy-to-solve
linear-quadratic OCP.

### 8. [LLMs-guided adaptive compensator: Bringing Adaptivity to Automatic Control Systems with Large Language Models](http://arxiv.org/pdf/2507.20509v1)

Authors: Zhongchao Zhou, Yuxi Lu, Yaonan Zhu, Yifan Zhao, Bin He, Liang He, Wenwen Yu, Yusuke Iwasawa

With rapid advances in code generation, reasoning, and problem-solving, Large
Language Models (LLMs) are increasingly applied in robotics. Most existing work
focuses on high-level tasks such as task decomposition. A few studies have
explored the use of LLMs in feedback controller design; however, these efforts
are restricted to overly simplified systems, fixed-structure gain tuning, and
lack real-world validation. To further investigate LLMs in automatic control,
this work targets a key subfield: adaptive control. Inspired by the framework
of model reference adaptive control (MRAC), we propose an LLM-guided adaptive
compensator framework that avoids designing controllers from scratch. Instead,
the LLMs are prompted using the discrepancies between an unknown system and a
reference system to design a compensator that aligns the response of the
unknown system with that of the reference, thereby achieving adaptivity.
Experiments evaluate five methods: LLM-guided adaptive compensator, LLM-guided
adaptive controller, indirect adaptive control, learning-based adaptive
control, and MRAC, on soft and humanoid robots in both simulated and real-world
environments. Results show that the LLM-guided adaptive compensator outperforms
traditional adaptive controllers and significantly reduces reasoning complexity
compared to the LLM-guided adaptive controller. The Lyapunov-based analysis and
reasoning-path inspection demonstrate that the LLM-guided adaptive compensator
enables a more structured design process by transforming mathematical
derivation into a reasoning task, while exhibiting strong generalizability,
adaptability, and robustness. This study opens a new direction for applying
LLMs in the field of automatic control, offering greater deployability and
practicality compared to vision-language models.

### 9. [Real-Time Distributed Optical Fiber Vibration Recognition via Extreme Lightweight Model and Cross-Domain Distillation](http://arxiv.org/pdf/2507.20587v1)

Authors: Zhongyao Luo, Hao Wu, Zhao Ge, Ming Tang

Distributed optical fiber vibration sensing (DVS) systems offer a promising
solution for large-scale monitoring and intrusion event recognition. However,
their practical deployment remains hindered by two major challenges:
degradation of recognition accuracy in dynamic conditions, and the
computational bottleneck of real-time processing for mass sensing data. This
paper presents a new solution to these challenges, through a FPGA-accelerated
extreme lightweight model along with a newly proposed knowledge distillation
framework. The proposed three-layer depthwise separable convolution network
contains only 4141 parameters, which is the most compact architecture in this
field to date, and achieves a maximum processing speed of 0.019 ms for each
sample covering a 12.5 m fiber length over 0.256 s. This performance
corresponds to real-time processing capabilities for sensing fibers extending
up to 168.68 km. To improve generalizability under changing environments, the
proposed cross-domain distillation framework guided by physical priors is used
here to embed frequency-domain insights into the time-domain model. This allows
for time-frequency representation learning without increasing complexity and
boosts recognition accuracy from 51.93% to 95.72% under unseen environmental
conditions. The proposed methodology provides key advancements including a
framework combining interpretable signal processing technique with deep
learning and a reference architecture for real-time processing and
edge-computing in DVS systems, and more general distributed optical fiber
sensing (DOFS) area. It mitigates the trade-off between sensing range and
real-time capability, bridging the gap between theoretical capabilities and
practical deployment requirements. Furthermore, this work reveals a new
direction for building more efficient, robust and explainable artificial
intelligence systems for DOFS technologies.

### 10. [Sequential Operation of Residential Energy Hubs](http://arxiv.org/pdf/2507.20621v1)

Authors: Darío Slaifstein, Gautham Ram Chandra Mouli, Laura Ramirez-Elizondo, Pavol Bauer

The operation of residential energy hubs with multiple energy carriers
(electricity, heat, mobility) poses a significant challenge due to different
carrier dynamics, hybrid storage coordination and high-dimensional
action-spaces. Energy management systems oversee their operation, deciding the
set points of the primary control layer. This paper presents a novel 2-stage
economic model predictive controller for electrified buildings including
physics-based models of the battery degradation and thermal systems. The
hierarchical control operates in the Dutch sequential energy markets. In
particular common assumptions regarding intra-day markets (auction and
continuous-time) are discussed as well as the coupling of the different storage
systems. The best control policy is to co-optimize day-ahead and intra-day
auctions in the first stage, to later follow intra-day auctions. If no
intra-day prices are known at the time of the day-ahead auction, its best to
follow continuous time intra-day in the summer and the intra-day auction in the
winter. Additionally, this sequential operation increases battery degradation.
Finally, under our controller the realized short-term flexibility of the
thermal energy storage is marginal compared to the flexibility delivered by
static battery pack and electric vehicles with bidirectional charging.

### Machine Learning (Statistics Category)

### 1. [Improving Group Fairness in Tensor Completion via Imbalance Mitigating Entity Augmentation](http://arxiv.org/pdf/2507.20542v1)

Authors: Dawon Ahn, Jun-Gi Jang, Evangelos E. Papalexakis

Group fairness is important to consider in tensor decomposition to prevent
discrimination based on social grounds such as gender or age. Although few
works have studied group fairness in tensor decomposition, they suffer from
performance degradation. To address this, we propose STAFF(Sparse Tensor
Augmentation For Fairness) to improve group fairness by minimizing the gap in
completion errors of different groups while reducing the overall tensor
completion error. Our main idea is to augment a tensor with augmented entities
including sufficient observed entries to mitigate imbalance and group bias in
the sparse tensor. We evaluate \method on tensor completion with various
datasets under conventional and deep learning-based tensor models. STAFF
consistently shows the best trade-off between completion error and group
fairness; at most, it yields 36% lower MSE and 59% lower MADE than the
second-best baseline.

### 2. [Locally Adaptive Conformal Inference for Operator Models](http://arxiv.org/pdf/2507.20975v1)

Authors: Trevor Harris, Yan Liu

Operator models are regression algorithms for functional data and have become
a key tool for emulating large-scale dynamical systems. Recent advances in deep
neural operators have dramatically improved the accuracy and scalability of
operator modeling, but lack an inherent notion of predictive uncertainty. We
introduce Local Spectral Conformal Inference (LSCI), a new framework for
locally adaptive, distribution-free uncertainty quantification for neural
operator models. LSCI uses projection-based depth scoring and localized
conformal inference to generate function-valued prediction sets with
statistical guarantees. We prove approximate finite-sample marginal coverage
under local exchangeability, and demonstrate significant gains in adaptivity
and coverage across synthetic and real-world operator learning tasks.

### 3. [Transformers as Unrolled Inference in Probabilistic Laplacian Eigenmaps: An Interpretation and Potential Improvements](http://arxiv.org/pdf/2507.21040v1)

Authors: Aditya Ravuri, Neil D. Lawrence

We propose a probabilistic interpretation of transformers as unrolled
inference steps assuming a probabilistic Laplacian Eigenmaps model from the
ProbDR framework. Our derivation shows that at initialisation, transformers
perform "linear" dimensionality reduction. We also show that within the
transformer block, a graph Laplacian term arises from our arguments, rather
than an attention matrix (which we interpret as an adjacency matrix). We
demonstrate that simply subtracting the identity from the attention matrix (and
thereby taking a graph diffusion step) improves validation performance on a
language model and a simple vision transformer.

### 4. [Statistical Inference for Differentially Private Stochastic Gradient Descent](http://arxiv.org/pdf/2507.20560v1)

Authors: Xintao Xia, Linjun Zhang, Zhanrui Cai

Privacy preservation in machine learning, particularly through Differentially
Private Stochastic Gradient Descent (DP-SGD), is critical for sensitive data
analysis. However, existing statistical inference methods for SGD predominantly
focus on cyclic subsampling, while DP-SGD requires randomized subsampling. This
paper first bridges this gap by establishing the asymptotic properties of SGD
under the randomized rule and extending these results to DP-SGD. For the output
of DP-SGD, we show that the asymptotic variance decomposes into statistical,
sampling, and privacy-induced components. Two methods are proposed for
constructing valid confidence intervals: the plug-in method and the random
scaling method. We also perform extensive numerical analysis, which shows that
the proposed confidence intervals achieve nominal coverage rates while
maintaining privacy.

### 5. [LargeMvC-Net: Anchor-based Deep Unfolding Network for Large-scale Multi-view Clustering](http://arxiv.org/pdf/2507.20980v1)

Authors: Shide Du, Chunming Wu, Zihan Fang, Wendi Zhao, Yilin Wu, Changwei Wang, Shiping Wang

Deep anchor-based multi-view clustering methods enhance the scalability of
neural networks by utilizing representative anchors to reduce the computational
complexity of large-scale clustering. Despite their scalability advantages,
existing approaches often incorporate anchor structures in a heuristic or
task-agnostic manner, either through post-hoc graph construction or as
auxiliary components for message passing. Such designs overlook the core
structural demands of anchor-based clustering, neglecting key optimization
principles. To bridge this gap, we revisit the underlying optimization problem
of large-scale anchor-based multi-view clustering and unfold its iterative
solution into a novel deep network architecture, termed LargeMvC-Net. The
proposed model decomposes the anchor-based clustering process into three
modules: RepresentModule, NoiseModule, and AnchorModule, corresponding to
representation learning, noise suppression, and anchor indicator estimation.
Each module is derived by unfolding a step of the original optimization
procedure into a dedicated network component, providing structural clarity and
optimization traceability. In addition, an unsupervised reconstruction loss
aligns each view with the anchor-induced latent space, encouraging consistent
clustering structures across views. Extensive experiments on several
large-scale multi-view benchmarks show that LargeMvC-Net consistently
outperforms state-of-the-art methods in terms of both effectiveness and
scalability.

### 6. [Personalized Treatment Effect Estimation from Unstructured Data](http://arxiv.org/pdf/2507.20993v1)

Authors: Henri Arno, Thomas Demeester

Existing methods for estimating personalized treatment effects typically rely
on structured covariates, limiting their applicability to unstructured data.
Yet, leveraging unstructured data for causal inference has considerable
application potential, for instance in healthcare, where clinical notes or
medical images are abundant. To this end, we first introduce an approximate
'plug-in' method trained directly on the neural representations of unstructured
data. However, when these fail to capture all confounding information, the
method may be subject to confounding bias. We therefore introduce two
theoretically grounded estimators that leverage structured measurements of the
confounders during training, but allow estimating personalized treatment
effects purely from unstructured inputs, while avoiding confounding bias. When
these structured measurements are only available for a non-representative
subset of the data, these estimators may suffer from sampling bias. To address
this, we further introduce a regression-based correction that accounts for the
non-uniform sampling, assuming the sampling mechanism is known or can be
well-estimated. Our experiments on two benchmark datasets show that the plug-in
method, directly trainable on large unstructured datasets, achieves strong
empirical performance across all settings, despite its simplicity.

### 7. [Multivariate Conformal Prediction via Conformalized Gaussian Scoring](http://arxiv.org/pdf/2507.20941v1)

Authors: Sacha Braun, Eugène Berta, Michael I. Jordan, Francis Bach

While achieving exact conditional coverage in conformal prediction is
unattainable without making strong, untestable regularity assumptions, the
promise of conformal prediction hinges on finding approximations to conditional
guarantees that are realizable in practice. A promising direction for obtaining
conditional dependence for conformal sets--in particular capturing
heteroskedasticity--is through estimating the conditional density
$\mathbb{P}_{Y|X}$ and conformalizing its level sets. Previous work in this
vein has focused on nonconformity scores based on the empirical cumulative
distribution function (CDF). Such scores are, however, computationally costly,
typically requiring expensive sampling methods. To avoid the need for sampling,
we observe that the CDF-based score reduces to a Mahalanobis distance in the
case of Gaussian scores, yielding a closed-form expression that can be directly
conformalized. Moreover, the use of a Gaussian-based score opens the door to a
number of extensions of the basic conformal method; in particular, we show how
to construct conformal sets with missing output values, refine conformal sets
as partial information about $Y$ becomes available, and construct conformal
sets on transformations of the output space. Finally, empirical results
indicate that our approach produces conformal sets that more closely
approximate conditional coverage in multivariate settings compared to
alternative methods.

### 8. [Diagonally-Weighted Generalized Method of Moments Estimation for Gaussian Mixture Modeling](http://arxiv.org/pdf/2507.20459v1)

Authors: Liu Zhang, Oscar Mickelin, Sheng Xu, Amit Singer

Since Pearson [Philosophical Transactions of the Royal Society of London. A,
185 (1894), pp. 71-110] first applied the method of moments (MM) for modeling
data as a mixture of one-dimensional Gaussians, moment-based estimation methods
have proliferated. Among these methods, the generalized method of moments (GMM)
improves the statistical efficiency of MM by weighting the moments
appropriately. However, the computational complexity and storage complexity of
MM and GMM grow exponentially with the dimension, making these methods
impractical for high-dimensional data or when higher-order moments are
required. Such computational bottlenecks are more severe in GMM since it
additionally requires estimating a large weighting matrix. To overcome these
bottlenecks, we propose the diagonally-weighted GMM (DGMM), which achieves a
balance among statistical efficiency, computational complexity, and numerical
stability. We apply DGMM to study the parameter estimation problem for weakly
separated heteroscedastic low-rank Gaussian mixtures and design a
computationally efficient and numerically stable algorithm that obtains the
DGMM estimator without explicitly computing or storing the moment tensors. We
implement the proposed algorithm and empirically validate the advantages of
DGMM: in numerical studies, DGMM attains smaller estimation errors while
requiring substantially shorter runtime than MM and GMM. The code and data will
be available upon publication at https://github.com/liu-lzhang/dgmm.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-29 PST.

### 1. [The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](https://www.nature.com/articles/s41586-025-09442-9)

Authors: Kyle Swanson et al.

### 2. [Could machine learning help to build a unified theory of cognition?](https://www.nature.com/articles/d41586-025-02353-9)

Authors: Giosuè Baggio

### 3. [Cognitive link adaptation via modulation scheme classification in narrowband networks under AWGN and SUI channel conditions](https://www.nature.com/articles/s41598-025-12277-z)

Authors: Fatima Ismail et al.

### 4. [Studying the performance of YOLOv11 incorporating DHSA BRA and PPA modules in railway track fasteners defect detection](https://www.nature.com/articles/s41598-025-13435-z)

Authors: Chengwei Zhang et al.

### 5. [Pretraining-improved Spatiotemporal graph network for the generalization performance enhancement of traffic forecasting](https://www.nature.com/articles/s41598-025-11375-2)

Authors: Xiangyue Zhang et al.

### 6. [Hydraulic Performance Modeling of Inclined Double Cutoff Walls Beneath Hydraulic Structures Using Optimized Ensemble Machine Learning](https://www.nature.com/articles/s41598-025-10990-3)

Authors: Mohamed Kamel Elshaarawy et al.

### 7. [Shape optimization and mechanical properties analysis of the free-form surface](https://www.nature.com/articles/s41598-025-07440-5)

Authors: Cui Guoyong et al.

### 8. [A hybrid filtering and deep learning approach for early Alzheimer’s disease identification](https://www.nature.com/articles/s41598-025-03472-z)

Authors: Md. Khabir Uddin Ahamed et al.

### 9. [Democratizing cost-effective, agentic artificial intelligence to multilingual medical summarization through knowledge distillation](https://www.nature.com/articles/s41598-025-10451-x)

Authors: Chanseo Lee et al.

### 10. [Lightweight underwater object detection method based on multi-scale edge information selection](https://www.nature.com/articles/s41598-025-13566-3)

Authors: Shaobin Cai et al.

### 11. [BLTTNet: feature fusion based on BiLSTM-Transfomer-TCN for prediction of remaining useful life of aircraft engines](https://www.nature.com/articles/s41598-025-13387-4)

Authors: Yixu Yang et al.

### 12. [Entanglement-induced provable and robust quantum learning advantages](https://www.nature.com/articles/s41534-025-01078-x)

Authors: Haimeng Zhao et al.

### 13. [A novel encrypted traffic detection model based on detachable convolutional GCN-LSTM](https://www.nature.com/articles/s41598-025-13397-2)

Authors: Xiaogang Yuan et al.

### 14. [Sign language recognition based on dual-channel star-attention convolutional neural network](https://www.nature.com/articles/s41598-025-13625-9)

Authors: Jing Qin et al.

