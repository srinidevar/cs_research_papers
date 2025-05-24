# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-23 18:50:43.958590 PST.

### Artificial Intelligence

### 1. [TrialPanorama: Database and Benchmark for Systematic Review and Design of Clinical Trials](http://arxiv.org/pdf/2505.16097v1)

Authors: Zifeng Wang, Qiao Jin, Jiacheng Lin, Junyi Gao, Jathurshan Pradeepkumar, Pengcheng Jiang, Benjamin Danek, Zhiyong Lu, Jimeng Sun

Developing artificial intelligence (AI) for vertical domains requires a solid
data foundation for both training and evaluation. In this work, we introduce
TrialPanorama, a large-scale, structured database comprising 1,657,476 clinical
trial records aggregated from 15 global sources. The database captures key
aspects of trial design and execution, including trial setups, interventions,
conditions, biomarkers, and outcomes, and links them to standard biomedical
ontologies such as DrugBank and MedDRA. This structured and ontology-grounded
design enables TrialPanorama to serve as a unified, extensible resource for a
wide range of clinical trial tasks, including trial planning, design, and
summarization. To demonstrate its utility, we derive a suite of benchmark tasks
directly from the TrialPanorama database. The benchmark spans eight tasks
across two categories: three for systematic review (study search, study
screening, and evidence summarization) and five for trial design (arm design,
eligibility criteria, endpoint selection, sample size estimation, and trial
completion assessment). The experiments using five state-of-the-art large
language models (LLMs) show that while general-purpose LLMs exhibit some
zero-shot capability, their performance is still inadequate for high-stakes
clinical trial workflows. We release TrialPanorama database and the benchmark
to facilitate further research on AI for clinical trials.

### 2. [Logic-of-Thought: Empowering Large Language Models with Logic Programs for Solving Puzzles in Natural Language](http://arxiv.org/pdf/2505.16114v1)

Authors: Naiqi Li, Peiyuan Liu, Zheng Liu, Tao Dai, Yong Jiang, Shu-Tao Xia

Solving puzzles in natural language poses a long-standing challenge in AI.
While large language models (LLMs) have recently shown impressive capabilities
in a variety of tasks, they continue to struggle with complex puzzles that
demand precise reasoning and exhaustive search. In this paper, we propose
Logic-of-Thought (Logot), a novel framework that bridges LLMs with logic
programming to address this problem. Our method leverages LLMs to translate
puzzle rules and states into answer set programs (ASPs), the solution of which
are then accurately and efficiently inferred by an ASP interpreter. This hybrid
approach combines the natural language understanding of LLMs with the precise
reasoning capabilities of logic programs. We evaluate our method on various
grid puzzles and dynamic puzzles involving actions, demonstrating near-perfect
accuracy across all tasks. Our code and data are available at:
https://github.com/naiqili/Logic-of-Thought.

### 3. [LLM-Powered AI Agent Systems and Their Applications in Industry](http://arxiv.org/pdf/2505.16120v1)

Authors: Guannan Liang, Qianqian Tong

The emergence of Large Language Models (LLMs) has reshaped agent systems.
Unlike traditional rule-based agents with limited task scope, LLM-powered
agents offer greater flexibility, cross-domain reasoning, and natural language
interaction. Moreover, with the integration of multi-modal LLMs, current agent
systems are highly capable of processing diverse data modalities, including
text, images, audio, and structured tabular data, enabling richer and more
adaptive real-world behavior. This paper comprehensively examines the evolution
of agent systems from the pre-LLM era to current LLM-powered architectures. We
categorize agent systems into software-based, physical, and adaptive hybrid
systems, highlighting applications across customer service, software
development, manufacturing automation, personalized education, financial
trading, and healthcare. We further discuss the primary challenges posed by
LLM-powered agents, including high inference latency, output uncertainty, lack
of evaluation metrics, and security vulnerabilities, and propose potential
solutions to mitigate these concerns.

### 4. [Losing is for Cherishing: Data Valuation Based on Machine Unlearning and Shapley Value](http://arxiv.org/pdf/2505.16147v1)

Authors: Le Ma, Shirao Yang, Zihao Wang, Yinggui Wang, Lei Wang, Tao Wei, Kejun Zhang

The proliferation of large models has intensified the need for efficient data
valuation methods to quantify the contribution of individual data providers.
Traditional approaches, such as game-theory-based Shapley value and
influence-function-based techniques, face prohibitive computational costs or
require access to full data and model training details, making them hardly
achieve partial data valuation. To address this, we propose Unlearning Shapley,
a novel framework that leverages machine unlearning to estimate data values
efficiently. By unlearning target data from a pretrained model and measuring
performance shifts on a reachable test set, our method computes Shapley values
via Monte Carlo sampling, avoiding retraining and eliminating dependence on
full data. Crucially, Unlearning Shapley supports both full and partial data
valuation, making it scalable for large models (e.g., LLMs) and practical for
data markets. Experiments on benchmark datasets and large-scale text corpora
demonstrate that our approach matches the accuracy of state-of-the-art methods
while reducing computational overhead by orders of magnitude. Further analysis
confirms a strong correlation between estimated values and the true impact of
data subsets, validating its reliability in real-world scenarios. This work
bridges the gap between data valuation theory and practical deployment,
offering a scalable, privacy-compliant solution for modern AI ecosystems.

### 5. [Velocity Completion Task and Method for Event-based Player Positional Data in Soccer](http://arxiv.org/pdf/2505.16199v1)

Authors: Rikuhei Umemoto, Keisuke Fujii

In many real-world complex systems, the behavior can be observed as a
collection of discrete events generated by multiple interacting agents.
Analyzing the dynamics of these multi-agent systems, especially team sports,
often relies on understanding the movement and interactions of individual
agents. However, while providing valuable snapshots, event-based positional
data typically lacks the continuous temporal information needed to directly
calculate crucial properties such as velocity. This absence severely limits the
depth of dynamic analysis, preventing a comprehensive understanding of
individual agent behaviors and emergent team strategies. To address this
challenge, we propose a new method to simultaneously complete the velocity of
all agents using only the event-based positional data from team sports. Based
on this completed velocity information, we investigate the applicability of
existing team sports analysis and evaluation methods. Experiments using soccer
event data demonstrate that neural network-based approaches outperformed
rule-based methods regarding velocity completion error, considering the
underlying temporal dependencies and graph structure of player-to-player or
player-to-ball interaction. Moreover, the space evaluation results obtained
using the completed velocity are closer to those derived from complete tracking
data, highlighting our method's potential for enhanced team sports system
analysis.

### 6. [LightRouter: Towards Efficient LLM Collaboration with Minimal Overhead](http://arxiv.org/pdf/2505.16221v1)

Authors: Yifan Zhang, Xinkui Zhao, Zuxin Wang, Guanjie Cheng, Yueshen Xu, Shuiguang Deng, Jianwei Yin

The rapid advancement of large language models has unlocked remarkable
capabilities across a diverse array of natural language processing tasks.
However, the considerable differences among available LLMs-in terms of cost,
performance, and computational demands-pose significant challenges for users
aiming to identify the most suitable model for specific tasks. In this work, we
present LightRouter, a novel framework designed to systematically select and
integrate a small subset of LLMs from a larger pool, with the objective of
jointly optimizing both task performance and cost efficiency. LightRouter
leverages an adaptive selection mechanism to identify models that require only
a minimal number of boot tokens, thereby reducing costs, and further employs an
effective integration strategy to combine their outputs. Extensive experiments
across multiple benchmarks demonstrate that LightRouter matches or outperforms
widely-used ensemble baselines, achieving up to a 25% improvement in accuracy.
Compared with leading high-performing models, LightRouter achieves comparable
performance while reducing inference costs by up to 27%. Importantly, our
framework operates without any prior knowledge of individual models and relies
exclusively on inexpensive, lightweight models. This work introduces a
practical approach for efficient LLM selection and provides valuable insights
into optimal strategies for model combination.

### 7. [MAPLE: Many-Shot Adaptive Pseudo-Labeling for In-Context Learning](http://arxiv.org/pdf/2505.16225v1)

Authors: Zihan Chen, Song Wang, Zhen Tan, Jundong Li, Cong Shen

In-Context Learning (ICL) empowers Large Language Models (LLMs) to tackle
diverse tasks by incorporating multiple input-output examples, known as
demonstrations, into the input of LLMs. More recently, advancements in the
expanded context windows of LLMs have led to many-shot ICL, which uses hundreds
of demonstrations and outperforms few-shot ICL, which relies on fewer examples.
However, this approach is often hindered by the high cost of obtaining large
amounts of labeled data. To address this challenge, we propose Many-Shot
Adaptive Pseudo-LabEling, namely MAPLE, a novel influence-based many-shot ICL
framework that utilizes pseudo-labeled samples to compensate for the lack of
label information. We first identify a subset of impactful unlabeled samples
and perform pseudo-labeling on them by querying LLMs. These pseudo-labeled
samples are then adaptively selected and tailored to each test query as input
to improve the performance of many-shot ICL, without significant labeling
costs. Extensive experiments on real-world datasets demonstrate the
effectiveness of our framework, showcasing its ability to enhance LLM
adaptability and performance with limited labeled data.

### 8. [No Black Boxes: Interpretable and Interactable Predictive Healthcare with Knowledge-Enhanced Agentic Causal Discovery](http://arxiv.org/pdf/2505.16288v1)

Authors: Xiaoxue Han, Pengfei Hu, Jun-En Ding, Chang Lu, Feng Liu, Yue Ning

Deep learning models trained on extensive Electronic Health Records (EHR)
data have achieved high accuracy in diagnosis prediction, offering the
potential to assist clinicians in decision-making and treatment planning.
However, these models lack two crucial features that clinicians highly value:
interpretability and interactivity. The ``black-box'' nature of these models
makes it difficult for clinicians to understand the reasoning behind
predictions, limiting their ability to make informed decisions. Additionally,
the absence of interactive mechanisms prevents clinicians from incorporating
their own knowledge and experience into the decision-making process. To address
these limitations, we propose II-KEA, a knowledge-enhanced agent-driven causal
discovery framework that integrates personalized knowledge databases and
agentic LLMs. II-KEA enhances interpretability through explicit reasoning and
causal analysis, while also improving interactivity by allowing clinicians to
inject their knowledge and experience through customized knowledge bases and
prompts. II-KEA is evaluated on both MIMIC-III and MIMIC-IV, demonstrating
superior performance along with enhanced interpretability and interactivity, as
evidenced by its strong results from extensive case studies.

### 9. [EquivPruner: Boosting Efficiency and Quality in LLM-Based Search via Action Pruning](http://arxiv.org/pdf/2505.16312v1)

Authors: Jiawei Liu, Qisi Chen, Jianshu Zhang, Quan Liu, Defu Lian

Large Language Models (LLMs) excel at complex reasoning through search
algorithms, yet current strategies often suffer from massive token consumption
due to redundant exploration of semantically equivalent steps. Existing
semantic similarity methods struggle to accurately identify such equivalence in
domain-specific contexts like mathematical reasoning. To address this, we
propose EquivPruner, a simple yet effective approach that identifies and prunes
semantically equivalent actions during LLM reasoning search. We also introduce
MathEquiv, the first dataset we created for mathematical statement equivalence,
which enables the training of a lightweight equivalence detector. Extensive
experiments across various models and tasks demonstrate that EquivPruner
significantly reduces token consumption, improving searching efficiency and
often bolstering reasoning accuracy. For instance, when applied to
Qwen2.5-Math-7B-Instruct on GSM8K, EquivPruner reduced token consumption by
48.1\% while also improving accuracy. Our code is available at
https://github.com/Lolo1222/EquivPruner.

### 10. [FREESON: Retriever-Free Retrieval-Augmented Reasoning via Corpus-Traversing MCTS](http://arxiv.org/pdf/2505.16409v1)

Authors: Chaeeun Kim, Seungone Kim

Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in
multi-step reasoning and calling search engines at appropriate steps. However,
existing retrieval-augmented reasoning approaches rely on separate retrieval
models, limiting the LRM's role in retrieval to deciding when to retrieve and
how to query. This separation not only increases hardware and operational costs
but also leads to errors in the retrieval process due to the representation
bottleneck, a phenomenon where the retriever's embedding space is not
expressive enough to meet the generator's requirements. To address this, we
shift our perspective from sequence-to-sequence matching to locating the
answer-containing paths within the corpus, and propose a novel framework called
FREESON (Retriever-FREE Retrieval-Augmented ReaSONing). This framework enables
LRMs to retrieve relevant knowledge on their own by acting as both a generator
and retriever. To achieve this, we introduce a variant of the MCTS algorithm
specialized for the retrieval task, which we call CT-MCTS (Corpus-Traversing
Monte Carlo Tree Search). In this algorithm, LRMs traverse through the corpus
toward answer-containing regions. Our results on five open-domain QA
benchmarks, including single-hop and multi-hop questions, show that FREESON
achieves an average improvement of 14.4% in EM and F1 over four multi-step
reasoning models with a separate retriever, and it also performs comparably to
the strongest baseline, surpassing it by 3% on PopQA and 2WikiMultihopQA.

### Hardware Architecture

### 1. [Cosmos: A CXL-Based Full In-Memory System for Approximate Nearest Neighbor Search](http://arxiv.org/pdf/2505.16096v1)

Authors: Seoyoung Ko, Hyunjeong Shim, Wanju Doh, Sungmin Yun, Jinin So, Yongsuk Kwon, Sang-Soo Park, Si-Dong Roh, Minyong Yoon, Taeksang Song, Jung Ho Ahn

Retrieval-Augmented Generation (RAG) is crucial for improving the quality of
large language models by injecting proper contexts extracted from external
sources. RAG requires high-throughput, low-latency Approximate Nearest Neighbor
Search (ANNS) over billion-scale vector databases. Conventional DRAM/SSD
solutions face capacity/latency limits, whereas specialized hardware or RDMA
clusters lack flexibility or incur network overhead. We present Cosmos,
integrating general-purpose cores within CXL memory devices for full ANNS
offload and introducing rank-level parallel distance computation to maximize
memory bandwidth. We also propose an adjacency-aware data placement that
balances search loads across CXL devices based on inter-cluster proximity.
Evaluations on SIFT1B and DEEP1B traces show that Cosmos achieves up to 6.72x
higher throughput than the baseline CXL system and 2.35x over a
state-of-the-art CXL-based solution, demonstrating scalability for RAG
pipelines.

### 2. [How to keep pushing ML accelerator performance? Know your rooflines!](http://arxiv.org/pdf/2505.16346v1)

Authors: Marian Verhelst, Luca Benini, Naveen Verma

The rapidly growing importance of Machine Learning (ML) applications, coupled
with their ever-increasing model size and inference energy footprint, has
created a strong need for specialized ML hardware architectures. Numerous ML
accelerators have been explored and implemented, primarily to increase
task-level throughput per unit area and reduce task-level energy consumption.
This paper surveys key trends toward these objectives for more efficient ML
accelerators and provides a unifying framework to understand how compute and
memory technologies/architectures interact to enhance system-level efficiency
and performance. To achieve this, the paper introduces an enhanced version of
the roofline model and applies it to ML accelerators as an effective tool for
understanding where various execution regimes fall within roofline bounds and
how to maximize performance and efficiency under the rooline. Key concepts are
illustrated with examples from state-of-the-art designs, with a view towards
open research opportunities to further advance accelerator performance.

### 3. [DAS-MP: Enabling High-Quality Macro Placement with Enhanced Dataflow Awareness](http://arxiv.org/pdf/2505.16445v1)

Authors: Xiaotian Zhao, Zixuan Li, Yichen Cai, Tianju Wang, Yushan Pan, Xinfei Guo

Dataflow is a critical yet underexplored factor in automatic macro placement,
which is becoming increasingly important for developing intelligent design
automation techniques that minimize reliance on manual adjustments and reduce
design iterations. Existing macro or mixed-size placers with dataflow awareness
primarily focus on intrinsic relationships among macros, overlooking the
crucial influence of standard cell clusters on macro placement. To address
this, we propose DAS-MP, which extracts hidden connections between macros and
standard cells and incorporates a series of algorithms to enhance dataflow
awareness, integrating them into placement constraints for improved macro
placement. To further optimize placement results, we introduce two fine-tuning
steps: (1) congestion optimization by taking macro area into consideration, and
(2) flipping decisions to determine the optimal macro orientation based on the
extracted dataflow information. By integrating enhanced dataflow awareness into
placement constraints and applying these fine-tuning steps, the proposed
approach achieves an average 7.9% improvement in half-perimeter wirelength
(HPWL) across multiple widely used benchmark designs compared to a
state-of-the-art dataflow-aware macro placer. Additionally, it significantly
improves congestion, reducing overflow by an average of 82.5%, and achieves
improvements of 36.97% in Worst Negative Slack (WNS) and 59.44% in Total
Negative Slack (TNS). The approach also maintains efficient runtime throughout
the entire placement process, incurring less than a 1.5% runtime overhead.
These results show that the proposed dataflow-driven methodology, combined with
the fine-tuning steps, provides an effective foundation for macro placement and
can be seamlessly integrated into existing design flows to enhance placement
quality.

### 4. [Advanced Integration Strategies for ESD Protection and Termination in High-Speed LVDS Systems](http://arxiv.org/pdf/2505.16200v1)

Authors: Kavya Gaddipati

This technical article explores comprehensive strategies for integrating
Electrostatic Discharge (ESD) protection diodes and termination resistors in
LowVoltage Differential Signaling (LVDS) designs. The article examines critical
aspects of protection mechanisms, design considerations, impedance matching,
and placement optimization techniques. Through detailed analysis of layout
considerations and advanced design strategies, the article presents solutions
for common integration challenges. It emphasizes the importance of signal
integrity maintenance and protection effectiveness while providing practical
guidelines for implementing robust LVDS systems. Various methodologies for
performance optimization and validation are discussed, offering designers a
thorough framework for creating reliable high-speed digital systems that
balance protection requirements with signal integrity demands.

### 5. [CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark](http://arxiv.org/pdf/2505.16968v1)

Authors: Ahmed Heakl, Sarim Hashmi, Gustavo Bertolo Stahl, Seung Hun Eddie Han, Salman Khan, Abdulrahman Mahmoud

We introduce \texttt{CASS}, the first large-scale dataset and model suite for
cross-architecture GPU code transpilation, targeting both source-level
(CUDA~$\leftrightarrow$~HIP) and assembly-level (Nvidia
SASS~$\leftrightarrow$~AMD RDNA3) translation. The dataset comprises 70k
verified code pairs across host and device, addressing a critical gap in
low-level GPU code portability. Leveraging this resource, we train the
\texttt{CASS} family of domain-specific language models, achieving 95\% source
translation accuracy and 37.5\% assembly translation accuracy, substantially
outperforming commercial baselines such as GPT-4o, Claude, and Hipify. Our
generated code matches native performance in over 85\% of test cases,
preserving runtime and memory behavior. To support rigorous evaluation, we
introduce \texttt{CASS-Bench}, a curated benchmark spanning 16 GPU domains with
ground-truth execution. All data, models, and evaluation tools are released as
open source to foster progress in GPU compiler tooling, binary compatibility,
and LLM-guided hardware translation. Dataset and benchmark are on
\href{https://huggingface.co/datasets/MBZUAI/cass}{\textcolor{blue}{HuggingFace}},
with code at
\href{https://github.com/GustavoStahl/CASS}{\textcolor{blue}{GitHub}}.

### Computational Complexity

### 1. [Maximum Separation of Quantum Communication Complexity With and Without Shared Entanglement](http://arxiv.org/pdf/2505.16457v1)

Authors: Atsuya Hasegawa, François Le Gall, Augusto Modanese

We present relation problems whose input size is $n$ such that they can be
solved with no communication for entanglement-assisted quantum communication
models, but require $\Omega(n)$ qubit communication for $2$-way quantum
communication models without prior shared entanglement. This is the maximum
separation of quantum communication complexity with and without shared
entanglement. To our knowledge, our result is the first lower bound on quantum
communication complexity without shared entanglement when the upper bound of
entanglement-assisted quantum communication models is zero. The problem we
consider is a parallel repetition of any non-local game which has a perfect
quantum strategy and no perfect classical strategy, and for which a parallel
repetition theorem for the classical value holds with exponential decay.

### 2. [The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](http://arxiv.org/pdf/2505.16716v1)

Authors: Moritz Stargalla, Christoph Hertrich, Daniel Reichman

An established measure of the expressive power of a given ReLU neural network
is the number of linear regions into which it partitions the input space. There
exist many different, non-equivalent definitions of what a linear region
actually is. We systematically assess which papers use which definitions and
discuss how they relate to each other. We then analyze the computational
complexity of counting the number of such regions for the various definitions.
Generally, this turns out to be an intractable problem. We prove NP- and
#P-hardness results already for networks with one hidden layer and strong
hardness of approximation results for two or more hidden layers. Finally, on
the algorithmic side, we demonstrate that counting linear regions can at least
be achieved in polynomial space for some common definitions.

### Computational Engineering

### 1. [From Local Patterns to Global Understanding: Cross-Stock Trend Integration for Enhanced Predictive Modeling](http://arxiv.org/pdf/2505.16573v1)

Authors: Yi Hu, Hanchi Ren, Jingjing Deng, Xianghua Xie

Stock price prediction is a critical area of financial forecasting,
traditionally approached by training models using the historical price data of
individual stocks. While these models effectively capture single-stock
patterns, they fail to leverage potential correlations among stock trends,
which could improve predictive performance. Current single-stock learning
methods are thus limited in their ability to provide a broader understanding of
price dynamics across multiple stocks. To address this, we propose a novel
method that merges local patterns into a global understanding through
cross-stock pattern integration. Our strategy is inspired by Federated Learning
(FL), a paradigm designed for decentralized model training. FL enables
collaborative learning across distributed datasets without sharing raw data,
facilitating the aggregation of global insights while preserving data privacy.
In our adaptation, we train models on individual stock data and iteratively
merge them to create a unified global model. This global model is subsequently
fine-tuned on specific stock data to retain local relevance. The proposed
strategy enables parallel training of individual stock models, facilitating
efficient utilization of computational resources and reducing overall training
time. We conducted extensive experiments to evaluate the proposed method,
demonstrating that it outperforms benchmark models and enhances the predictive
capabilities of state-of-the-art approaches. Our results highlight the efficacy
of Cross-Stock Trend Integration (CSTI) in advancing stock price prediction,
offering a robust alternative to traditional single-stock learning
methodologies.

### 2. [Advanced Integration Strategies for ESD Protection and Termination in High-Speed LVDS Systems](http://arxiv.org/pdf/2505.16200v1)

Authors: Kavya Gaddipati

This technical article explores comprehensive strategies for integrating
Electrostatic Discharge (ESD) protection diodes and termination resistors in
LowVoltage Differential Signaling (LVDS) designs. The article examines critical
aspects of protection mechanisms, design considerations, impedance matching,
and placement optimization techniques. Through detailed analysis of layout
considerations and advanced design strategies, the article presents solutions
for common integration challenges. It emphasizes the importance of signal
integrity maintenance and protection effectiveness while providing practical
guidelines for implementing robust LVDS systems. Various methodologies for
performance optimization and validation are discussed, offering designers a
thorough framework for creating reliable high-speed digital systems that
balance protection requirements with signal integrity demands.

### 3. [A finite element solver for a thermodynamically consistent electrolyte model](http://arxiv.org/pdf/2505.16296v1)

Authors: Jan Habscheid, Satyvir Singh, Lambert Theisen, Stefanie Braun, Manuel Torrilhon

In this study, we present a finite element solver for a thermodynamically
consistent electrolyte model that accurately captures multicomponent ionic
transport by incorporating key physical phenomena such as steric effects,
solvation, and pressure coupling. The model is rooted in the principles of
non-equilibrium thermodynamics and strictly enforces mass conservation, charge
neutrality, and entropy production. It extends beyond classical frameworks like
the Nernst-Planck system by employing modified partial mass balances, the
electrostatic Poisson equation, and a momentum balance expressed in terms of
electrostatic potential, atomic fractions, and pressure, thereby enhancing
numerical stability and physical consistency. Implemented using the FEniCSx
platform, the solver efficiently handles one- and two-dimensional problems with
varied boundary conditions and demonstrates excellent convergence behavior and
robustness. Validation against benchmark problems confirms its improved
physical fidelity, particularly in regimes characterized by high ionic
concentrations and strong electrochemical gradients. Simulation results reveal
critical electrolyte phenomena, including electric double layer formation,
rectification behavior, and the effects of solvation number, Debye length, and
compressibility. The solver's modular variational formulation facilitates its
extension to complex electrochemical systems involving multiple ionic species
with asymmetric valences.

### Computation and Language

### 1. [Continually Self-Improving Language Models for Bariatric Surgery Question--Answering](http://arxiv.org/pdf/2505.16102v1)

Authors: Yash Kumar Atri, Thomas H Shin, Thomas Hartvigsen

While bariatric and metabolic surgery (MBS) is considered the gold standard
treatment for severe and morbid obesity, its therapeutic efficacy hinges upon
active and longitudinal engagement with multidisciplinary providers, including
surgeons, dietitians/nutritionists, psychologists, and endocrinologists. This
engagement spans the entire patient journey, from preoperative preparation to
long-term postoperative management. However, this process is often hindered by
numerous healthcare disparities, such as logistical and access barriers, which
impair easy patient access to timely, evidence-based, clinician-endorsed
information. To address these gaps, we introduce bRAGgen, a novel adaptive
retrieval-augmented generation (RAG)-based model that autonomously integrates
real-time medical evidence when response confidence dips below dynamic
thresholds. This self-updating architecture ensures that responses remain
current and accurate, reducing the risk of misinformation. Additionally, we
present bRAGq, a curated dataset of 1,302 bariatric surgery--related questions,
validated by an expert bariatric surgeon. bRAGq constitutes the first
large-scale, domain-specific benchmark for comprehensive MBS care. In a
two-phase evaluation, bRAGgen is benchmarked against state-of-the-art models
using both large language model (LLM)--based metrics and expert surgeon review.
Across all evaluation dimensions, bRAGgen demonstrates substantially superior
performance in generating clinically accurate and relevant responses.

### 2. [MPL: Multiple Programming Languages with Large Language Models for Information Extraction](http://arxiv.org/pdf/2505.16107v1)

Authors: Bo Li, Gexiang Fang, Wei Ye, Zhenghua Xu, Jinglei Zhang, Hao Cheng, Shikun Zhang

Recent research in information extraction (IE) focuses on utilizing
code-style inputs to enhance structured output generation. The intuition behind
this is that the programming languages (PLs) inherently exhibit greater
structural organization than natural languages (NLs). This structural advantage
makes PLs particularly suited for IE tasks. Nevertheless, existing research
primarily focuses on Python for code-style simulation, overlooking the
potential of other widely-used PLs (e.g., C++ and Java) during the supervised
fine-tuning (SFT) phase. In this research, we propose \textbf{M}ultiple
\textbf{P}rogramming \textbf{L}anguages with large language models for
information extraction (abbreviated as \textbf{MPL}), a novel framework that
explores the potential of incorporating different PLs in the SFT phase.
Additionally, we introduce \texttt{function-prompt} with virtual running to
simulate code-style inputs more effectively and efficiently. Experimental
results on a wide range of datasets demonstrate the effectiveness of MPL.
Furthermore, we conduct extensive experiments to provide a comprehensive
analysis. We have released our code for future research.

### 3. [KoBALT: Korean Benchmark For Advanced Linguistic Tasks](http://arxiv.org/pdf/2505.16125v1)

Authors: Hyopil Shin, Sangah Lee, Dongjun Jang, Wooseok Song, Jaeyoon Kim, Chaeyoung Oh, Hyemi Jo, Youngchae Ahn, Sihyun Oh, Hyohyeong Chang, Sunkyoung Kim, Jinsik Lee

We introduce KoBALT (Korean Benchmark for Advanced Linguistic Tasks), a
comprehensive linguistically-motivated benchmark comprising 700 multiple-choice
questions spanning 24 phenomena across five linguistic domains: syntax,
semantics, pragmatics, phonetics/phonology, and morphology. KoBALT is designed
to advance the evaluation of large language models (LLMs) in Korean, a
morphologically rich language, by addressing the limitations of conventional
benchmarks that often lack linguistic depth and typological grounding. It
introduces a suite of expert-curated, linguistically motivated questions with
minimal n-gram overlap with standard Korean corpora, substantially mitigating
the risk of data contamination and allowing a more robust assessment of true
language understanding. Our evaluation of 20 contemporary LLMs reveals
significant performance disparities, with the highest-performing model
achieving 61\% general accuracy but showing substantial variation across
linguistic domains - from stronger performance in semantics (66\%) to
considerable weaknesses in phonology (31\%) and morphology (36\%). Through
human preference evaluation with 95 annotators, we demonstrate a strong
correlation between KoBALT scores and human judgments, validating our
benchmark's effectiveness as a discriminative measure of Korean language
understanding. KoBALT addresses critical gaps in linguistic evaluation for
typologically diverse languages and provides a robust framework for assessing
genuine linguistic competence in Korean language models.

### 4. [Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](http://arxiv.org/pdf/2505.16128v1)

Authors: Yue Zhou, Barbara Di Eugenio

Despite LLMs' explicit alignment against demographic stereotypes, they have
been shown to exhibit biases under various social contexts. In this work, we
find that LLMs exhibit concerning biases in how they associate solution
veracity with demographics. Through experiments across five human value-aligned
LLMs on mathematics, coding, commonsense, and writing problems, we reveal two
forms of such veracity biases: Attribution Bias, where models
disproportionately attribute correct solutions to certain demographic groups,
and Evaluation Bias, where models' assessment of identical solutions varies
based on perceived demographic authorship. Our results show pervasive biases:
LLMs consistently attribute fewer correct solutions and more incorrect ones to
African-American groups in math and coding, while Asian authorships are least
preferred in writing evaluation. In additional studies, we show LLMs
automatically assign racially stereotypical colors to demographic groups in
visualization code, suggesting these biases are deeply embedded in models'
reasoning processes. Our findings indicate that demographic bias extends beyond
surface-level stereotypes and social context provocations, raising concerns
about LLMs' deployment in educational and evaluation settings.

### 5. [LLMs Are Not Scorers: Rethinking MT Evaluation with Generation-Based Methods](http://arxiv.org/pdf/2505.16129v1)

Authors: Hyang Cui

Recent studies have applied large language models (LLMs) to machine
translation quality estimation (MTQE) by prompting models to assign numeric
scores. Nonetheless, these direct scoring methods tend to show low
segment-level correlation with human judgments. In this paper, we propose a
generation-based evaluation paradigm that leverages decoder-only LLMs to
produce high-quality references, followed by semantic similarity scoring using
sentence embeddings. We conduct the most extensive evaluation to date in MTQE,
covering 8 LLMs and 8 language pairs. Empirical results show that our method
outperforms both intra-LLM direct scoring baselines and external non-LLM
reference-free metrics from MTME. These findings demonstrate the strength of
generation-based evaluation and support a shift toward hybrid approaches that
combine fluent generation with accurate semantic assessment.

### 6. [Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning](http://arxiv.org/pdf/2505.16142v1)

Authors: Shicheng Xu, Liang Pang, Yunchang Zhu, Jia Gu, Zihao Wei, Jingcheng Deng, Feiyang Pan, Huawei Shen, Xueqi Cheng

Distilling reasoning paths from teacher to student models via supervised
fine-tuning (SFT) provides a shortcut for improving the reasoning ability of
smaller Large Language Models (LLMs). However, the reasoning paths generated by
teacher models often reflect only surface-level traces of their underlying
authentic reasoning. Insights from cognitive neuroscience suggest that
authentic reasoning involves a complex interweaving between meta-reasoning
(which selects appropriate sub-problems from multiple candidates) and solving
(which addresses the sub-problem). This implies authentic reasoning has an
implicit multi-branch structure. Supervised fine-tuning collapses this rich
structure into a flat sequence of token prediction in the teacher's reasoning
path, preventing effective distillation of this structure to students. To
address this limitation, we propose RLKD, a reinforcement learning (RL)-based
distillation framework guided by a novel Generative Structure Reward Model
(GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving
steps and computes rewards to measure structural alignment between student and
teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to
internalize the teacher's implicit multi-branch reasoning structure rather than
merely mimicking fixed output paths. Experiments show RLKD surpasses standard
SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime,
unlocking greater student reasoning potential than SFT-based distillation.

### 7. [EduBench: A Comprehensive Benchmarking Dataset for Evaluating Large Language Models in Diverse Educational Scenarios](http://arxiv.org/pdf/2505.16160v1)

Authors: Bin Xu, Yu Bai, Huashan Sun, Yiguan Lin, Siming Liu, Xinyue Liang, Yaolin Li, Yang Gao, Heyan Huang

As large language models continue to advance, their application in
educational contexts remains underexplored and under-optimized. In this paper,
we address this gap by introducing the first diverse benchmark tailored for
educational scenarios, incorporating synthetic data containing 9 major
scenarios and over 4,000 distinct educational contexts. To enable comprehensive
assessment, we propose a set of multi-dimensional evaluation metrics that cover
12 critical aspects relevant to both teachers and students. We further apply
human annotation to ensure the effectiveness of the model-generated evaluation
responses. Additionally, we succeed to train a relatively small-scale model on
our constructed dataset and demonstrate that it can achieve performance
comparable to state-of-the-art large models (e.g., Deepseek V3, Qwen Max) on
the test set. Overall, this work provides a practical foundation for the
development and evaluation of education-oriented language models. Code and data
are released at https://github.com/ybai-nlp/EduBench.

### 8. [KNN-SSD: Enabling Dynamic Self-Speculative Decoding via Nearest Neighbor Layer Set Optimization](http://arxiv.org/pdf/2505.16162v1)

Authors: Mingbo Song, Heming Xia, Jun Zhang, Chak Tou Leong, Qiancheng Xu, Wenjie Li, Sujian Li

Speculative Decoding (SD) has emerged as a widely used paradigm to accelerate
the inference of large language models (LLMs) without compromising generation
quality. It works by efficiently drafting multiple tokens using a compact model
and then verifying them in parallel using the target LLM. Notably,
Self-Speculative Decoding proposes skipping certain layers to construct the
draft model, which eliminates the need for additional parameters or training.
Despite its strengths, we observe in this work that drafting with layer
skipping exhibits significant sensitivity to domain shifts, leading to a
substantial drop in acceleration performance. To enhance the domain
generalizability of this paradigm, we introduce KNN-SSD, an algorithm that
leverages K-Nearest Neighbor (KNN) search to match different skipped layers
with various domain inputs. We evaluated our algorithm in various models and
multiple tasks, observing that its application leads to 1.3x-1.6x speedup in
LLM inference.

### 9. [Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task](http://arxiv.org/pdf/2505.16164v1)

Authors: Mengyang Qiu, Zoe Brisebois, Siena Sun

Large language models (LLMs) are increasingly explored as substitutes for
human participants in cognitive tasks, but their ability to simulate human
behavioral variability remains unclear. This study examines whether LLMs can
approximate individual differences in the phonemic fluency task, where
participants generate words beginning with a target letter. We evaluated 34
model configurations, varying prompt specificity, sampling temperature, and
model type, and compared outputs to responses from 106 human participants.
While some configurations, especially Claude 3.7 Sonnet, matched human averages
and lexical preferences, none reproduced the scope of human variability. LLM
outputs were consistently less diverse and structurally rigid, and LLM
ensembles failed to increase diversity. Network analyses further revealed
fundamental differences in retrieval structure between humans and models. These
results highlight key limitations in using LLMs to simulate human cognition and
behavior.

### 10. [When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction](http://arxiv.org/pdf/2505.16170v1)

Authors: Yuqing Yang, Robin Jia

Can large language models (LLMs) admit their mistakes when they should know
better? In this work, we define the behavior of acknowledging errors in
previously generated answers as "retraction" and aim to understand when and why
LLMs choose to retract. We first construct model-specific datasets to evaluate
whether a model will retract an incorrect answer that contradicts its own
parametric knowledge. While LLMs are capable of retraction, they do so only
infrequently. We demonstrate that retraction is closely tied to previously
identified indicators of models' internal belief: models fail to retract wrong
answers that they "believe" to be factually correct. Steering experiments
further demonstrate that internal belief causally influences model retraction.
In particular, when the model does not believe its answer, this not only
encourages the model to attempt to verify the answer, but also alters attention
behavior during self-verification. Finally, we demonstrate that simple
supervised fine-tuning significantly improves retraction performance by helping
the model learn more accurate internal beliefs. Code and datasets are available
on https://github.com/ayyyq/llm-retraction.

### Cryptography and Security

### 1. [Extensible Post Quantum Cryptography Based Authentication](http://arxiv.org/pdf/2505.16112v1)

Authors: Homer A. Riva-Cambrin, Rahul Singh, Sanju Lama, Garnette R. Sutherland

Cryptography underpins the security of modern digital infrastructure, from
cloud services to health data. However, many widely deployed systems will
become vulnerable after the advent of scalable quantum computing. Although
quantum-safe cryptographic primitives have been developed, such as
lattice-based digital signature algorithms (DSAs) and key encapsulation
mechanisms (KEMs), their unique structural and performance characteristics make
them unsuitable for existing protocols. In this work, we introduce a
quantum-safe single-shot protocol for machine-to-machine authentication and
authorization that is specifically designed to leverage the strengths of
lattice-based DSAs and KEMs. Operating entirely over insecure channels, this
protocol enables the forward-secure establishment of tokens in constrained
environments. By demonstrating how new quantum-safe cryptographic primitives
can be incorporated into secure systems, this study lays the groundwork for
scalable, resilient, and future-proof identity infrastructures in a
quantum-enabled world.

### 2. [Outsourcing SAT-based Verification Computations in Network Security](http://arxiv.org/pdf/2505.16137v1)

Authors: Qi Duan, Ehab Al-Shaer

The emergence of cloud computing gives huge impact on large computations.
Cloud computing platforms offer servers with large computation power to be
available for customers. These servers can be used efficiently to solve
problems that are complex by nature, for example, satisfiability (SAT)
problems. Many practical problems can be converted to SAT, for example, circuit
verification and network configuration analysis. However, outsourcing SAT
instances to the servers may cause data leakage that can jeopardize system's
security. Before
  outsourcing the SAT instance, one needs to hide the input information. One
way to preserve privacy and hide information is to randomize the SAT
  instance before outsourcing. In this paper, we present multiple novel methods
to randomize SAT instances. We present a novel method to randomize the SAT
instance, a variable randomization method to randomize the solution set, and
methods to randomize Mincost SAT and MAX3SAT instances. Our analysis and
evaluation show the correctness and feasibility of these randomization methods.
The scalability and generality of our methods make it applicable for real world
problems.

### 3. [VIVID: A Novel Approach to Remediation Prioritization in Static Application Security Testing (SAST)](http://arxiv.org/pdf/2505.16205v1)

Authors: Naeem Budhwani, Mohammad Faghani, Hayden Richard

Static Application Security Testing (SAST) enables organizations to detect
vulnerabilities in code early; however, major SAST platforms do not include
visual aids and present little insight on correlations between tainted data
chains. We propose VIVID - Vulnerability Information Via Data flow - a novel
method to extract and consume SAST insights, which is to graph the
application's vulnerability data flows (VDFs) and carry out graph theory
analysis on the resulting VDF directed graph. Nine metrics were assessed to
evaluate their effectiveness in analyzing the VDF graphs of deliberately
insecure web applications. These metrics include 3 centrality metrics, 2
structural metrics, PageRank, in-degree, out-degree, and cross-clique
connectivity. We present simulations that find that out-degree, betweenness
centrality, in-eigenvector centrality, and cross-clique connectivity were found
to be associated with files exhibiting high vulnerability traffic, making them
refactoring candidates where input sanitization may have been missed.
Meanwhile, out-eigenvector centrality, PageRank, and in-degree were found to be
associated with nodes enabling vulnerability flow and sinks, but not
necessarily where input validation should be placed. This is a novel method to
automatically provide development teams an evidence-based prioritized list of
files to embed security controls into, informed by vulnerability propagation
patterns in the application architecture.

### 4. [Verifying Differentially Private Median Estimation](http://arxiv.org/pdf/2505.16246v1)

Authors: Hyukjun Kwon, Chenglin Fan

Differential Privacy (DP) is a robust privacy guarantee that is widely
employed in private data analysis today, finding broad application in domains
such as statistical query release and machine learning. However, DP achieves
privacy by introducing noise into data or query answers, which malicious actors
could exploit during analysis. To address this concern, we propose the first
verifiable differentially private median estimation scheme based on zk-SNARKs.
Our scheme combines the exponential mechanism and a utility function for median
estimation into an arithmetic circuit, leveraging a scaled version of the
inverse cumulative distribution function (CDF) method for precise sampling from
the distribution derived from the utility function. This approach not only
ensures privacy but also provides a mechanism to verify that the algorithm
achieves DP guarantees without revealing sensitive information in the process.

### 5. [Interpretable Anomaly Detection in Encrypted Traffic Using SHAP with Machine Learning Models](http://arxiv.org/pdf/2505.16261v1)

Authors: Kalindi Singh, Aayush Kashyap, Aswani Kumar Cherukuri

The widespread adoption of encrypted communication protocols such as HTTPS
and TLS has enhanced data privacy but also rendered traditional anomaly
detection techniques less effective, as they often rely on inspecting
unencrypted payloads. This study aims to develop an interpretable machine
learning-based framework for anomaly detection in encrypted network traffic.
This study proposes a model-agnostic framework that integrates multiple machine
learning classifiers, with SHapley Additive exPlanations SHAP to ensure
post-hoc model interpretability. The models are trained and evaluated on three
benchmark encrypted traffic datasets. Performance is assessed using standard
classification metrics, and SHAP is used to explain model predictions by
attributing importance to individual input features. SHAP visualizations
successfully revealed the most influential traffic features contributing to
anomaly predictions, enhancing the transparency and trustworthiness of the
models. Unlike conventional approaches that treat machine learning as a black
box, this work combines robust classification techniques with explainability
through SHAP, offering a novel interpretable anomaly detection system tailored
for encrypted traffic environments. While the framework is generalizable,
real-time deployment and performance under adversarial conditions require
further investigation. Future work may explore adaptive models and real-time
interpretability in operational network environments. This interpretable
anomaly detection framework can be integrated into modern security operations
for encrypted environments, allowing analysts not only to detect anomalies with
high precision but also to understand why a model made a particular decision a
crucial capability in compliance-driven and mission-critical settings.

### 6. [Poster: Towards an Automated Security Testing Framework for Industrial UEs](http://arxiv.org/pdf/2505.16300v1)

Authors: Sotiris Michaelides, Daniel Eguiguren Chavez, Martin Henze

With the ongoing adoption of 5G for communication in industrial systems and
critical infrastructure, the security of industrial UEs such as 5G-enabled
industrial robots becomes an increasingly important topic. Most notably, to
meet the stringent security requirements of industrial deployments, industrial
UEs not only have to fully comply with the 5G specifications but also implement
and use correctly secure communication protocols such as TLS. To ensure the
security of industrial UEs, operators of industrial 5G networks rely on
security testing before deploying new devices to their production networks.
However, currently only isolated tests for individual security aspects of
industrial UEs exist, severely hindering comprehensive testing. In this paper,
we report on our ongoing efforts to alleviate this situation by creating an
automated security testing framework for industrial UEs to comprehensively
evaluate their security posture before deployment. With this framework, we aim
to provide stakeholders with a fully automated-method to verify that
higher-layer security protocols are correctly implemented, while simultaneously
ensuring that the UE's protocol stack adheres to 3GPP specifications.

### 7. [ReCopilot: Reverse Engineering Copilot in Binary Analysis](http://arxiv.org/pdf/2505.16366v1)

Authors: Guoqiang Chen, Huiqi Sun, Daguang Liu, Zhiqi Wang, Qiang Wang, Bin Yin, Lu Liu, Lingyun Ying

Binary analysis plays a pivotal role in security domains such as malware
detection and vulnerability discovery, yet it remains labor-intensive and
heavily reliant on expert knowledge. General-purpose large language models
(LLMs) perform well in programming analysis on source code, while
binaryspecific LLMs are underexplored. In this work, we present ReCopilot, an
expert LLM designed for binary analysis tasks. ReCopilot integrates binary code
knowledge through a meticulously constructed dataset, encompassing continue
pretraining (CPT), supervised fine-tuning (SFT), and direct preference
optimization (DPO) stages. It leverages variable data flow and call graph to
enhance context awareness and employs test-time scaling to improve reasoning
capabilities. Evaluations on a comprehensive binary analysis benchmark
demonstrate that ReCopilot achieves state-of-the-art performance in tasks such
as function name recovery and variable type inference on the decompiled pseudo
code, outperforming both existing tools and LLMs by 13%. Our findings highlight
the effectiveness of domain-specific training and context enhancement, while
also revealing challenges in building super long chain-of-thought. ReCopilot
represents a significant step toward automating binary analysis with
interpretable and scalable AI assistance in this domain.

### 8. [Privacy-Aware Cyberterrorism Network Analysis using Graph Neural Networks and Federated Learning](http://arxiv.org/pdf/2505.16371v1)

Authors: Anas Ali, Mubashar Husain, Peter Hans

Cyberterrorism poses a formidable threat to digital infrastructures, with
increasing reliance on encrypted, decentralized platforms that obscure threat
actor activity. To address the challenge of analyzing such adversarial networks
while preserving the privacy of distributed intelligence data, we propose a
Privacy-Aware Federated Graph Neural Network (PA-FGNN) framework. PA-FGNN
integrates graph attention networks, differential privacy, and homomorphic
encryption into a robust federated learning pipeline tailored for
cyberterrorism network analysis. Each client trains locally on sensitive graph
data and exchanges encrypted, noise-perturbed model updates with a central
aggregator, which performs secure aggregation and broadcasts global updates. We
implement anomaly detection for flagging high-risk nodes and incorporate
defenses against gradient poisoning. Experimental evaluations on simulated dark
web and cyber-intelligence graphs demonstrate that PA-FGNN achieves over 91\%
classification accuracy, maintains resilience under 20\% adversarial client
behavior, and incurs less than 18\% communication overhead. Our results
highlight that privacy-preserving GNNs can support large-scale cyber threat
detection without compromising on utility, privacy, or robustness.

### 9. [Consistent and Compatible Modelling of Cyber Intrusions and Incident Response Demonstrated in the Context of Malware Attacks on Critical Infrastructure](http://arxiv.org/pdf/2505.16398v1)

Authors: Peter Maynard, Yulia Cherdantseva, Avi Shaked, Pete Burnap, Arif Mehmood

Cyber Security Incident Response (IR) Playbooks are used to capture the steps
required to recover from a cyber intrusion. Individual IR playbooks should
focus on a specific type of incident and be aligned with the architecture of a
system under attack. Intrusion modelling focuses on a specific potential cyber
intrusion and is used to identify where and what countermeasures are needed,
and the resulting intrusion models are expected to be used in effective IR,
ideally by feeding IR Playbooks designs. IR playbooks and intrusion models,
however, are created in isolation and at varying stages of the system's
lifecycle. We take nine critical national infrastructure intrusion models -
expressed using Sequential AND Attack Trees - and transform them into models of
the same format as IR playbooks. We use Security Modelling Framework for
modelling attacks and playbooks, and for demonstrating the feasibility of the
better integration between risk assessment and IR at the modelling level. This
results in improved intrusion models and tighter coupling between IR playbooks
and threat modelling which - as we demonstrate - yields novel insights into the
analysis of attacks and response actions. The main contributions of this paper
are (a) a novel way of representing attack trees using the Security Modelling
Framework,(b) a new tool for converting Sequential AND attack trees into models
compatible with playbooks, and (c) the examples of nine intrusion models
represented using the Security Modelling Framework.

### 10. [Password Strength Detection via Machine Learning: Analysis, Modeling, and Evaluation](http://arxiv.org/pdf/2505.16439v1)

Authors: Jiazhi Mo, Hailu Kuang, Xiaoqi Li

As network security issues continue gaining prominence, password security has
become crucial in safeguarding personal information and network systems. This
study first introduces various methods for system password cracking, outlines
password defense strategies, and discusses the application of machine learning
in the realm of password security. Subsequently, we conduct a detailed public
password database analysis, uncovering standard features and patterns among
passwords. We extract multiple characteristics of passwords, including length,
the number of digits, the number of uppercase and lowercase letters, and the
number of special characters. We then experiment with six different machine
learning algorithms: support vector machines, logistic regression, neural
networks, decision trees, random forests, and stacked models, evaluating each
model's performance based on various metrics, including accuracy, recall, and
F1 score through model validation and hyperparameter tuning. The evaluation
results on the test set indicate that decision trees and stacked models excel
in accuracy, recall, and F1 score, making them a practical option for the
strong and weak password classification task.

### Computer Vision and Pattern Recognition

### 1. [GMatch: Geometry-Constrained Feature Matching for RGB-D Object Pose Estimation](http://arxiv.org/pdf/2505.16144v1)

Authors: Ming Yang, Haoran Li

We present GMatch, a learning-free feature matcher designed for robust 6DoF
object pose estimation, addressing common local ambiguities in sparse feature
matching. Unlike traditional methods that rely solely on descriptor similarity,
GMatch performs a guided, incremental search, enforcing SE(3)-invariant
geometric consistency throughout the matching process. It leverages a provably
complete set of geometric features that uniquely determine 3D keypoint
configurations, ensuring globally consistent correspondences without the need
for training or GPU support. When combined with classical descriptors such as
SIFT, GMatch-SIFT forms a general-purpose pose estimation pipeline that offers
strong interpretability and generalization across diverse objects and scenes.
Experiments on the HOPE dataset show that GMatch outperforms both traditional
and learning-based matchers, with GMatch-SIFT achieving or surpassing the
performance of instance-level pose networks. On the YCB-Video dataset,
GMatch-SIFT demonstrates high accuracy and low variance on texture-rich
objects. These results not only validate the effectiveness of GMatch-SIFT for
object pose estimation but also highlight the broader applicability of GMatch
as a general-purpose feature matcher. Code will be released upon acceptance.

### 2. [Training-Free Reasoning and Reflection in MLLMs](http://arxiv.org/pdf/2505.16151v1)

Authors: Hongchen Wei, Zhenzhong Chen

Recent advances in Reasoning LLMs (e.g., DeepSeek-R1 and OpenAI-o1) have
showcased impressive reasoning capabilities via reinforcement learning.
However, extending these capabilities to Multimodal LLMs (MLLMs) is hampered by
the prohibitive costs of retraining and the scarcity of high-quality,
verifiable multimodal reasoning datasets. This paper introduces FRANK Model, a
training-FRee ANd r1-liKe MLLM that imbues off-the-shelf MLLMs with reasoning
and reflection abilities, without any gradient updates or extra supervision.
Our key insight is to decouple perception and reasoning across MLLM decoder
layers. Specifically, we observe that compared to the deeper decoder layers,
the shallow decoder layers allocate more attention to visual tokens, while the
deeper decoder layers concentrate on textual semantics. This observation
motivates a hierarchical weight merging approach that combines a
visual-pretrained MLLM with a reasoning-specialized LLM. To this end, we
propose a layer-wise, Taylor-derived closed-form fusion mechanism that
integrates reasoning capacity into deep decoder layers while preserving visual
grounding in shallow decoder layers. Extensive experiments on challenging
multimodal reasoning benchmarks demonstrate the effectiveness of our approach.
On the MMMU benchmark, our model FRANK-38B achieves an accuracy of 69.2,
outperforming the strongest baseline InternVL2.5-38B by +5.3, and even
surpasses the proprietary GPT-4o model. Our project homepage is at:
http://iip.whu.edu.cn/frank/index.html

### 3. [BadDepth: Backdoor Attacks Against Monocular Depth Estimation in the Physical World](http://arxiv.org/pdf/2505.16154v1)

Authors: Ji Guo, Long Zhou, Zhijin Wang, Jiaming He, Qiyang Song, Aiguo Chen, Wenbo Jiang

In recent years, deep learning-based Monocular Depth Estimation (MDE) models
have been widely applied in fields such as autonomous driving and robotics.
However, their vulnerability to backdoor attacks remains unexplored. To fill
the gap in this area, we conduct a comprehensive investigation of backdoor
attacks against MDE models. Typically, existing backdoor attack methods can not
be applied to MDE models. This is because the label used in MDE is in the form
of a depth map. To address this, we propose BadDepth, the first backdoor attack
targeting MDE models. BadDepth overcomes this limitation by selectively
manipulating the target object's depth using an image segmentation model and
restoring the surrounding areas via depth completion, thereby generating
poisoned datasets for object-level backdoor attacks. To improve robustness in
physical world scenarios, we further introduce digital-to-physical augmentation
to adapt to the domain gap between the physical world and the digital domain.
Extensive experiments on multiple models validate the effectiveness of BadDepth
in both the digital domain and the physical world, without being affected by
environmental factors.

### 4. [Breaking Complexity Barriers: High-Resolution Image Restoration with Rank Enhanced Linear Attention](http://arxiv.org/pdf/2505.16157v1)

Authors: Yuang Ai, Huaibo Huang, Tao Wu, Qihang Fan, Ran He

Transformer-based models have made remarkable progress in image restoration
(IR) tasks. However, the quadratic complexity of self-attention in Transformer
hinders its applicability to high-resolution images. Existing methods mitigate
this issue with sparse or window-based attention, yet inherently limit global
context modeling. Linear attention, a variant of softmax attention,
demonstrates promise in global context modeling while maintaining linear
complexity, offering a potential solution to the above challenge. Despite its
efficiency benefits, vanilla linear attention suffers from a significant
performance drop in IR, largely due to the low-rank nature of its attention
map. To counter this, we propose Rank Enhanced Linear Attention (RELA), a
simple yet effective method that enriches feature representations by
integrating a lightweight depthwise convolution. Building upon RELA, we propose
an efficient and effective image restoration Transformer, named LAformer.
LAformer achieves effective global perception by integrating linear attention
and channel attention, while also enhancing local fitting capabilities through
a convolutional gated feed-forward network. Notably, LAformer eliminates
hardware-inefficient operations such as softmax and window shifting, enabling
efficient processing of high-resolution images. Extensive experiments across 7
IR tasks and 21 benchmarks demonstrate that LAformer outperforms SOTA methods
and offers significant computational advantages.

### 5. [Deep Learning-Driven Ultra-High-Definition Image Restoration: A Survey](http://arxiv.org/pdf/2505.16161v1)

Authors: Liyan Wang, Weixiang Zhou, Cong Wang, Kin-Man Lam, Zhixun Su, Jinshan Pan

Ultra-high-definition (UHD) image restoration aims to specifically solve the
problem of quality degradation in ultra-high-resolution images. Recent
advancements in this field are predominantly driven by deep learning-based
innovations, including enhancements in dataset construction, network
architecture, sampling strategies, prior knowledge integration, and loss
functions. In this paper, we systematically review recent progress in UHD image
restoration, covering various aspects ranging from dataset construction to
algorithm design. This serves as a valuable resource for understanding
state-of-the-art developments in the field. We begin by summarizing degradation
models for various image restoration subproblems, such as super-resolution,
low-light enhancement, deblurring, dehazing, deraining, and desnowing, and
emphasizing the unique challenges of their application to UHD image
restoration. We then highlight existing UHD benchmark datasets and organize the
literature according to degradation types and dataset construction methods.
Following this, we showcase major milestones in deep learning-driven UHD image
restoration, reviewing the progression of restoration tasks, technological
developments, and evaluations of existing methods. We further propose a
classification framework based on network architectures and sampling
strategies, helping to clearly organize existing methods. Finally, we share
insights into the current research landscape and propose directions for further
advancements. A related repository is available at
https://github.com/wlydlut/UHD-Image-Restoration-Survey.

### 6. [TRAIL: Transferable Robust Adversarial Images via Latent diffusion](http://arxiv.org/pdf/2505.16166v1)

Authors: Yuhao Xue, Zhifei Zhang, Xinyang Jiang, Yifei Shen, Junyao Gao, Wentao Gu, Jiale Zhao, Miaojing Shi, Cairong Zhao

Adversarial attacks exploiting unrestricted natural perturbations present
severe security risks to deep learning systems, yet their transferability
across models remains limited due to distribution mismatches between generated
adversarial features and real-world data. While recent works utilize
pre-trained diffusion models as adversarial priors, they still encounter
challenges due to the distribution shift between the distribution of ideal
adversarial samples and the natural image distribution learned by the diffusion
model. To address the challenge, we propose Transferable Robust Adversarial
Images via Latent Diffusion (TRAIL), a test-time adaptation framework that
enables the model to generate images from a distribution of images with
adversarial features and closely resembles the target images. To mitigate the
distribution shift, during attacks, TRAIL updates the diffusion U-Net's weights
by combining adversarial objectives (to mislead victim models) and perceptual
constraints (to preserve image realism). The adapted model then generates
adversarial samples through iterative noise injection and denoising guided by
these objectives. Experiments demonstrate that TRAIL significantly outperforms
state-of-the-art methods in cross-model attack transferability, validating that
distribution-aligned adversarial feature synthesis is critical for practical
black-box attacks.

### 7. [Erased or Dormant? Rethinking Concept Erasure Through Reversibility](http://arxiv.org/pdf/2505.16174v1)

Authors: Ping Liu, Chi Zhang

To what extent does concept erasure eliminate generative capacity in
diffusion models? While prior evaluations have primarily focused on measuring
concept suppression under specific textual prompts, we explore a complementary
and fundamental question: do current concept erasure techniques genuinely
remove the ability to generate targeted concepts, or do they merely achieve
superficial, prompt-specific suppression? We systematically evaluate the
robustness and reversibility of two representative concept erasure methods,
Unified Concept Editing and Erased Stable Diffusion, by probing their ability
to eliminate targeted generative behaviors in text-to-image models. These
methods attempt to suppress undesired semantic concepts by modifying internal
model parameters, either through targeted attention edits or model-level
fine-tuning strategies. To rigorously assess whether these techniques truly
erase generative capacity, we propose an instance-level evaluation strategy
that employs lightweight fine-tuning to explicitly test the reactivation
potential of erased concepts. Through quantitative metrics and qualitative
analyses, we show that erased concepts often reemerge with substantial visual
fidelity after minimal adaptation, indicating that current methods suppress
latent generative representations without fully eliminating them. Our findings
reveal critical limitations in existing concept erasure approaches and
highlight the need for deeper, representation-level interventions and more
rigorous evaluation standards to ensure genuine, irreversible removal of
concepts from generative models.

### 8. [A Causal Approach to Mitigate Modality Preference Bias in Medical Visual Question Answering](http://arxiv.org/pdf/2505.16209v1)

Authors: Shuchang Ye, Usman Naseem, Mingyuan Meng, Dagan Feng, Jinman Kim

Medical Visual Question Answering (MedVQA) is crucial for enhancing the
efficiency of clinical diagnosis by providing accurate and timely responses to
clinicians' inquiries regarding medical images. Existing MedVQA models suffered
from modality preference bias, where predictions are heavily dominated by one
modality while overlooking the other (in MedVQA, usually questions dominate the
answer but images are overlooked), thereby failing to learn multimodal
knowledge. To overcome the modality preference bias, we proposed a Medical
CounterFactual VQA (MedCFVQA) model, which trains with bias and leverages
causal graphs to eliminate the modality preference bias during inference.
Existing MedVQA datasets exhibit substantial prior dependencies between
questions and answers, which results in acceptable performance even if the
model significantly suffers from the modality preference bias. To address this
issue, we reconstructed new datasets by leveraging existing MedVQA datasets and
Changed their P3rior dependencies (CP) between questions and their answers in
the training and test set. Extensive experiments demonstrate that MedCFVQA
significantly outperforms its non-causal counterpart on both SLAKE, RadVQA and
SLAKE-CP, RadVQA-CP datasets.

### 9. [A Shape-Aware Total Body Photography System for In-focus Surface Coverage Optimization](http://arxiv.org/pdf/2505.16228v1)

Authors: Wei-Lun Huang, Joshua Liu, Davood Tashayyod, Jun Kang, Amir Gandjbakhche, Misha Kazhdan, Mehran Armand

Total Body Photography (TBP) is becoming a useful screening tool for patients
at high risk for skin cancer. While much progress has been made, existing TBP
systems can be further improved for automatic detection and analysis of
suspicious skin lesions, which is in part related to the resolution and
sharpness of acquired images. This paper proposes a novel shape-aware TBP
system automatically capturing full-body images while optimizing image quality
in terms of resolution and sharpness over the body surface. The system uses
depth and RGB cameras mounted on a 360-degree rotary beam, along with 3D body
shape estimation and an in-focus surface optimization method to select the
optimal focus distance for each camera pose. This allows for optimizing the
focused coverage over the complex 3D geometry of the human body given the
calibrated camera poses. We evaluate the effectiveness of the system in
capturing high-fidelity body images. The proposed system achieves an average
resolution of 0.068 mm/pixel and 0.0566 mm/pixel with approximately 85% and 95%
of surface area in-focus, evaluated on simulation data of diverse body shapes
and poses as well as a real scan of a mannequin respectively. Furthermore, the
proposed shape-aware focus method outperforms existing focus protocols (e.g.
auto-focus). We believe the high-fidelity imaging enabled by the proposed
system will improve automated skin lesion analysis for skin cancer screening.

### 10. [CT-Agent: A Multimodal-LLM Agent for 3D CT Radiology Question Answering](http://arxiv.org/pdf/2505.16229v1)

Authors: Yuren Mao, Wenyi Xu, Yuyang Qin, Yunjun Gao

Computed Tomography (CT) scan, which produces 3D volumetric medical data that
can be viewed as hundreds of cross-sectional images (a.k.a. slices), provides
detailed anatomical information for diagnosis. For radiologists, creating CT
radiology reports is time-consuming and error-prone. A visual question
answering (VQA) system that can answer radiologists' questions about some
anatomical regions on the CT scan and even automatically generate a radiology
report is urgently needed. However, existing VQA systems cannot adequately
handle the CT radiology question answering (CTQA) task for: (1) anatomic
complexity makes CT images difficult to understand; (2) spatial relationship
across hundreds slices is difficult to capture. To address these issues, this
paper proposes CT-Agent, a multimodal agentic framework for CTQA. CT-Agent
adopts anatomically independent tools to break down the anatomic complexity;
furthermore, it efficiently captures the across-slice spatial relationship with
a global-local token compression strategy. Experimental results on two 3D chest
CT datasets, CT-RATE and RadGenome-ChestCT, verify the superior performance of
CT-Agent.

### Computers and Society

### 1. [Multimodal AI-based visualization of strategic leaders' emotional dynamics: a deep behavioral analysis of Trump's trade war discourse](http://arxiv.org/pdf/2505.16274v1)

Authors: Wei Meng

This study investigates the emotional rhythms and behavioral mechanisms of
dominant political leaders in strategic decision-making. Using the Trump
administration's 125 percent tariff hike on China as a case, it adopts a
Multimodal Cognitive Behavioral Modeling framework. This includes
micro-expression tracking, acoustic intonation analysis, semantic flow
modeling, cognitive load simulation, and strategic behavior mapping to
construct a full-cycle simulation of emotion, motivation, and output. Results
reveal that Trump's decisions are not driven by rational deduction, but emerge
from dominance-coherence rhythms. A six-axis National Strategic Tempo
Intervention Framework is proposed to support anticipatory policy modeling.

### 2. [TAPAS: A Pattern-Based Approach to Assessing Government Transparency](http://arxiv.org/pdf/2505.16413v1)

Authors: Jos Zuijderwijk, Iris Beerepoot, Thomas Martens, Eva Knies, Tanja van der Lippe, Hajo A. Reijers

Government transparency, widely recognized as a cornerstone of open
government, depends on robust information management practices. Yet effective
assessment of information management remains challenging, as existing methods
fail to consider the actual working behavior of civil servants and are
resource-intensive. Using a design science research approach, we present the
Transparency Anti-Pattern Assessment System (TAPAS) -- a novel, data-driven
methodology designed to evaluate government transparency through the
identification of behavioral patterns that impede transparency. We demonstrate
TAPAS's real-world applicability at a Dutch ministry, analyzing their
electronic document management system data from the past two decades. We
identify eight transparency anti-patterns grouped into four categories:
Incomplete Documentation, Limited Accessibility, Unclear Information, and
Delayed Documentation. We show that TAPAS enables continuous monitoring and
provides actionable insights without requiring significant resource
investments.

### 3. [Psychology-driven LLM Agents for Explainable Panic Prediction on Social Media during Sudden Disaster Events](http://arxiv.org/pdf/2505.16455v1)

Authors: Mengzhu Liu, Zhengqiu Zhu, Chuan Ai, Chen Gao, Xinghong Li, Lingnan He, Kaisheng Lai, Yingfeng Chen, Xin Lu, Yong Li, Quanjun Yin

During sudden disaster events, accurately predicting public panic sentiment
on social media is crucial for proactive governance and crisis management.
Current efforts on this problem face three main challenges: lack of finely
annotated data hinders emotion prediction studies, unmodeled risk perception
causes prediction inaccuracies, and insufficient interpretability of panic
formation mechanisms. We address these issues by proposing a Psychology-driven
generative Agent framework (PsychoAgent) for explainable panic prediction based
on emotion arousal theory. Specifically, we first construct a fine-grained open
panic emotion dataset (namely COPE) via human-large language models (LLMs)
collaboration to mitigate semantic bias. Then, we develop a framework
integrating cross-domain heterogeneous data grounded in psychological
mechanisms to model risk perception and cognitive differences in emotion
generation. To enhance interpretability, we design an LLM-based role-playing
agent that simulates individual psychological chains through dedicatedly
designed prompts. Experimental results on our annotated dataset show that
PsychoAgent improves panic emotion prediction performance by 12.6% to 21.7%
compared to baseline models. Furthermore, the explainability and generalization
of our approach is validated. Crucially, this represents a paradigm shift from
opaque "data-driven fitting" to transparent "role-based simulation with
mechanistic interpretation" for panic emotion prediction during emergencies.
Our implementation is publicly available at:
https://anonymous.4open.science/r/PsychoAgent-19DD.

### 4. [NY Real Estate Racial Equity Analysis via Applied Machine Learning](http://arxiv.org/pdf/2505.16946v1)

Authors: Sanjana Chalavadi, Andrei Pastor, Terry Leitch

This study analyzes tract-level real estate ownership patterns in New York
State (NYS) and New York City (NYC) to uncover racial disparities. We use an
advanced race/ethnicity imputation model (LSTM+Geo with XGBoost filtering,
validated at 89.2% accuracy) to compare the predicted racial composition of
property owners to the resident population from census data. We examine both a
Full Model (statewide) and a Name-Only LSTM Model (NYC) to assess how
incorporating geospatial context affects our predictions and disparity
estimates. The results reveal significant inequities: White individuals hold a
disproportionate share of properties and property value relative to their
population, while Black, Hispanic, and Asian communities are underrepresented
as property owners. These disparities are most pronounced in minority-majority
neighborhoods, where ownership is predominantly White despite a predominantly
non-White population. Corporate ownership (LLCs, trusts, etc.) exacerbates
these gaps by reducing owner-occupied opportunities in urban minority
communities. We provide a breakdown of ownership vs. population by race for
majority-White, -Black, -Hispanic, and -Asian tracts, identify those with
extreme ownership disparities, and compare patterns in urban, suburban, and
rural contexts. The findings underscore persistent racial inequity in property
ownership, reflecting broader historical and socio-economic forces, and
highlight the importance of data-driven approaches to address these issues.

### 5. [CLEAR: A Clinically-Grounded Tabular Framework for Radiology Report Evaluation](http://arxiv.org/pdf/2505.16325v1)

Authors: Yuyang Jiang, Chacha Chen, Shengyuan Wang, Feng Li, Zecong Tang, Benjamin M. Mervak, Lydia Chelala, Christopher M Straus, Reve Chahine, Samuel G. Armato III, Chenhao Tan

Existing metrics often lack the granularity and interpretability to capture
nuanced clinical differences between candidate and ground-truth radiology
reports, resulting in suboptimal evaluation. We introduce a Clinically-grounded
tabular framework with Expert-curated labels and Attribute-level comparison for
Radiology report evaluation (CLEAR). CLEAR not only examines whether a report
can accurately identify the presence or absence of medical conditions, but also
assesses whether it can precisely describe each positively identified condition
across five key attributes: first occurrence, change, severity, descriptive
location, and recommendation. Compared to prior works, CLEAR's
multi-dimensional, attribute-level outputs enable a more comprehensive and
clinically interpretable evaluation of report quality. Additionally, to measure
the clinical alignment of CLEAR, we collaborate with five board-certified
radiologists to develop CLEAR-Bench, a dataset of 100 chest X-ray reports from
MIMIC-CXR, annotated across 6 curated attributes and 13 CheXpert conditions.
Our experiments show that CLEAR achieves high accuracy in extracting clinical
attributes and provides automated metrics that are strongly aligned with
clinical judgment.

### 6. [Utilizing citation index and synthetic quality measure to compare Wikipedia languages across various topics](http://arxiv.org/pdf/2505.16506v1)

Authors: Włodzimierz Lewoniewski, Krzysztof Węcel, Witold Abramowicz

This study presents a comparative analysis of 55 Wikipedia language editions
employing a citation index alongside a synthetic quality measure. Specifically,
we identified the most significant Wikipedia articles within distinct topical
areas, selecting the top 10, top 25, and top 100 most cited articles in each
topic and language version. This index was built on the basis of wikilinks
between Wikipedia articles in each language version and in order to do that we
processed 6.6 billion page-to-page link records. Next, we used a quality score
for each Wikipedia article - a synthetic measure scaled from 0 to 100. This
approach enabled quality comparison of Wikipedia articles even between language
versions with different quality grading schemes. Our results highlight
disparities among Wikipedia language editions, revealing strengths and gaps in
content coverage and quality across topics.

### 7. [Reconsidering Fairness Through Unawareness from the Perspective of Model Multiplicity](http://arxiv.org/pdf/2505.16638v1)

Authors: Benedikt Höltgen, Nuria Oliver

Fairness through Unawareness (FtU) describes the idea that discrimination
against demographic groups can be avoided by not considering group membership
in the decisions or predictions. This idea has long been criticized in the
machine learning literature as not being sufficient to ensure fairness. In
addition, the use of additional features is typically thought to increase the
accuracy of the predictions for all groups, so that FtU is sometimes thought to
be detrimental to all groups. In this paper, we show both theoretically and
empirically that FtU can reduce algorithmic discrimination without necessarily
reducing accuracy. We connect this insight with the literature on Model
Multiplicity, to which we contribute with novel theoretical and empirical
results. Furthermore, we illustrate how, in a real-life application, FtU can
contribute to the deployment of more equitable policies without losing
efficacy. Our findings suggest that FtU is worth considering in practical
applications, particularly in high-risk scenarios, and that the use of
protected attributes such as gender in predictive models should be accompanied
by a clear and well-founded justification.

### 8. [Let Androids Dream of Electric Sheep: A Human-like Image Implication Understanding and Reasoning Framework](http://arxiv.org/pdf/2505.17019v1)

Authors: Chenhao Zhang, Yazhe Niu

Metaphorical comprehension in images remains a critical challenge for AI
systems, as existing models struggle to grasp the nuanced cultural, emotional,
and contextual implications embedded in visual content. While multimodal large
language models (MLLMs) excel in basic Visual Question Answer (VQA) tasks, they
struggle with a fundamental limitation on image implication tasks: contextual
gaps that obscure the relationships between different visual elements and their
abstract meanings. Inspired by the human cognitive process, we propose Let
Androids Dream (LAD), a novel framework for image implication understanding and
reasoning. LAD addresses contextual missing through the three-stage framework:
(1) Perception: converting visual information into rich and multi-level textual
representations, (2) Search: iteratively searching and integrating cross-domain
knowledge to resolve ambiguity, and (3) Reasoning: generating context-alignment
image implication via explicit reasoning. Our framework with the lightweight
GPT-4o-mini model achieves SOTA performance compared to 15+ MLLMs on English
image implication benchmark and a huge improvement on Chinese benchmark,
performing comparable with the GPT-4o model on Multiple-Choice Question (MCQ)
and outperforms 36.7% on Open-Style Question (OSQ). Additionally, our work
provides new insights into how AI can more effectively interpret image
implications, advancing the field of vision-language reasoning and human-AI
interaction. Our project is publicly available at
https://github.com/MING-ZCH/Let-Androids-Dream-of-Electric-Sheep.

### Databases

### 1. [A Chase-based Approach to Consistent Answers of Analytic Queries in Star Schemas](http://arxiv.org/pdf/2505.16802v1)

Authors: Dominique Laurent, Nicolas Spyratos

We present an approach to computing consistent answers to analytic queries in
data warehouses operating under a star schema and possibly containing missing
values and inconsistent data. Our approach is based on earlier work concerning
consistent query answering for standard, non-analytic queries in multi-table
databases. In that work we presented polynomial algorithms for computing either
the exact consistent answer to a standard, non analytic query or bounds of the
exact answer, depending on whether the query involves a selection condition or
not.
  We extend this approach to computing exact consistent answers of analytic
queries over star schemas, provided that the selection condition in the query
involves no keys and satisfies the property of independency (i.e., the
condition can be expressed as a conjunction of conditions each involving a
single attribute). The main contributions of this paper are: (a) a polynomial
algorithm for computing the exact consistent answer to a usual
projection-selection-join query over a star schema under the above restrictions
on the selection condition, and (b) showing that, under the same restrictions
the exact consistent answer to an analytic query over a star schema can be
computed in time polynomial in the size of the data warehouse.

### 2. [Towards Machine-actionable FAIR Digital Objects with a Typing Model that Enables Operations](http://arxiv.org/pdf/2505.16550v1)

Authors: Maximilian Inckmann, Nicolas Blumenröhr, Rossella Aversa

FAIR Digital Objects support research data management aligned with the FAIR
principles. To be machine-actionable, they must support operations that
interact with their contents. This can be achieved by associating operations
with FAIR-DO data types. However, current typing models and Data Type
Registries lack support for type-associated operations. In this work, we
introduce a typing model that describes type-associated and technology-agnostic
FAIR Digital Object Operations in a machine-actionable way, building and
improving on the existing concepts. In addition, we introduce the Integrated
Data Type and Operations Registry with Inheritance System, a prototypical
implementation of this model that integrates inheritance mechanisms for data
types, a rule-based validation system, and the computation of type-operation
associations. Our approach significantly improves the machine-actionability of
FAIR Digital Objects, paving the way towards dynamic, interoperable, and
reproducible research workflows.

### 3. [Restricted Chase Termination: You Want More than Fairness](http://arxiv.org/pdf/2505.16551v1)

Authors: David Carral, Lukas Gerlach, Lucas Larroque, Michaël Thomazo

The chase is a fundamental algorithm with ubiquitous uses in database theory.
Given a database and a set of existential rules (aka tuple-generating
dependencies), it iteratively extends the database to ensure that the rules are
satisfied in a most general way. This process may not terminate, and a major
problem is to decide whether it does. This problem has been studied for a large
number of chase variants, which differ by the conditions under which a rule is
applied to extend the database. Surprisingly, the complexity of the universal
termination of the restricted (aka standard) chase is not fully understood. We
close this gap by placing universal restricted chase termination in the
analytical hierarchy. This higher hardness is due to the fairness condition,
and we propose an alternative condition to reduce the hardness of universal
termination.

### 4. [WikiDBGraph: Large-Scale Database Graph of Wikidata for Collaborative Learning](http://arxiv.org/pdf/2505.16635v1)

Authors: Zhaomin Wu, Ziyang Wang, Bingsheng He

Tabular data, ubiquitous and rich in informational value, is an increasing
focus for deep representation learning, yet progress is hindered by studies
centered on single tables or isolated databases, which limits model
capabilities due to data scale. While collaborative learning approaches such as
federated learning, transfer learning, split learning, and tabular foundation
models aim to learn from multiple correlated databases, they are challenged by
a scarcity of real-world interconnected tabular resources. Current data lakes
and corpora largely consist of isolated databases lacking defined
inter-database correlations. To overcome this, we introduce WikiDBGraph, a
large-scale graph of 100,000 real-world tabular databases from WikiData,
interconnected by 17 million edges and characterized by 13 node and 12 edge
properties derived from its database schema and data distribution.
WikiDBGraph's weighted edges identify both instance- and feature-overlapped
databases. Experiments on these newly identified databases confirm that
collaborative learning yields superior performance, thereby offering
considerable promise for structured foundation model training while also
exposing key challenges and future directions for learning from interconnected
tabular data.

### Distributed, Parallel, and Cluster Computing

### 1. [On the Runtime of Local Mutual Exclusion for Anonymous Dynamic Networks](http://arxiv.org/pdf/2505.16139v1)

Authors: Anya Chaturvedi, Joshua J. Daymude, Andréa W. Richa

Algorithms for mutual exclusion aim to isolate potentially concurrent
accesses to the same shared resources. Motivated by distributed computing
research on programmable matter and population protocols where interactions
among entities are often assumed to be isolated, Daymude, Richa, and Scheideler
(SAND`22) introduced a variant of the local mutual exclusion problem that
applies to arbitrary dynamic networks: each node, on issuing a lock request,
must acquire exclusive locks on itself and all its persistent neighbors, i.e.,
the neighbors that remain connected to it over the duration of the lock
request. Assuming adversarial edge dynamics, semi-synchronous or asynchronous
concurrency, and anonymous nodes communicating via message passing, their
randomized algorithm achieves mutual exclusion (non-intersecting lock sets) and
lockout freedom (eventual success with probability 1). However, they did not
analyze their algorithm's runtime. In this paper, we prove that any node will
successfully lock itself and its persistent neighbors within O$(n\Delta^3)$
open rounds of its lock request in expectation, where $n$ is the number of
nodes in the dynamic network, $\Delta$ is the maximum degree of the dynamic
network, rounds are normalized to the execution time of the ``slowest'' node,
and ``closed'' rounds when some persistent neighbors are already locked by
another node are ignored (i.e., only ``open" rounds are considered).

### 2. [Brand: Managing Training Data with Batched Random Access](http://arxiv.org/pdf/2505.16280v1)

Authors: Yuhao Li, Xuanhua Shi, Yunfei Zhao, Yongluan Zhou, Yusheng Hua, Xuehai Qian

This paper propose Brand, a comprehensive memory management system for deep
learning training (DLT) where the memory capacity is much smaller than the size
of the training datasets. Brand starts with a bold design choice that data
files are always read from disk in batch, named chunk. Based on this
assumption, we propose efficient data access protocol in both single-node
setting and distributed environment with multiple nodes. The protocol minimizes
the wasted data read due to larger granularity, enables efficient inter-node
prefetching, while still ensuring randomness required by DLT. The experimental
results indicate that Brand can significantly accelerate data fetching in DLT,
achieving up to a 4.57x improvement in end-to-end training compared to PyTorch.

### 3. [Minimizing Energy in Reliability and Deadline-Ensured Workflow Scheduling in Cloud](http://arxiv.org/pdf/2505.16496v1)

Authors: Suvarthi Sarkar, Dhanesh V, Ketan Singh, Aryabartta Sahu

With the increasing prevalence of computationally intensive workflows in
cloud environments, it has become crucial for cloud platforms to optimize
energy consumption while ensuring the feasibility of user workflow schedules
with respect to strict deadlines and reliability constraints. The key
challenges faced when cloud systems provide virtual machines of varying levels
of reliability, energy consumption, processing frequencies, and computing
capabilities to execute tasks of these workflows. To address these issues, we
propose an adaptive strategy based on maximum fan-out ratio considering the
slack of tasks and deadline distribution for scheduling workflows in a single
cloud platform, intending to minimise energy consumption while ensuring strict
reliability and deadline constraints. We also propose an approach for dynamic
scheduling of workflow using the rolling horizon concept to consider the
dynamic execution time of tasks of the workflow where the actual task execution
time at run time is shorter than worst-case execution time in most of the
cases. Our proposed static approach outperforms the state-of-the-art (SOTA) by
up to 70% on average in scenarios without deadline constraints, and achieves an
improvement of approximately 2% in deadline-constrained cases. The dynamic
variant of our approach demonstrates even stronger performance, surpassing SOTA
by 82% in non-deadline scenarios and by up to 27% on average when deadline
constraints are enforced. Furthermore, in comparison with the static optimal
solution, our static approach yields results within a factor of 1.1, while the
dynamic approach surpasses the optimal baseline by an average of 25%.

### 4. [Towards Stream-Based Monitoring for EVM Networks](http://arxiv.org/pdf/2505.16095v1)

Authors: Emanuel Onica, Claudiu-Nicu Bărbieru, Andrei Arusoaie, Oana-Otilia Captarencu, Ciprian Amariei

We believe that leveraging real-time blockchain operational data is of
particular interest in the context of the current rapid expansion of rollup
networks in the Ethereum ecosystem. Given the compatible but also competing
ground that rollups offer for applications, stream-based monitoring can be of
use both to developers and to EVM networks governance. In this paper, we
discuss this perspective and propose a basic monitoring pipeline.

### 5. [Multimodal Online Federated Learning with Modality Missing in Internet of Things](http://arxiv.org/pdf/2505.16138v1)

Authors: Heqiang Wang, Xiang Liu, Xiaoxiong Zhong, Lixing Chen, Fangming Liu, Weizhe Zhang

The Internet of Things (IoT) ecosystem generates vast amounts of multimodal
data from heterogeneous sources such as sensors, cameras, and microphones. As
edge intelligence continues to evolve, IoT devices have progressed from simple
data collection units to nodes capable of executing complex computational
tasks. This evolution necessitates the adoption of distributed learning
strategies to effectively handle multimodal data in an IoT environment.
Furthermore, the real-time nature of data collection and limited local storage
on edge devices in IoT call for an online learning paradigm. To address these
challenges, we introduce the concept of Multimodal Online Federated Learning
(MMO-FL), a novel framework designed for dynamic and decentralized multimodal
learning in IoT environments. Building on this framework, we further account
for the inherent instability of edge devices, which frequently results in
missing modalities during the learning process. We conduct a comprehensive
theoretical analysis under both complete and missing modality scenarios,
providing insights into the performance degradation caused by missing
modalities. To mitigate the impact of modality missing, we propose the
Prototypical Modality Mitigation (PMM) algorithm, which leverages prototype
learning to effectively compensate for missing modalities. Experimental results
on two multimodal datasets further demonstrate the superior performance of PMM
compared to benchmarks.

### 6. [Recursive Offloading for LLM Serving in Multi-tier Networks](http://arxiv.org/pdf/2505.16502v1)

Authors: Zhiyuan Wu, Sheng Sun, Yuwei Wang, Min Liu, Bo Gao, Jinda Lu, Zheming Yang, Tian Wen

Heterogeneous device-edge-cloud computing infrastructures have become widely
adopted in telecommunication operators and Wide Area Networks (WANs), offering
multi-tier computational support for emerging intelligent services. With the
rapid proliferation of Large Language Model (LLM) services, efficiently
coordinating inference tasks and reducing communication overhead within these
multi-tier network architectures becomes a critical deployment challenge.
Existing LLM serving paradigms exhibit significant limitations: on-device
deployment supports only lightweight LLMs due to hardware constraints, while
cloud-centric deployment suffers from resource congestion and considerable
prompt communication overhead caused by frequent service requests during peak
periods. Although the model-cascading-based inference strategy adapts better to
multi-tier networks, its reliance on fine-grained, manually adjusted thresholds
makes it less responsive to dynamic network conditions and varying task
complexities. To address these challenges, we propose RecServe, a recursive
offloading framework tailored for LLM serving in multi-tier networks. RecServe
integrates a task-specific hierarchical confidence evaluation mechanism that
guides offloading decisions based on inferred task complexity in progressively
scaled LLMs across device, edge, and cloud tiers. To further enable intelligent
task routing across tiers, RecServe employs a sliding-window-based dynamic
offloading strategy with quantile interpolation, enabling real-time tracking of
historical confidence distributions and adaptive offloading threshold
adjustments. Experiments on eight datasets demonstrate that RecServe
outperforms CasServe in both service quality and communication efficiency, and
reduces the communication burden by over 50\% compared to centralized
cloud-based serving.

### 7. [Smaller, Smarter, Closer: The Edge of Collaborative Generative AI](http://arxiv.org/pdf/2505.16499v1)

Authors: Roberto Morabito, SiYoung Jang

The rapid adoption of generative AI (GenAI), particularly Large Language
Models (LLMs), has exposed critical limitations of cloud-centric deployments,
including latency, cost, and privacy concerns. Meanwhile, Small Language Models
(SLMs) are emerging as viable alternatives for resource-constrained edge
environments, though they often lack the capabilities of their larger
counterparts. This article explores the potential of collaborative inference
systems that leverage both edge and cloud resources to address these
challenges. By presenting distinct cooperation strategies alongside practical
design principles and experimental insights, we offer actionable guidance for
deploying GenAI across the computing continuum.

### 8. [Edge-First Language Model Inference: Models, Metrics, and Tradeoffs](http://arxiv.org/pdf/2505.16508v1)

Authors: SiYoung Jang, Roberto Morabito

The widespread adoption of Language Models (LMs) across industries is driving
interest in deploying these services across the computing continuum, from the
cloud to the network edge. This shift aims to reduce costs, lower latency, and
improve reliability and privacy. Small Language Models (SLMs), enabled by
advances in model compression, are central to this shift, offering a path to
on-device inference on resource-constrained edge platforms. This work examines
the interplay between edge and cloud deployments, starting from detailed
benchmarking of SLM capabilities on single edge devices, and extending to
distributed edge clusters. We identify scenarios where edge inference offers
comparable performance with lower costs, and others where cloud fallback
becomes essential due to limits in scalability or model capacity. Rather than
proposing a one-size-fits-all solution, we present platform-level comparisons
and design insights for building efficient, adaptive LM inference systems
across heterogeneous environments.

### Digital Libraries

### 1. [Towards Machine-actionable FAIR Digital Objects with a Typing Model that Enables Operations](http://arxiv.org/pdf/2505.16550v1)

Authors: Maximilian Inckmann, Nicolas Blumenröhr, Rossella Aversa

FAIR Digital Objects support research data management aligned with the FAIR
principles. To be machine-actionable, they must support operations that
interact with their contents. This can be achieved by associating operations
with FAIR-DO data types. However, current typing models and Data Type
Registries lack support for type-associated operations. In this work, we
introduce a typing model that describes type-associated and technology-agnostic
FAIR Digital Object Operations in a machine-actionable way, building and
improving on the existing concepts. In addition, we introduce the Integrated
Data Type and Operations Registry with Inheritance System, a prototypical
implementation of this model that integrates inheritance mechanisms for data
types, a rule-based validation system, and the computation of type-operation
associations. Our approach significantly improves the machine-actionability of
FAIR Digital Objects, paving the way towards dynamic, interoperable, and
reproducible research workflows.

### 2. [SC4ANM: Identifying Optimal Section Combinations for Automated Novelty Prediction in Academic Papers](http://arxiv.org/pdf/2505.16330v1)

Authors: Wenqing Wu, Chengzhi Zhang, Tong Bao, Yi Zhao

Novelty is a core component of academic papers, and there are multiple
perspectives on the assessment of novelty. Existing methods often focus on word
or entity combinations, which provide limited insights. The content related to
a paper's novelty is typically distributed across different core sections,
e.g., Introduction, Methodology and Results. Therefore, exploring the optimal
combination of sections for evaluating the novelty of a paper is important for
advancing automated novelty assessment. In this paper, we utilize different
combinations of sections from academic papers as inputs to drive language
models to predict novelty scores. We then analyze the results to determine the
optimal section combinations for novelty score prediction. We first employ
natural language processing techniques to identify the sectional structure of
academic papers, categorizing them into introduction, methods, results, and
discussion (IMRaD). Subsequently, we used different combinations of these
sections (e.g., introduction and methods) as inputs for pretrained language
models (PLMs) and large language models (LLMs), employing novelty scores
provided by human expert reviewers as ground truth labels to obtain prediction
results. The results indicate that using introduction, results and discussion
is most appropriate for assessing the novelty of a paper, while the use of the
entire text does not yield significant results. Furthermore, based on the
results of the PLMs and LLMs, the introduction and results appear to be the
most important section for the task of novelty score prediction. The code and
dataset for this paper can be accessed at
https://github.com/njust-winchy/SC4ANM.

### 3. [Utilizing citation index and synthetic quality measure to compare Wikipedia languages across various topics](http://arxiv.org/pdf/2505.16506v1)

Authors: Włodzimierz Lewoniewski, Krzysztof Węcel, Witold Abramowicz

This study presents a comparative analysis of 55 Wikipedia language editions
employing a citation index alongside a synthetic quality measure. Specifically,
we identified the most significant Wikipedia articles within distinct topical
areas, selecting the top 10, top 25, and top 100 most cited articles in each
topic and language version. This index was built on the basis of wikilinks
between Wikipedia articles in each language version and in order to do that we
processed 6.6 billion page-to-page link records. Next, we used a quality score
for each Wikipedia article - a synthetic measure scaled from 0 to 100. This
approach enabled quality comparison of Wikipedia articles even between language
versions with different quality grading schemes. Our results highlight
disparities among Wikipedia language editions, revealing strengths and gaps in
content coverage and quality across topics.

### Discrete Mathematics

### 1. [Continuous Petri Nets Faithfully Fluidify Most Permissive Boolean Networks](http://arxiv.org/pdf/2505.16683v1)

Authors: Stefan Haar, Juri Kolčák

The analysis of biological networks has benefited from the richness of
Boolean networks (BNs) and the associated theory. These results have been
further fortified in recent years by the emergence of Most Permissive (MP)
semantics, combining efficient analysis methods with a greater capacity of
explaining pathways to states hitherto thought unreachable, owing to
limitations of the classical update modes. While MPBNs are understood to
capture any behaviours that can be observed at a lower level of abstraction,
all the way down to continuous refinements, the specifics and potential of the
models and analysis, especially attractors, across the abstraction scale remain
unexplored. Here, we fluidify MPBNs by means of Continuous Petri nets (CPNs), a
model of (uncountably infinite) dynamic systems that has been successfully
explored for modelling and theoretical purposes. CPNs create a formal link
between MPBNs and their continuous dynamical refinements such as ODE models.
The benefits of CPNs extend beyond the model refinement, and constitute well
established theory and analysis methods, recently augmented by abstract and
symbolic reachability graphs. These structures are shown to compact the
possible behaviours of the system with focus on events which drive the choice
of long-term behaviour in which the system eventually stabilises. The current
paper brings an important keystone to this novel methodology for biological
networks, namely the proof that extant PN encoding of BNs instantiated as a CPN
simulates the MP semantics. In spite of the underlying dynamics being
continuous, the analysis remains in the realm of discrete methods, constituting
an extension of all previous work.

### 2. [Scaling Quantum Simulation-Based Optimization: Demonstrating Efficient Power Grid Management with Deep QAOA Circuits](http://arxiv.org/pdf/2505.16444v1)

Authors: Maximilian Adler, Jonas Stein, Michael Lachner

Quantum Simulation-based Optimization (QuSO) is a recently proposed class of
optimization problems that entails industrially relevant problems characterized
by cost functions or constraints that depend on summary statistic information
about the simulation of a physical system or process. This work extends initial
theoretical results that proved an up-to-exponential speedup for the simulation
component of the QAOA-based QuSO solver proposed by Stein et al. for the unit
commitment problem by an empirical evaluation of the optimization component
using a standard benchmark dataset, the IEEE 57-bus system. Exploiting clever
classical pre-computation, we develop a very efficient classical quantum
circuit simulation that bypasses costly ancillary qubit requirements by the
original algorithm, allowing for large-scale experiments. Utilizing more than
1000 QAOA layers and up to 20 qubits, our experiments complete a proof of
concept implementation for the proposed QuSO solver, showing that it can
achieve both highly competitive performance and efficiency in its optimization
component compared to a standard classical baseline, i.e., simulated annealing.

### 3. [The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](http://arxiv.org/pdf/2505.16716v1)

Authors: Moritz Stargalla, Christoph Hertrich, Daniel Reichman

An established measure of the expressive power of a given ReLU neural network
is the number of linear regions into which it partitions the input space. There
exist many different, non-equivalent definitions of what a linear region
actually is. We systematically assess which papers use which definitions and
discuss how they relate to each other. We then analyze the computational
complexity of counting the number of such regions for the various definitions.
Generally, this turns out to be an intractable problem. We prove NP- and
#P-hardness results already for networks with one hidden layer and strong
hardness of approximation results for two or more hidden layers. Finally, on
the algorithmic side, we demonstrate that counting linear regions can at least
be achieved in polynomial space for some common definitions.

### Data Structures and Algorithms

### 1. [Streaming Diameter of High-Dimensional Points](http://arxiv.org/pdf/2505.16720v1)

Authors: Magnús M. Halldórsson, Nicolaos Matsakis, Pavel Veselý

We improve the space bound for streaming approximation of Diameter but also
of Farthest Neighbor queries, Minimum Enclosing Ball and its Coreset, in
high-dimensional Euclidean spaces. In particular, our deterministic streaming
algorithms store $\mathcal{O}(\varepsilon^{-2}\log(\frac{1}{\varepsilon}))$
points. This improves by a factor of $\varepsilon^{-1}$ the previous space
bound of Agarwal and Sharathkumar (SODA 2010), while offering a simpler and
more complete argument. We also show that storing $\Omega(\varepsilon^{-1})$
points is necessary for a $(\sqrt{2}+\varepsilon)$-approximation of Farthest
Pair or Farthest Neighbor queries.

### 2. [On the Two Paths Theorem and the Two Disjoint Paths Problem](http://arxiv.org/pdf/2505.16431v1)

Authors: Samuel Humeau, Damien Pous

A tuple (s1,t1,s2,t2) of vertices in a simple undirected graph is 2-linked
when there are two vertex-disjoint paths respectively from s1 to t1 and s2 to
t2. A graph is 2-linked when all such tuples are 2-linked. We give a new and
simple proof of the ``two paths theorem'', a characterisation of edge-maximal
graphs which are not 2-linked as webs: particular near triangulations filled
with cliques. Our proof works by generalising the theorem, replacing the four
vertices above by an arbitrary tuple; it does not require major theorems such
as Kuratowski's or Menger's theorems. Instead it follows an inductive
characterisation of generalised webs via parallel composition, a graph
operation consisting in taking a disjoint union before identifying some pairs
of vertices. We use the insights provided by this proof to design a simple
O(nm) recursive algorithm for the ``two vertex-disjoint paths'' problem. This
algorithm is constructive in that it returns either two disjoint paths, or an
embedding of the input graph into a web.

### 3. [Contextual Learning for Stochastic Optimization](http://arxiv.org/pdf/2505.16829v1)

Authors: Anna Heuser, Thomas Kesselheim

Motivated by stochastic optimization, we introduce the problem of learning
from samples of contextual value distributions. A contextual value distribution
can be understood as a family of real-valued distributions, where each sample
consists of a context $x$ and a random variable drawn from the corresponding
real-valued distribution $D_x$. By minimizing a convex surrogate loss, we learn
an empirical distribution $D'_x$ for each context, ensuring a small L\'evy
distance to $D_x$. We apply this result to obtain the sample complexity bounds
for the learning of an $\epsilon$-optimal policy for stochastic optimization
problems defined on an unknown contextual value distribution. The sample
complexity is shown to be polynomial for the general case of strongly monotone
and stable optimization problems, including Single-item Revenue Maximization,
Pandora's Box and Optimal Stopping.

### 4. [Quasi-optimal hierarchically semi-separable matrix approximation](http://arxiv.org/pdf/2505.16937v1)

Authors: Noah Amsel, Tyler Chen, Feyza Duman Keles, Diana Halikias, Cameron Musco, Christopher Musco, David Persson

We present a randomized algorithm for producing a quasi-optimal
hierarchically semi-separable (HSS) approximation to an $N\times N$ matrix $A$
using only matrix-vector products with $A$ and $A^T$. We prove that, using $O(k
\log(N/k))$ matrix-vector products and ${O}(N k^2 \log(N/k))$ additional
runtime, the algorithm returns an HSS matrix $B$ with rank-$k$ blocks whose
expected Frobenius norm error $\mathbb{E}[\|A - B\|_F^2]$ is at most
$O(\log(N/k))$ times worse than the best possible approximation error by an HSS
rank-$k$ matrix. In fact, the algorithm we analyze in a simple modification of
an empirically effective method proposed by [Levitt & Martinsson, SISC 2024].
As a stepping stone towards our main result, we prove two results that are of
independent interest: a similar guarantee for a variant of the algorithm which
accesses $A$'s entries directly, and explicit error bounds for near-optimal
subspace approximation using projection-cost-preserving sketches. To the best
of our knowledge, our analysis constitutes the first polynomial-time
quasi-optimality result for HSS matrix approximation, both in the explicit
access model and the matrix-vector product query model.

### Emerging Technologies

### 1. [Simulation-Guided Approximate Logic Synthesis Under the Maximum Error Constraint](http://arxiv.org/pdf/2505.16769v1)

Authors: Chang Meng, Weikang Qian, Giovanni De Micheli

Approximate computing is an effective computing paradigm for improving energy
efficiency of error-tolerant applications. Approximate logic synthesis (ALS) is
an automatic process to generate approximate circuits with reduced area, delay,
and power, while satisfying user-specified error constraints. This paper
focuses on ALS under the maximum error constraint. As an essential error metric
that provides a worst-case error guarantee, the maximum error is crucial for
many applications such as image processing and machine learning. This work
proposes an efficient simulation-guided ALS flow that handles this constraint.
It utilizes logic simulation to 1) prune local approximate changes (LACs) with
large errors that violate the error constraint, and 2) accelerate the SAT-based
LAC selection process. Furthermore, to enhance scalability, our ALS flow
iteratively selects a set of promising LACs satisfying the error constraint to
improve the efficiency. The experimental results show that compared with the
state-of-the-art method, our ALS flow accelerates by 30.6 times, and further
reduces circuit area and delay by 18.2% and 4.9%, respectively. Notably, our
flow scales to large EPFL benchmarks with up to 38540 nodes, which cannot be
handled by any existing ALS method for maximum error.

### 2. [Is Circuit Depth Accurate for Comparing Quantum Circuit Runtimes?](http://arxiv.org/pdf/2505.16908v1)

Authors: Matthew Tremba, Ji Liu, Paul Hovland

Although quantum circuit depth is commonly used to estimate differences in
circuit runtimes, it overlooks a prevailing trait of current hardware
implementation: different gates have different execution times. Consequently,
the use of depth may lead to inaccurate comparisons of circuit runtimes,
especially for circuits of similar scale. In this paper, we introduce an
alternative metric, gate-aware depth, that uses unique gate weights, and
investigate how its accuracy in comparing circuit runtimes compares to the
existing metrics of traditional and multi-qubit circuit depth. To do so, we
compiled a suite of 15 practical circuits using different algorithms and
compared depths and runtimes between the compiled versions to determine how
accurately the size of the change in depth approximated the size of the change
in runtime, and how accurately the order of circuits by depth matched their
order by runtime. When approximating the size of runtime changes, gate-aware
depth decreased the approximation error by an average of 412 times relative to
traditional depth and 124 times relative to multi-qubit depth. When matching
the order of true runtimes, gate-aware depth achieved the highest accuracy on
all devices and a perfect accuracy of 100% on five out of six devices.
Additionally, we show that the optimal weights needed to achieve these accuracy
improvements can be easily calculated using device gate times, and provide good
general weight values for the IBM Eagle and Heron architectures.

### 3. [Advanced Integration Strategies for ESD Protection and Termination in High-Speed LVDS Systems](http://arxiv.org/pdf/2505.16200v1)

Authors: Kavya Gaddipati

This technical article explores comprehensive strategies for integrating
Electrostatic Discharge (ESD) protection diodes and termination resistors in
LowVoltage Differential Signaling (LVDS) designs. The article examines critical
aspects of protection mechanisms, design considerations, impedance matching,
and placement optimization techniques. Through detailed analysis of layout
considerations and advanced design strategies, the article presents solutions
for common integration challenges. It emphasizes the importance of signal
integrity maintenance and protection effectiveness while providing practical
guidelines for implementing robust LVDS systems. Various methodologies for
performance optimization and validation are discussed, offering designers a
thorough framework for creating reliable high-speed digital systems that
balance protection requirements with signal integrity demands.

### 4. [Dynamic Reservoir Computing with Physical Neuromorphic Networks](http://arxiv.org/pdf/2505.16813v1)

Authors: Yinhao Xu, Georg A. Gottwald, Zdenka Kuncic

Reservoir Computing (RC) with physical systems requires an understanding of
the underlying structure and internal dynamics of the specific physical
reservoir. In this study, physical nano-electronic networks with neuromorphic
dynamics are investigated for their use as physical reservoirs in an RC
framework. These neuromorphic networks operate as dynamic reservoirs, with node
activities in general coupled to the edge dynamics through nonlinear
nano-electronic circuit elements, and the reservoir outputs influenced by the
underlying network connectivity structure. This study finds that networks with
varying degrees of sparsity generate more useful nonlinear temporal outputs for
dynamic RC compared to dense networks. Dynamic RC is also tested on an
autonomous multivariate chaotic time series prediction task with networks of
varying densities, which revealed the importance of network sparsity in
maintaining network activity and overall dynamics, that in turn enabled the
learning of the chaotic Lorenz63 system's attractor behavior.

### Graphics

### 1. [From Reality to Virtual Worlds: The Role of Photogrammetry in Game Development](http://arxiv.org/pdf/2505.16951v1)

Authors: Santiago Berrezueta-Guzman, Andrei Koshelev, Stefan Wagner

Photogrammetry is transforming digital content creation by enabling the rapid
conversion of real-world objects into highly detailed 3D models. This paper
evaluates the role of RealityCapture, a GPU-accelerated photogrammetry tool, in
game development of Virtual Reality (VR). We assess its efficiency,
reconstruction accuracy, and integration with Unreal Engine, comparing its
advantages and limitations against traditional modeling workflows.
Additionally, we examined user preferences between designed 3D assets and
photogrammetry-generated models. The results revealed that while photogrammetry
enhances realism and interactivity, users slightly preferred manually designed
models for small, manipulable elements because of the level of detail. However,
from a developer perspective, RealityCapture significantly reduces development
time while maintaining geometric precision and photorealistic textures. Despite
its reliance on high-performance hardware, its automation, scalability, and
seamless integration with real-time rendering engines make it a valuable tool
for game developers and VR creators. Future improvements in AI-driven
optimization and cloud-based processing could enhance accessibility, broadening
its applications in gaming, cultural heritage preservation, and simulation.

### 2. [Dynamic Caustics by Ultrasonically Modulated Liquid Surface](http://arxiv.org/pdf/2505.16397v1)

Authors: Koki Nagakura, Tatsuki Fushimi, Ayaka Tsutsui, Yoichi Ochiai

This paper presents a method for generating dynamic caustic patterns by
utilising dual-optimised holographic fields with Phased Array Transducer (PAT).
Building on previous research in static caustic optimisation and ultrasonic
manipulation, this approach employs computational techniques to dynamically
shape fluid surfaces, thereby creating controllable and real-time caustic
images. The system employs a Digital Twin framework, which enables iterative
feedback and refinement, thereby improving the accuracy and quality of the
caustic patterns produced. This paper extends the foundational work in caustic
generation by integrating liquid surfaces as refractive media. This concept has
previously been explored in simulations but not fully realised in practical
applications. The utilisation of ultrasound to directly manipulate these
surfaces enables the generation of dynamic caustics with a high degree of
flexibility. The Digital Twin approach further enhances this process by
allowing for precise adjustments and optimisation based on real-time feedback.
Experimental results demonstrate the technique's capacity to generate
continuous animations and complex caustic patterns at high frequencies.
Although there are limitations in contrast and resolution compared to
solid-surface methods, this approach offers advantages in terms of real-time
adaptability and scalability. This technique has the potential to be applied in
a number of areas, including interactive displays, artistic installations and
educational tools. This research builds upon the work of previous researchers
in the fields of caustics optimisation, ultrasonic manipulation, and
computational displays. Future research will concentrate on enhancing the
resolution and intricacy of the generated patterns.

### Computer Science and Game Theory

### 1. [Persuasive Prediction via Decision Calibration](http://arxiv.org/pdf/2505.16141v1)

Authors: Jingwu Tang, Jiahao Zhang, Fei Fang, Zhiwei Steven Wu

Bayesian persuasion, a central model in information design, studies how a
sender, who privately observes a state drawn from a prior distribution,
strategically sends a signal to influence a receiver's action. A key assumption
is that both sender and receiver share the precise knowledge of the prior.
Although this prior can be estimated from past data, such assumptions break
down in high-dimensional or infinite state spaces, where learning an accurate
prior may require a prohibitive amount of data. In this paper, we study a
learning-based variant of persuasion, which we term persuasive prediction. This
setting mirrors Bayesian persuasion with large state spaces, but crucially does
not assume a common prior: the sender observes covariates $X$, learns to
predict a payoff-relevant outcome $Y$ from past data, and releases a prediction
to influence a population of receivers.
  To model rational receiver behavior without a common prior, we adopt a
learnable proxy: decision calibration, which requires the prediction to be
unbiased conditioned on the receiver's best response to the prediction. This
condition guarantees that myopically responding to the prediction yields no
swap regret. Assuming the receivers best respond to decision-calibrated
predictors, we design a computationally and statistically efficient algorithm
that learns a decision-calibrated predictor within a randomized predictor class
that optimizes the sender's utility. In the commonly studied single-receiver
case, our method matches the utility of a Bayesian sender who has full
knowledge of the underlying prior distribution. Finally, we extend our
algorithmic result to a setting where receivers respond stochastically to
predictions and the sender may randomize over an infinite predictor class.

### 2. [Strategic Content Creation in the Age of GenAI: To Share or Not to Share?](http://arxiv.org/pdf/2505.16358v1)

Authors: Gur Keinan, Omer Ben-Porat

We introduce a game-theoretic framework examining strategic interactions
between a platform and its content creators in the presence of AI-generated
content. Our model's main novelty is in capturing creators' dual strategic
decisions: The investment in content quality and their (possible) consent to
share their content with the platform's GenAI, both of which significantly
impact their utility. To incentivize creators, the platform strategically
allocates a portion of its GenAI-driven revenue to creators who share their
content. We focus on the class of full-sharing equilibrium profiles, in which
all creators willingly share their content with the platform's GenAI system.
Such equilibria are highly desirable both theoretically and practically. Our
main technical contribution is formulating and efficiently solving a novel
optimization problem that approximates the platform's optimal revenue subject
to inducing a full-sharing equilibrium. A key aspect of our approach is
identifying conditions under which full-sharing equilibria exist and a
surprising connection to the Prisoner's Dilemma. Finally, our simulations
demonstrate how revenue-allocation mechanisms affect creator utility and the
platform's revenue.

### 3. [Fairness under Competition](http://arxiv.org/pdf/2505.16291v1)

Authors: Ronen Gradwohl, Eilam Shapira, Moshe Tennenholtz

Algorithmic fairness has emerged as a central issue in ML, and it has become
standard practice to adjust ML algorithms so that they will satisfy fairness
requirements such as Equal Opportunity. In this paper we consider the effects
of adopting such fair classifiers on the overall level of ecosystem fairness.
Specifically, we introduce the study of fairness with competing firms, and
demonstrate the failure of fair classifiers in yielding fair ecosystems. Our
results quantify the loss of fairness in systems, under a variety of
conditions, based on classifiers' correlation and the level of their data
overlap. We show that even if competing classifiers are individually fair, the
ecosystem's outcome may be unfair; and that adjusting biased algorithms to
improve their individual fairness may lead to an overall decline in ecosystem
fairness. In addition to these theoretical results, we also provide supporting
experimental evidence. Together, our model and results provide a novel and
essential call for action.

### 4. [Serious Games: Human-AI Interaction, Evolution, and Coevolution](http://arxiv.org/pdf/2505.16388v1)

Authors: Nandini Doreswamy, Louise Horstmanshof

The serious games between humans and AI have only just begun. Evolutionary
Game Theory (EGT) models the competitive and cooperative strategies of
biological entities. EGT could help predict the potential evolutionary
equilibrium of humans and AI. The objective of this work was to examine some of
the EGT models relevant to human-AI interaction, evolution, and coevolution. Of
thirteen EGT models considered, three were examined: the Hawk-Dove Game,
Iterated Prisoner's Dilemma, and the War of Attrition. This selection was based
on the widespread acceptance and clear relevance of these models to potential
human-AI evolutionary dynamics and coevolutionary trajectories. The Hawk-Dove
Game predicts balanced mixed-strategy equilibria based on the costs of
conflict. It also shows the potential for balanced coevolution rather than
dominance. Iterated Prisoner's Dilemma suggests that repeated interaction may
lead to cognitive coevolution. It demonstrates how memory and reciprocity can
lead to cooperation. The War of Attrition suggests that competition for
resources may result in strategic coevolution, asymmetric equilibria, and
conventions on sharing resources. Therefore, EGT may provide a suitable
framework to understand and predict the human-AI evolutionary dynamic. However,
future research could extend beyond EGT and explore additional frameworks,
empirical validation methods, and interdisciplinary perspectives. AI is being
shaped by human input and is evolving in response to it. So too,
neuroplasticity allows the human brain to grow and evolve in response to
stimuli. If humans and AI converge in future, what might be the result of human
neuroplasticity combined with an ever-evolving AI? Future research should be
mindful of the ethical and cognitive implications of human-AI interaction,
evolution, and coevolution.

### 5. [Modeling Inequality in Complex Networks of Strategic Agents using Iterative Game-Theoretic Transactions](http://arxiv.org/pdf/2505.16966v1)

Authors: Mayank Kejriwal, Yuesheng Luo

Transactions are an important aspect of human social life, and represent
dynamic flow of information, intangible values, such as trust, as well as
monetary and social capital. Although much research has been conducted on the
nature of transactions in fields ranging from the social sciences to game
theory, the systemic effects of different types of agents transacting in
real-world social networks (often following a scale-free distribution) are not
fully understood. A particular systemic measure that has not received adequate
attention in the complex networks and game theory communities, is the Gini
Coefficient, which is widely used in economics to quantify and understand
wealth inequality. In part, the problem is a lack of experimentation using a
replicable algorithm and publicly available data. Motivated by this problem,
this article proposes a model and simulation algorithm, based on game theory,
for quantifying the evolution of inequality in complex networks of strategic
agents. Our results shed light on several complex drivers of inequality, even
in simple, abstract settings, and exhibit consistency across networks with
different origins and descriptions.

### 6. [Contextual Learning for Stochastic Optimization](http://arxiv.org/pdf/2505.16829v1)

Authors: Anna Heuser, Thomas Kesselheim

Motivated by stochastic optimization, we introduce the problem of learning
from samples of contextual value distributions. A contextual value distribution
can be understood as a family of real-valued distributions, where each sample
consists of a context $x$ and a random variable drawn from the corresponding
real-valued distribution $D_x$. By minimizing a convex surrogate loss, we learn
an empirical distribution $D'_x$ for each context, ensuring a small L\'evy
distance to $D_x$. We apply this result to obtain the sample complexity bounds
for the learning of an $\epsilon$-optimal policy for stochastic optimization
problems defined on an unknown contextual value distribution. The sample
complexity is shown to be polynomial for the general case of strongly monotone
and stable optimization problems, including Single-item Revenue Maximization,
Pandora's Box and Optimal Stopping.

### Human-Computer Interaction

### 1. ["If anybody finds out you are in BIG TROUBLE": Understanding Children's Hopes, Fears, and Evaluations of Generative AI](http://arxiv.org/pdf/2505.16089v1)

Authors: Aayushi Dangol, Robert Wolfe, Daeun Yoo, Arya Thiruvillakkat, Ben Chickadel, Julie A. Kientz

As generative artificial intelligence (genAI) increasingly mediates how
children learn, communicate, and engage with digital content, understanding
children's hopes and fears about this emerging technology is crucial. In a
pilot study with 37 fifth-graders, we explored how children (ages 9-10)
envision genAI and the roles they believe it should play in their daily life.
Our findings reveal three key ways children envision genAI: as a companion
providing guidance, a collaborator working alongside them, and a task automator
that offloads responsibilities. However, alongside these hopeful views,
children expressed fears about overreliance, particularly in academic settings,
linking it to fears of diminished learning, disciplinary consequences, and
long-term failure. This study highlights the need for child-centric AI design
that balances these tensions, empowering children with the skills to critically
engage with and navigate their evolving relationships with digital
technologies.

### 2. [Fairness and Efficiency in Human-Agent Teams: An Iterative Algorithm Design Approach](http://arxiv.org/pdf/2505.16171v1)

Authors: Mai Lee Chang, Kim Baraka, Greg Trafton, Zach Lalu Vazhekatt, Andrea Lockerd Thomaz

When agents interact with people as part of a team, fairness becomes an
important factor. Prior work has proposed fairness metrics based on teammates'
capabilities for task allocation within human-agent teams. However, most
metrics only consider teammate capabilities from a third-person point of view
(POV). In this work, we extend these metrics to include task preferences and
consider a first-person POV. We leverage an iterative design method consisting
of simulation data and human data to design a task allocation algorithm that
balances task efficiency and fairness based on both capabilities and
preferences. We first show that these metrics may not align with people's
perceived fairness from a first-person POV. In light of this result, we propose
a new fairness metric, fair-equity, and the Fair-Efficient Algorithm (FEA). Our
findings suggest that an agent teammate who balances efficiency and fairness
based on equity will be perceived to be fairer and preferred by human teammates
in various human-agent team types. We suggest that the perception of fairness
may also depend on a person's POV.

### 3. [Reassessing Collaborative Writing Theories and Frameworks in the Age of LLMs: What Still Applies and What We Must Leave Behind](http://arxiv.org/pdf/2505.16254v1)

Authors: Daisuke Yukita, Tim Miller, Joel Mackenzie

In this paper, we conduct a critical review of existing theories and
frameworks on human-human collaborative writing to assess their relevance to
the current human-AI paradigm in professional contexts, and draw seven insights
along with design implications for human-AI collaborative writing tools. We
found that, as LLMs nudge the writing process more towards an empirical "trial
and error" process analogous to prototyping, the non-linear cognitive process
of writing will stay the same, but more rigor will be required for revision
methodologies. This shift would shed further light on the importance of
coherence support, but the large language model (LLM)'s unprecedented semantic
capabilities can bring novel approaches to this ongoing challenge. We argue
that teamwork-related factors such as group awareness, consensus building and
authorship - which have been central in human-human collaborative writing
studies - should not apply to the human-AI paradigm due to excessive
anthropomorphism. With the LLM's text generation capabilities becoming
essentially indistinguishable from human-written ones, we are entering an era
where, for the first time in the history of computing, we are engaging in
collaborative writing with AI at workplaces on a daily basis. We aim to bring
theoretical grounding and practical design guidance to the interaction designs
of human-AI collaborative writing, with the goal of enhancing future human-AI
writing software.

### 4. [Estimating Perceptual Attributes of Haptic Textures Using Visuo-Tactile Data](http://arxiv.org/pdf/2505.16352v1)

Authors: Mudassir Ibrahim Awan, Seokhee Jeon

Accurate prediction of perceptual attributes of haptic textures is essential
for advancing VR and AR applications and enhancing robotic interaction with
physical surfaces. This paper presents a deep learning-based multi-modal
framework, incorporating visual and tactile data, to predict perceptual texture
ratings by leveraging multi-feature inputs. To achieve this, a four-dimensional
haptic attribute space encompassing rough-smooth, flat-bumpy, sticky-slippery,
and hard-soft dimensions is first constructed through psychophysical
experiments, where participants evaluate 50 diverse real-world texture samples.
A physical signal space is subsequently created by collecting visual and
tactile data from these textures. Finally, a deep learning architecture
integrating a CNN-based autoencoder for visual feature learning and a ConvLSTM
network for tactile data processing is trained to predict user-assigned
attribute ratings. This multi-modal, multi-feature approach maps physical
signals to perceptual ratings, enabling accurate predictions for unseen
textures. To evaluate predictive accuracy, we employed leave-one-out
cross-validation to rigorously assess the model's reliability and
generalizability against several machine learning and deep learning baselines.
Experimental results demonstrate that the framework consistently outperforms
single-modality approaches, achieving lower MAE and RMSE, highlighting the
efficacy of combining visual and tactile modalities.

### 5. [Truth and Trust: Fake News Detection via Biosignals](http://arxiv.org/pdf/2505.16702v1)

Authors: Gennie Nguyen, Lei Wang, Yangxueqing Jiang, Tom Gedeon

Understanding how individuals physiologically respond to false information is
crucial for advancing misinformation detection systems. This study explores the
potential of using physiological signals, specifically electrodermal activity
(EDA) and photoplethysmography (PPG), to classify both the veracity of
information and its interaction with user belief. In a controlled laboratory
experiment, we collected EDA and PPG signals while participants evaluated the
truthfulness of climate-related claims. Each trial was labeled based on the
objective truth of the claim and the participant's belief, enabling two
classification tasks: binary veracity detection and a novel four-class joint
belief-veracity classification. We extracted handcrafted features from the raw
signals and trained several machine learning models to benchmark the dataset.
Our results show that EDA outperforms PPG, indicating its greater sensitivity
to physiological responses related to truth perception. However, performance
significantly drops in the joint belief-veracity classification task,
highlighting the complexity of modeling the interaction between belief and
truth. These findings suggest that while physiological signals can reflect
basic truth perception, accurately modeling the intricate relationships between
belief and veracity remains a significant challenge. This study emphasizes the
importance of multimodal approaches that incorporate psychological,
physiological, and cognitive factors to improve fake news detection systems.
Our work provides a foundation for future research aimed at enhancing
misinformation detection via addressing the complexities of human belief and
truth processing.

### 6. [Detecting Fake News Belief via Skin and Blood Flow Signals](http://arxiv.org/pdf/2505.16730v1)

Authors: Gennie Nguyen, Lei Wang, Yangxueqing Jiang, Tom Gedeon

Misinformation poses significant risks to public opinion, health, and
security. While most fake news detection methods rely on text analysis, little
is known about how people physically respond to false information or repeated
exposure to the same statements. This study investigates whether wearable
sensors can detect belief in a statement or prior exposure to it. We conducted
a controlled experiment where participants evaluated statements while wearing
an EmotiBit sensor that measured their skin conductance (electrodermal
activity, EDA) and peripheral blood flow (photoplethysmography, PPG). From 28
participants, we collected a dataset of 672 trials, each labeled with whether
the participant believed the statement and whether they had seen it before.
This dataset introduces a new resource for studying physiological responses to
misinformation. Using machine learning models, including KNN, CNN, and
LightGBM, we analyzed these physiological patterns. The best-performing model
achieved 67.83\% accuracy, with skin conductance outperforming PPG. These
findings demonstrate the potential of wearable sensors as a minimally intrusive
tool for detecting belief and prior exposure, offering new directions for
real-time misinformation detection and adaptive, user-aware systems.

### 7. [Cracking Aegis: An Adversarial LLM-based Game for Raising Awareness of Vulnerabilities in Privacy Protection](http://arxiv.org/pdf/2505.16954v1)

Authors: Jiaying Fu, Yiyang Lu, Zehua Yang, Fiona Nah, RAY LC

Traditional methods for raising awareness of privacy protection often fail to
engage users or provide hands-on insights into how privacy vulnerabilities are
exploited. To address this, we incorporate an adversarial mechanic in the
design of the dialogue-based serious game Cracking Aegis. Leveraging LLMs to
simulate natural interactions, the game challenges players to impersonate
characters and extract sensitive information from an AI agent, Aegis. A user
study (n=22) revealed that players employed diverse deceptive linguistic
strategies, including storytelling and emotional rapport, to manipulate Aegis.
After playing, players reported connecting in-game scenarios with real-world
privacy vulnerabilities, such as phishing and impersonation, and expressed
intentions to strengthen privacy control, such as avoiding oversharing personal
information with AI systems. This work highlights the potential of LLMs to
simulate complex relational interactions in serious games, while demonstrating
how an adversarial game strategy provides unique insights for designs for
social good, particularly privacy protection.

### 8. [MAGE: A Multi-task Architecture for Gaze Estimation with an Efficient Calibration Module](http://arxiv.org/pdf/2505.16384v1)

Authors: Haoming Huang, Musen Zhang, Jianxin Yang, Zhen Li, Jinkai Li, Yao Guo

Eye gaze can provide rich information on human psychological activities, and
has garnered significant attention in the field of Human-Robot Interaction
(HRI). However, existing gaze estimation methods merely predict either the gaze
direction or the Point-of-Gaze (PoG) on the screen, failing to provide
sufficient information for a comprehensive six Degree-of-Freedom (DoF) gaze
analysis in 3D space. Moreover, the variations of eye shape and structure among
individuals also impede the generalization capability of these methods. In this
study, we propose MAGE, a Multi-task Architecture for Gaze Estimation with an
efficient calibration module, to predict the 6-DoF gaze information that is
applicable for the real-word HRI. Our basic model encodes both the directional
and positional features from facial images, and predicts gaze results with
dedicated information flow and multiple decoders. To reduce the impact of
individual variations, we propose a novel calibration module, namely
Easy-Calibration, to fine-tune the basic model with subject-specific data,
which is efficient to implement without the need of a screen. Experimental
results demonstrate that our method achieves state-of-the-art performance on
the public MPIIFaceGaze, EYEDIAP, and our built IMRGaze datasets.

### 9. [Dynamic Caustics by Ultrasonically Modulated Liquid Surface](http://arxiv.org/pdf/2505.16397v1)

Authors: Koki Nagakura, Tatsuki Fushimi, Ayaka Tsutsui, Yoichi Ochiai

This paper presents a method for generating dynamic caustic patterns by
utilising dual-optimised holographic fields with Phased Array Transducer (PAT).
Building on previous research in static caustic optimisation and ultrasonic
manipulation, this approach employs computational techniques to dynamically
shape fluid surfaces, thereby creating controllable and real-time caustic
images. The system employs a Digital Twin framework, which enables iterative
feedback and refinement, thereby improving the accuracy and quality of the
caustic patterns produced. This paper extends the foundational work in caustic
generation by integrating liquid surfaces as refractive media. This concept has
previously been explored in simulations but not fully realised in practical
applications. The utilisation of ultrasound to directly manipulate these
surfaces enables the generation of dynamic caustics with a high degree of
flexibility. The Digital Twin approach further enhances this process by
allowing for precise adjustments and optimisation based on real-time feedback.
Experimental results demonstrate the technique's capacity to generate
continuous animations and complex caustic patterns at high frequencies.
Although there are limitations in contrast and resolution compared to
solid-surface methods, this approach offers advantages in terms of real-time
adaptability and scalability. This technique has the potential to be applied in
a number of areas, including interactive displays, artistic installations and
educational tools. This research builds upon the work of previous researchers
in the fields of caustics optimisation, ultrasonic manipulation, and
computational displays. Future research will concentrate on enhancing the
resolution and intricacy of the generated patterns.

### 10. [Sparse Activation Editing for Reliable Instruction Following in Narratives](http://arxiv.org/pdf/2505.16505v1)

Authors: Runcong Zhao, Chengyu Cao, Qinglin Zhu, Xiucheng Lv, Shun Shao, Lin Gui, Ruifeng Xu, Yulan He

Complex narrative contexts often challenge language models' ability to follow
instructions, and existing benchmarks fail to capture these difficulties. To
address this, we propose Concise-SAE, a training-free framework that improves
instruction following by identifying and editing instruction-relevant neurons
using only natural language instructions, without requiring labelled data. To
thoroughly evaluate our method, we introduce FreeInstruct, a diverse and
realistic benchmark of 1,212 examples that highlights the challenges of
instruction following in narrative-rich settings. While initially motivated by
complex narratives, Concise-SAE demonstrates state-of-the-art instruction
adherence across varied tasks without compromising generation quality.

### Information Retrieval

### 1. [Emotion-based Recommender System](http://arxiv.org/pdf/2505.16121v1)

Authors: Hao Wang

Recommender system is one of the most critical technologies for large
internet companies such as Amazon and TikTok. Although millions of users use
recommender systems globally everyday, and indeed, much data analysis work has
been done to improve the technical accuracy of the system, to our limited
knowledge, there has been little attention paid to analysis of users' emotion
in recommender systems. In this paper, we create a new theory and metrics that
could capture users' emotion when they are interacting with recommender
systems. We also provide effective and efficient visualization techniques for
visualization of users' emotion and its change in the customers' lifetime
cycle. In the end, we design a framework for emotion-based recommendation
algorithms, illustrated in a straightforward example with experimental results
to demonstrate the effectiveness of our new theory.

### 2. [HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](http://arxiv.org/pdf/2505.16133v1)

Authors: Jinyu Guo, Xunlei Chen, Qiyang Xia, Zhaokun Wang, Jie Ou, Libo Qin, Shunyu Yao, Wenhong Tian

Retrieval-Augmented Generation (RAG) encounters efficiency challenges when
scaling to massive knowledge bases while preserving contextual relevance. We
propose Hash-RAG, a framework that integrates deep hashing techniques with
systematic optimizations to address these limitations. Our queries directly
learn binary hash codes from knowledgebase code, eliminating intermediate
feature extraction steps, and significantly reducing storage and computational
overhead. Building upon this hash-based efficient retrieval framework, we
establish the foundation for fine-grained chunking. Consequently, we design a
Prompt-Guided Chunk-to-Context (PGCC) module that leverages retrieved
hash-indexed propositions and their original document segments through prompt
engineering to enhance the LLM's contextual awareness. Experimental evaluations
on NQ, TriviaQA, and HotpotQA datasets demonstrate that our approach achieves a
90% reduction in retrieval time compared to conventional methods while
maintaining considerate recall performance. Additionally, The proposed system
outperforms retrieval/non-retrieval baselines by 1.4-4.3% in EM scores.

### 3. [Flow Matching based Sequential Recommender Model](http://arxiv.org/pdf/2505.16298v1)

Authors: Feng Liu, Lixin Zou, Xiangyu Zhao, Min Tang, Liming Dong, Dan Luo, Xiangyang Luo, Chenliang Li

Generative models, particularly diffusion model, have emerged as powerful
tools for sequential recommendation. However, accurately modeling user
preferences remains challenging due to the noise perturbations inherent in the
forward and reverse processes of diffusion-based methods. Towards this end,
this study introduces FMRec, a Flow Matching based model that employs a
straight flow trajectory and a modified loss tailored for the recommendation
task. Additionally, from the diffusion-model perspective, we integrate a
reconstruction loss to improve robustness against noise perturbations, thereby
retaining user preferences during the forward process. In the reverse process,
we employ a deterministic reverse sampler, specifically an ODE-based updating
function, to eliminate unnecessary randomness, thereby ensuring that the
generated recommendations closely align with user needs. Extensive evaluations
on four benchmark datasets reveal that FMRec achieves an average improvement of
6.53% over state-of-the-art methods. The replication code is available at
https://github.com/FengLiu-1/FMRec.

### 4. [Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems](http://arxiv.org/pdf/2505.16367v1)

Authors: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

Retrieval-augmented generation (RAG) systems can effectively mitigate the
hallucination problem of large language models (LLMs),but they also possess
inherent vulnerabilities. Identifying these weaknesses before the large-scale
real-world deployment of RAG systems is of great importance, as it lays the
foundation for building more secure and robust RAG systems in the future.
Existing adversarial attack methods typically exploit knowledge base poisoning
to probe the vulnerabilities of RAG systems, which can effectively deceive
standard RAG models. However, with the rapid advancement of deep reasoning
capabilities in modern LLMs, previous approaches that merely inject incorrect
knowledge are inadequate when attacking RAG systems equipped with deep
reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this
paper extracts reasoning process templates from R1-based RAG systems, uses
these templates to wrap erroneous knowledge into adversarial documents, and
injects them into the knowledge base to attack RAG systems. The key idea of our
approach is that adversarial documents, by simulating the chain-of-thought
patterns aligned with the model's training signals, may be misinterpreted by
the model as authentic historical reasoning processes, thus increasing their
likelihood of being referenced. Experiments conducted on the MS MARCO passage
ranking dataset demonstrate the effectiveness of our proposed method.

### 5. [Causal-Invariant Cross-Domain Out-of-Distribution Recommendation](http://arxiv.org/pdf/2505.16532v1)

Authors: Jiajie Zhu, Yan Wang, Feng Zhu, Pengfei Ding, Hongyang Liu, Zhu Sun

Cross-Domain Recommendation (CDR) aims to leverage knowledge from a
relatively data-richer source domain to address the data sparsity problem in a
relatively data-sparser target domain. While CDR methods need to address the
distribution shifts between different domains, i.e., cross-domain distribution
shifts (CDDS), they typically assume independent and identical distribution
(IID) between training and testing data within the target domain. However, this
IID assumption rarely holds in real-world scenarios due to single-domain
distribution shift (SDDS). The above two co-existing distribution shifts lead
to out-of-distribution (OOD) environments that hinder effective knowledge
transfer and generalization, ultimately degrading recommendation performance in
CDR. To address these co-existing distribution shifts, we propose a novel
Causal-Invariant Cross-Domain Out-of-distribution Recommendation framework,
called CICDOR. In CICDOR, we first learn dual-level causal structures to infer
domain-specific and domain-shared causal-invariant user preferences for
tackling both CDDS and SDDS under OOD environments in CDR. Then, we propose an
LLM-guided confounder discovery module that seamlessly integrates LLMs with a
conventional causal discovery method to extract observed confounders for
effective deconfounding, thereby enabling accurate causal-invariant preference
inference. Extensive experiments on two real-world datasets demonstrate the
superior recommendation accuracy of CICDOR over state-of-the-art methods across
various OOD scenarios.

### 6. [MDVT: Enhancing Multimodal Recommendation with Model-Agnostic Multimodal-Driven Virtual Triplets](http://arxiv.org/pdf/2505.16665v1)

Authors: Jinfeng Xu, Zheyu Chen, Jinze Li, Shuo Yang, Hewei Wang, Yijie Li, Mengran Li, Puzhen Wu, Edith C. H. Ngai

The data sparsity problem significantly hinders the performance of
recommender systems, as traditional models rely on limited historical
interactions to learn user preferences and item properties. While incorporating
multimodal information can explicitly represent these preferences and
properties, existing works often use it only as side information, failing to
fully leverage its potential. In this paper, we propose MDVT, a model-agnostic
approach that constructs multimodal-driven virtual triplets to provide valuable
supervision signals, effectively mitigating the data sparsity problem in
multimodal recommendation systems. To ensure high-quality virtual triplets, we
introduce three tailored warm-up threshold strategies: static, dynamic, and
hybrid. The static warm-up threshold strategy exhaustively searches for the
optimal number of warm-up epochs but is time-consuming and computationally
intensive. The dynamic warm-up threshold strategy adjusts the warm-up period
based on loss trends, improving efficiency but potentially missing optimal
performance. The hybrid strategy combines both, using the dynamic strategy to
find the approximate optimal number of warm-up epochs and then refining it with
the static strategy in a narrow hyper-parameter space. Once the warm-up
threshold is satisfied, the virtual triplets are used for joint model
optimization by our enhanced pair-wise loss function without causing
significant gradient skew. Extensive experiments on multiple real-world
datasets demonstrate that integrating MDVT into advanced multimodal
recommendation models effectively alleviates the data sparsity problem and
improves recommendation performance, particularly in sparse data scenarios.

### 7. [A Novel Generative Model with Causality Constraint for Mitigating Biases in Recommender Systems](http://arxiv.org/pdf/2505.16708v1)

Authors: Jianfeng Deng, Qingfeng Chen, Debo Cheng, Jiuyong Li, Lin Liu, Shichao Zhang

Accurately predicting counterfactual user feedback is essential for building
effective recommender systems. However, latent confounding bias can obscure the
true causal relationship between user feedback and item exposure, ultimately
degrading recommendation performance. Existing causal debiasing approaches
often rely on strong assumptions-such as the availability of instrumental
variables (IVs) or strong correlations between latent confounders and proxy
variables-that are rarely satisfied in real-world scenarios. To address these
limitations, we propose a novel generative framework called Latent Causality
Constraints for Debiasing representation learning in Recommender Systems
(LCDR). Specifically, LCDR leverages an identifiable Variational Autoencoder
(iVAE) as a causal constraint to align the latent representations learned by a
standard Variational Autoencoder (VAE) through a unified loss function. This
alignment allows the model to leverage even weak or noisy proxy variables to
recover latent confounders effectively. The resulting representations are then
used to improve recommendation performance. Extensive experiments on three
real-world datasets demonstrate that LCDR consistently outperforms existing
methods in both mitigating bias and improving recommendation accuracy.

### 8. [DeepRec: Towards a Deep Dive Into the Item Space with Large Language Model Based Recommendation](http://arxiv.org/pdf/2505.16810v1)

Authors: Bowen Zheng, Xiaolei Wang, Enze Liu, Xi Wang, Lu Hongyu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen

Recently, large language models (LLMs) have been introduced into recommender
systems (RSs), either to enhance traditional recommendation models (TRMs) or
serve as recommendation backbones. However, existing LLM-based RSs often do not
fully exploit the complementary advantages of LLMs (e.g., world knowledge and
reasoning) and TRMs (e.g., recommendation-specific knowledge and efficiency) to
fully explore the item space. To address this, we propose DeepRec, a novel
LLM-based RS that enables autonomous multi-turn interactions between LLMs and
TRMs for deep exploration of the item space. In each interaction turn, LLMs
reason over user preferences and interact with TRMs to retrieve candidate
items. After multi-turn interactions, LLMs rank the retrieved items to generate
the final recommendations. We adopt reinforcement learning(RL) based
optimization and propose novel designs from three aspects: recommendation model
based data rollout, recommendation-oriented hierarchical rewards, and a
two-stage RL training strategy. For data rollout, we introduce a
preference-aware TRM, with which LLMs interact to construct trajectory data.
For rewards, we design a hierarchical reward function that involves both
process-level and outcome-level rewards to optimize the interaction process and
recommendation performance, respectively. For RL training, we develop a
two-stage training strategy, where the first stage aims to guide LLMs to
interact with TRMs and the second stage focuses on performance improvement.
Experiments on public datasets demonstrate that DeepRec significantly
outperforms both traditional and LLM-based baselines, offering a new paradigm
for deep exploration in recommendation systems.

### 9. [Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks](http://arxiv.org/pdf/2505.16849v1)

Authors: Martin Böckling, Heiko Paulheim, Andreea Iana

Large Language Models (LLMs) have showcased impressive reasoning abilities,
but often suffer from hallucinations or outdated knowledge. Knowledge Graph
(KG)-based Retrieval-Augmented Generation (RAG) remedies these shortcomings by
grounding LLM responses in structured external information from a knowledge
base. However, many KG-based RAG approaches struggle with (i) aligning KG and
textual representations, (ii) balancing retrieval accuracy and efficiency, and
(iii) adapting to dynamically updated KGs. In this work, we introduce
Walk&Retrieve, a simple yet effective KG-based framework that leverages
walk-based graph traversal and knowledge verbalization for corpus generation
for zero-shot RAG. Built around efficient KG walks, our method does not require
fine-tuning on domain-specific data, enabling seamless adaptation to KG
updates, reducing computational overhead, and allowing integration with any
off-the-shelf backbone LLM. Despite its simplicity, Walk&Retrieve performs
competitively, often outperforming existing RAG systems in response accuracy
and hallucination reduction. Moreover, it demonstrates lower query latency and
robust scalability to large KGs, highlighting the potential of lightweight
retrieval strategies as strong baselines for future RAG research.

### 10. [LARES: Latent Reasoning for Sequential Recommendation](http://arxiv.org/pdf/2505.16865v1)

Authors: Enze Liu, Bowen Zheng, Xiaolei Wang, Wayne Xin Zhao, Jinpeng Wang, Sheng Chen, Ji-Rong Wen

Sequential recommender systems have become increasingly important in
real-world applications that model user behavior sequences to predict their
preferences. However, existing sequential recommendation methods predominantly
rely on non-reasoning paradigms, which may limit the model's computational
capacity and result in suboptimal recommendation performance. To address these
limitations, we present LARES, a novel and scalable LAtent REasoning framework
for Sequential recommendation that enhances model's representation capabilities
through increasing the computation density of parameters by depth-recurrent
latent reasoning. Our proposed approach employs a recurrent architecture that
allows flexible expansion of reasoning depth without increasing parameter
complexity, thereby effectively capturing dynamic and intricate user interest
patterns. A key difference of LARES lies in refining all input tokens at each
implicit reasoning step to improve the computation utilization. To fully unlock
the model's reasoning potential, we design a two-phase training strategy: (1)
Self-supervised pre-training (SPT) with dual alignment objectives; (2)
Reinforcement post-training (RPT). During the first phase, we introduce
trajectory-level alignment and step-level alignment objectives, which enable
the model to learn recommendation-oriented latent reasoning patterns without
requiring supplementary annotated data. The subsequent phase utilizes
reinforcement learning (RL) to harness the model's exploratory ability, further
refining its reasoning capabilities. Comprehensive experiments on real-world
benchmarks demonstrate our framework's superior performance. Notably, LARES
exhibits seamless compatibility with existing advanced models, further
improving their recommendation performance.

### Machine Learning

### 1. [Reinforcement Learning for Stock Transactions](http://arxiv.org/pdf/2505.16099v1)

Authors: Ziyi, Zhou, Nicholas Stern, Julien Laasri

Much research has been done to analyze the stock market. After all, if one
can determine a pattern in the chaotic frenzy of transactions, then they could
make a hefty profit from capitalizing on these insights. As such, the goal of
our project was to apply reinforcement learning (RL) to determine the best time
to buy a stock within a given time frame. With only a few adjustments, our
model can be extended to identify the best time to sell a stock as well. In
order to use the format of free, real-world data to train the model, we define
our own Markov Decision Process (MDP) problem. These two papers [5] [6] helped
us in formulating the state space and the reward system of our MDP problem. We
train a series of agents using Q-Learning, Q-Learning with linear function
approximation, and deep Q-Learning. In addition, we try to predict the stock
prices using machine learning regression and classification models. We then
compare our agents to see if they converge on a policy, and if so, which one
learned the best policy to maximize profit on the stock market.

### 2. [Tools in the Loop: Quantifying Uncertainty of LLM Question Answering Systems That Use Tools](http://arxiv.org/pdf/2505.16113v1)

Authors: Panagiotis Lymperopoulos, Vasanth Sarathy

Modern Large Language Models (LLMs) often require external tools, such as
machine learning classifiers or knowledge retrieval systems, to provide
accurate answers in domains where their pre-trained knowledge is insufficient.
This integration of LLMs with external tools expands their utility but also
introduces a critical challenge: determining the trustworthiness of responses
generated by the combined system. In high-stakes applications, such as medical
decision-making, it is essential to assess the uncertainty of both the LLM's
generated text and the tool's output to ensure the reliability of the final
response. However, existing uncertainty quantification methods do not account
for the tool-calling scenario, where both the LLM and external tool contribute
to the overall system's uncertainty. In this work, we present a novel framework
for modeling tool-calling LLMs that quantifies uncertainty by jointly
considering the predictive uncertainty of the LLM and the external tool. We
extend previous methods for uncertainty quantification over token sequences to
this setting and propose efficient approximations that make uncertainty
computation practical for real-world applications. We evaluate our framework on
two new synthetic QA datasets, derived from well-known machine learning
datasets, which require tool-calling for accurate answers. Additionally, we
apply our method to retrieval-augmented generation (RAG) systems and conduct a
proof-of-concept experiment demonstrating the effectiveness of our uncertainty
metrics in scenarios where external information retrieval is needed. Our
results show that the framework is effective in enhancing trust in LLM-based
systems, especially in cases where the LLM's internal knowledge is insufficient
and external tools are required.

### 3. [A Generic Framework for Conformal Fairness](http://arxiv.org/pdf/2505.16115v1)

Authors: Aditya T. Vadlamani, Anutam Srinivasan, Pranav Maneriker, Ali Payani, Srinivasan Parthasarathy

Conformal Prediction (CP) is a popular method for uncertainty quantification
with machine learning models. While conformal prediction provides probabilistic
guarantees regarding the coverage of the true label, these guarantees are
agnostic to the presence of sensitive attributes within the dataset. In this
work, we formalize \textit{Conformal Fairness}, a notion of fairness using
conformal predictors, and provide a theoretically well-founded algorithm and
associated framework to control for the gaps in coverage between different
sensitive groups. Our framework leverages the exchangeability assumption
(implicit to CP) rather than the typical IID assumption, allowing us to apply
the notion of Conformal Fairness to data types and tasks that are not IID, such
as graph data. Experiments were conducted on graph and tabular datasets to
demonstrate that the algorithm can control fairness-related gaps in addition to
coverage aligned with theoretical expectations.

### 4. [Plan and Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning](http://arxiv.org/pdf/2505.16122v1)

Authors: Junhong Lin, Xinyue Zeng, Jie Zhu, Song Wang, Julian Shun, Jun Wu, Dawei Zhou

Large Language Models (LLMs) have achieved remarkable success in complex
reasoning tasks, but their inference remains computationally inefficient. We
observe a common failure mode in many prevalent LLMs, overthinking, where
models generate verbose and tangential reasoning traces even for simple
queries. Recent works have tried to mitigate this by enforcing fixed token
budgets, however, this can lead to underthinking, especially on harder
problems. Through empirical analysis, we identify that this inefficiency often
stems from unclear problem-solving strategies. To formalize this, we develop a
theoretical model, BBAM (Bayesian Budget Allocation Model), which models
reasoning as a sequence of sub-questions with varying uncertainty, and
introduce the $E^3$ metric to capture the trade-off between correctness and
computation efficiency. Building on theoretical results from BBAM, we propose
Plan-and-Budget, a model-agnostic, test-time framework that decomposes complex
queries into sub-questions and allocates token budgets based on estimated
complexity using adaptive scheduling. Plan-and-Budget improves reasoning
efficiency across a range of tasks and models, achieving up to +70% accuracy
gains, -39% token reduction, and +187.5% improvement in $E^3$. Notably, it
elevates a smaller model (DS-Qwen-32B) to match the efficiency of a larger
model (DS-LLaMA-70B)-demonstrating Plan-and-Budget's ability to close
performance gaps without retraining. Our code is available at
anonymous.4open.science/r/P-and-B-6513/.

### 5. [Robust Invariant Representation Learning by Distribution Extrapolation](http://arxiv.org/pdf/2505.16126v1)

Authors: Kotaro Yoshida, Slavakis Konstantinos

Invariant risk minimization (IRM) aims to enable out-of-distribution (OOD)
generalization in deep learning by learning invariant representations. As IRM
poses an inherently challenging bi-level optimization problem, most existing
approaches -- including IRMv1 -- adopt penalty-based single-level
approximations. However, empirical studies consistently show that these methods
often fail to outperform well-tuned empirical risk minimization (ERM),
highlighting the need for more robust IRM implementations. This work
theoretically identifies a key limitation common to many IRM variants: their
penalty terms are highly sensitive to limited environment diversity and
over-parameterization, resulting in performance degradation. To address this
issue, a novel extrapolation-based framework is proposed that enhances
environmental diversity by augmenting the IRM penalty through synthetic
distributional shifts. Extensive experiments -- ranging from synthetic setups
to realistic, over-parameterized scenarios -- demonstrate that the proposed
method consistently outperforms state-of-the-art IRM variants, validating its
effectiveness and robustness.

### 6. [Why Can Accurate Models Be Learned from Inaccurate Annotations?](http://arxiv.org/pdf/2505.16159v1)

Authors: Chongjie Si, Yidan Cui, Fuchao Yang, Xiaokang Yang, Wei Shen

Learning from inaccurate annotations has gained significant attention due to
the high cost of precise labeling. However, despite the presence of erroneous
labels, models trained on noisy data often retain the ability to make accurate
predictions. This intriguing phenomenon raises a fundamental yet largely
unexplored question: why models can still extract correct label information
from inaccurate annotations remains unexplored. In this paper, we conduct a
comprehensive investigation into this issue. By analyzing weight matrices from
both empirical and theoretical perspectives, we find that label inaccuracy
primarily accumulates noise in lower singular components and subtly perturbs
the principal subspace. Within a certain range, the principal subspaces of
weights trained on inaccurate labels remain largely aligned with those learned
from clean labels, preserving essential task-relevant information. We formally
prove that the angles of principal subspaces exhibit minimal deviation under
moderate label inaccuracy, explaining why models can still generalize
effectively. Building on these insights, we propose LIP, a lightweight plug-in
designed to help classifiers retain principal subspace information while
mitigating noise induced by label inaccuracy. Extensive experiments on tasks
with various inaccuracy conditions demonstrate that LIP consistently enhances
the performance of existing algorithms. We hope our findings can offer valuable
theoretical and practical insights to understand of model robustness under
inaccurate supervision.

### 7. [Enhancing Federated Survival Analysis through Peer-Driven Client Reputation in Healthcare](http://arxiv.org/pdf/2505.16190v1)

Authors: Navid Seidi, Satyaki Roy, Sajal Das

Federated Learning (FL) holds great promise for digital health by enabling
collaborative model training without compromising patient data privacy.
However, heterogeneity across institutions, lack of sustained reputation, and
unreliable contributions remain major challenges. In this paper, we propose a
robust, peer-driven reputation mechanism for federated healthcare that employs
a hybrid communication model to integrate decentralized peer feedback with
clustering-based noise handling to enhance model aggregation. Crucially, our
approach decouples the federated aggregation and reputation mechanisms by
applying differential privacy to client-side model updates before sharing them
for peer evaluation. This ensures sensitive information remains protected
during reputation computation, while unaltered updates are sent to the server
for global model training. Using the Cox Proportional Hazards model for
survival analysis across multiple federated nodes, our framework addresses both
data heterogeneity and reputation deficit by dynamically adjusting trust scores
based on local performance improvements measured via the concordance index.
Experimental evaluations on both synthetic datasets and the SEER dataset
demonstrate that our method consistently achieves high and stable C-index
values, effectively down-weighing noisy client updates and outperforming FL
methods that lack a reputation system.

### 8. [Reward-Aware Proto-Representations in Reinforcement Learning](http://arxiv.org/pdf/2505.16217v1)

Authors: Hon Tik Tse, Siddarth Chandrasekar, Marlos C. Machado

In recent years, the successor representation (SR) has attracted increasing
attention in reinforcement learning (RL), and it has been used to address some
of its key challenges, such as exploration, credit assignment, and
generalization. The SR can be seen as representing the underlying credit
assignment structure of the environment by implicitly encoding its induced
transition dynamics. However, the SR is reward-agnostic. In this paper, we
discuss a similar representation that also takes into account the reward
dynamics of the problem. We study the default representation (DR), a recently
proposed representation with limited theoretical (and empirical) analysis.
Here, we lay some of the theoretical foundation underlying the DR in the
tabular case by (1) deriving dynamic programming and (2) temporal-difference
methods to learn the DR, (3) characterizing the basis for the vector space of
the DR, and (4) formally extending the DR to the function approximation case
through default features. Empirically, we analyze the benefits of the DR in
many of the settings in which the SR has been applied, including (1) reward
shaping, (2) option discovery, (3) exploration, and (4) transfer learning. Our
results show that, compared to the SR, the DR gives rise to qualitatively
different, reward-aware behaviour and quantitatively better performance in
several settings.

### 9. [Realistic Evaluation of TabPFN v2 in Open Environments](http://arxiv.org/pdf/2505.16226v1)

Authors: Zi-Jian Cheng, Zi-Yi Jia, Zhi Zhou, Yu-Feng Li, Lan-Zhe Guo

Tabular data, owing to its ubiquitous presence in real-world domains, has
garnered significant attention in machine learning research. While tree-based
models have long dominated tabular machine learning tasks, the recently
proposed deep learning model TabPFN v2 has emerged, demonstrating unparalleled
performance and scalability potential. Although extensive research has been
conducted on TabPFN v2 to further improve performance, the majority of this
research remains confined to closed environments, neglecting the challenges
that frequently arise in open environments. This raises the question: Can
TabPFN v2 maintain good performance in open environments? To this end, we
conduct the first comprehensive evaluation of TabPFN v2's adaptability in open
environments. We construct a unified evaluation framework covering various
real-world challenges and assess the robustness of TabPFN v2 under open
environments scenarios using this framework. Empirical results demonstrate that
TabPFN v2 shows significant limitations in open environments but is suitable
for small-scale, covariate-shifted, and class-balanced tasks. Tree-based models
remain the optimal choice for general tabular tasks in open environments. To
facilitate future research on open environments challenges, we advocate for
open environments tabular benchmarks, multi-metric evaluation, and universal
modules to strengthen model robustness. We publicly release our evaluation
framework at https://anonymous.4open.science/r/tabpfn-ood-4E65.

### 10. [Graph Neural Network-Based Collaborative Perception for Adaptive Scheduling in Distributed Systems](http://arxiv.org/pdf/2505.16248v1)

Authors: Wenxuan Zhu, Qiyuan Wu, Tengda Tang, Renzi Meng, Sheng Chai, Xuehui Quan

This paper addresses the limitations of multi-node perception and delayed
scheduling response in distributed systems by proposing a GNN-based multi-node
collaborative perception mechanism. The system is modeled as a graph structure.
Message-passing and state-update modules are introduced. A multi-layer graph
neural network is constructed to enable efficient information aggregation and
dynamic state inference among nodes. In addition, a perception representation
method is designed by fusing local states with global features. This improves
each node's ability to perceive the overall system status. The proposed method
is evaluated within a customized experimental framework. A dataset featuring
heterogeneous task loads and dynamic communication topologies is used.
Performance is measured in terms of task completion rate, average latency, load
balancing, and transmission efficiency. Experimental results show that the
proposed method outperforms mainstream algorithms under various conditions,
including limited bandwidth and dynamic structural changes. It demonstrates
superior perception capabilities and cooperative scheduling performance. The
model achieves rapid convergence and efficient responses to complex system
states.

### Neural and Evolutionary Computing

### 1. [CMA-ES with Radial Basis Function Surrogate for Black-Box Optimization](http://arxiv.org/pdf/2505.16127v1)

Authors: Farshid Farhadi Khouzani, Abdolreza Mirzaei, Paul La Plante, Laxmi Gewali

Evolutionary optimization algorithms often face defects and limitations that
complicate the evolution processes or even prevent them from reaching the
global optimum. A notable constraint pertains to the considerable quantity of
function evaluations required to achieve the intended solution. This concern
assumes heightened significance when addressing costly optimization problems.
However, recent research has shown that integrating machine learning methods,
specifically surrogate models, with evolutionary optimization can enhance
various aspects of these algorithms. Among the evolutionary algorithms, the
Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES) is particularly
favored. This preference is due to its use of Gaussian distribution for
calculating evolution and its ability to adapt optimization parameters, which
reduces the need for user intervention in adjusting initial parameters. In this
research endeavor, we propose the adoption of surrogate models within the
CMA-ES framework called CMA-SAO to develop an initial surrogate model that
facilitates the adaptation of optimization parameters through the acquisition
of pertinent information derived from the associated surrogate model. Empirical
validation reveals that CMA-SAO algorithm markedly diminishes the number of
function evaluations in comparison to prevailing algorithms, thereby providing
a significant enhancement in operational efficiency.

### 2. [Neuromorphic-based metaheuristics: A new generation of low power, low latency and small footprint optimization algorithms](http://arxiv.org/pdf/2505.16362v1)

Authors: El-ghazali Talbi

Neuromorphic computing (NC) introduces a novel algorithmic paradigm
representing a major shift from traditional digital computing of Von Neumann
architectures. NC emulates or simulates the neural dynamics of brains in the
form of Spiking Neural Networks (SNNs). Much of the research in NC has
concentrated on machine learning applications and neuroscience simulations.
This paper investigates the modelling and implementation of optimization
algorithms and particularly metaheuristics using the NC paradigm as an
alternative to Von Neumann architectures, leading to breakthroughs in solving
optimization problems.
  Neuromorphic-based metaheuristics (Nheuristics) are supposed to be
characterized by low power, low latency and small footprint. Since NC systems
are fundamentally different from conventional Von Neumann computers, several
challenges are posed to the design and implementation of Nheuristics. A
guideline based on a classification and critical analysis is conducted on the
different families of metaheuristics and optimization problems they address. We
also discuss future directions that need to be addressed to expand both the
development and application of Nheuristics.

### 3. [Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](http://arxiv.org/pdf/2505.16471v1)

Authors: Robbert Reijnen, Yaoxin Wu, Zaharah Bukhsh, Yingqian Zhang

Deep reinforcement learning (DRL) has been widely used for dynamic algorithm
configuration, particularly in evolutionary computation, which benefits from
the adaptive update of parameters during the algorithmic execution. However,
applying DRL to algorithm configuration for multi-objective combinatorial
optimization (MOCO) problems remains relatively unexplored. This paper presents
a novel graph neural network (GNN) based DRL to configure multi-objective
evolutionary algorithms. We model the dynamic algorithm configuration as a
Markov decision process, representing the convergence of solutions in the
objective space by a graph, with their embeddings learned by a GNN to enhance
the state representation. Experiments on diverse MOCO challenges indicate that
our method outperforms traditional and DRL-based algorithm configuration
methods in terms of efficacy and adaptability. It also exhibits advantageous
generalizability across objective types and problem sizes, and applicability to
different evolutionary computation methods.

### 4. [Minimizing the energy depletion in wireless rechargeable sensor networks using bi-level metaheuristic charging schemes](http://arxiv.org/pdf/2505.16482v1)

Authors: Huynh Thi Thanh Binh, Le Van Cuong, Dang Hai Dang, Le Trong Vinh

Recently, Wireless Rechargeable Sensor Networks (WRSNs) that leveraged the
advantage of wireless energy transfer technology have opened a promising
opportunity in solving the limited energy issue. However, an ineffective
charging strategy may reduce the charging performance. Although many practical
charging algorithms have been introduced, these studies mainly focus on
optimizing the charging path with a fully charging approach. This approach may
lead to the death of a series of sensors due to their extended charging
latency. This paper introduces a novel partial charging approach that follows a
bi-level optimized scheme to minimize energy depletion in WRSNs. We aim at
optimizing simultaneously two factors: the charging path and time. To
accomplish this, we first formulate a mathematical model of the investigated
problem. We then propose two approximate algorithms in which the optimization
of the charging path and the charging time are considered as the upper and
lower level, respectively. The first algorithm combines a Multi-start Local
Search method and a Genetic Algorithm to find a solution. The second algorithm
adopts a nested approach that utilizes the advantages of the Multitasking and
Covariance Matrix Adaptation Evolutionary Strategies. Experimental validations
on various network scenarios demonstrate that our proposed algorithms
outperform the existing works.

### 5. [Stochastic Forward-Forward Learning through Representational Dimensionality Compression](http://arxiv.org/pdf/2505.16649v1)

Authors: Zhichao Zhu, Yang Qi, Hengyuan Ma, Wenlian Lu, Jianfeng Feng

The Forward-Forward (FF) algorithm provides a bottom-up alternative to
backpropagation (BP) for training neural networks, relying on a layer-wise
"goodness" function to guide learning. Existing goodness functions, inspired by
energy-based learning (EBL), are typically defined as the sum of squared
post-synaptic activations, neglecting the correlations between neurons. In this
work, we propose a novel goodness function termed dimensionality compression
that uses the effective dimensionality (ED) of fluctuating neural responses to
incorporate second-order statistical structure. Our objective minimizes ED for
clamped inputs when noise is considered while maximizing it across the sample
distribution, promoting structured representations without the need to prepare
negative samples. We demonstrate that this formulation achieves competitive
performance compared to other non-BP methods. Moreover, we show that noise
plays a constructive role that can enhance generalization and improve inference
when predictions are derived from the mean of squared outputs, which is
equivalent to making predictions based on the energy term. Our findings
contribute to the development of more biologically plausible learning
algorithms and suggest a natural fit for neuromorphic computing, where
stochasticity is a computational resource rather than a nuisance. The code is
available at https://github.com/ZhichaoZhu/StochasticForwardForward

### 6. [Sufficient conditions for offline reactivation in recurrent neural networks](http://arxiv.org/pdf/2505.17003v1)

Authors: Nanda H. Krishna, Colin Bredenberg, Daniel Levenstein, Blake A. Richards, Guillaume Lajoie

During periods of quiescence, such as sleep, neural activity in many brain
circuits resembles that observed during periods of task engagement. However,
the precise conditions under which task-optimized networks can autonomously
reactivate the same network states responsible for online behavior is poorly
understood. In this study, we develop a mathematical framework that outlines
sufficient conditions for the emergence of neural reactivation in circuits that
encode features of smoothly varying stimuli. We demonstrate mathematically that
noisy recurrent networks optimized to track environmental state variables using
change-based sensory information naturally develop denoising dynamics, which,
in the absence of input, cause the network to revisit state configurations
observed during periods of online activity. We validate our findings using
numerical experiments on two canonical neuroscience tasks: spatial position
estimation based on self-motion cues, and head direction estimation based on
angular velocity cues. Overall, our work provides theoretical support for
modeling offline reactivation as an emergent consequence of task optimization
in noisy neural circuits.

### 7. [The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](http://arxiv.org/pdf/2505.16716v1)

Authors: Moritz Stargalla, Christoph Hertrich, Daniel Reichman

An established measure of the expressive power of a given ReLU neural network
is the number of linear regions into which it partitions the input space. There
exist many different, non-equivalent definitions of what a linear region
actually is. We systematically assess which papers use which definitions and
discuss how they relate to each other. We then analyze the computational
complexity of counting the number of such regions for the various definitions.
Generally, this turns out to be an intractable problem. We prove NP- and
#P-hardness results already for networks with one hidden layer and strong
hardness of approximation results for two or more hidden layers. Finally, on
the algorithmic side, we demonstrate that counting linear regions can at least
be achieved in polynomial space for some common definitions.

### Networking and Internet Architecture

### 1. [SONIC: Cost-Effective Web Access for Developing Countries](http://arxiv.org/pdf/2505.16519v1)

Authors: Ayush Pandey, Rohail Asim, Jean Louis K. E. Fendji, Talal Rahwan, Matteo Varvello, Yasir Zaki

Over 2.6 billion people remain without access to the Internet in 2025. This
phenomenon is especially pronounced in developing regions, where cost and
infrastructure limitations are major barriers to connectivity. In response, we
design SONIC, a low-cost, scalable data delivery system that builds on existing
infrastructures: FM radio for downlink broadcasting, and SMS for personalized
uplink. SONIC is motivated by the widespread availability of FM radio and SMS
infrastructure in developing regions, along with embedded FM radio tuners in
affordable mobile phones. SONIC offers several innovations to effectively
transmit Web content over sound over FM radio, in a reliable and compressed
form. For example, we transmit pre-rendered webpages and leverage pixel
interpolation to recover errors at the receiver. We further modify Android to
offer a simpler deployment pipeline, supporting a wide range of devices. We
deployed SONIC at an FM radio station in Cameroon for six weeks with 30
participants. Our results demonstrate a sustained downlink throughput of 10
kbps, less than 20% loss for a majority of transmissions with signal strength
above -90 dbM, and a strong user engagement across both Web browsing and
ChatGPT interactions.

### 2. [Recursive Offloading for LLM Serving in Multi-tier Networks](http://arxiv.org/pdf/2505.16502v1)

Authors: Zhiyuan Wu, Sheng Sun, Yuwei Wang, Min Liu, Bo Gao, Jinda Lu, Zheming Yang, Tian Wen

Heterogeneous device-edge-cloud computing infrastructures have become widely
adopted in telecommunication operators and Wide Area Networks (WANs), offering
multi-tier computational support for emerging intelligent services. With the
rapid proliferation of Large Language Model (LLM) services, efficiently
coordinating inference tasks and reducing communication overhead within these
multi-tier network architectures becomes a critical deployment challenge.
Existing LLM serving paradigms exhibit significant limitations: on-device
deployment supports only lightweight LLMs due to hardware constraints, while
cloud-centric deployment suffers from resource congestion and considerable
prompt communication overhead caused by frequent service requests during peak
periods. Although the model-cascading-based inference strategy adapts better to
multi-tier networks, its reliance on fine-grained, manually adjusted thresholds
makes it less responsive to dynamic network conditions and varying task
complexities. To address these challenges, we propose RecServe, a recursive
offloading framework tailored for LLM serving in multi-tier networks. RecServe
integrates a task-specific hierarchical confidence evaluation mechanism that
guides offloading decisions based on inferred task complexity in progressively
scaled LLMs across device, edge, and cloud tiers. To further enable intelligent
task routing across tiers, RecServe employs a sliding-window-based dynamic
offloading strategy with quantile interpolation, enabling real-time tracking of
historical confidence distributions and adaptive offloading threshold
adjustments. Experiments on eight datasets demonstrate that RecServe
outperforms CasServe in both service quality and communication efficiency, and
reduces the communication burden by over 50\% compared to centralized
cloud-based serving.

### 3. [Graph Attention Network for Optimal User Association in Wireless Networks](http://arxiv.org/pdf/2505.16347v1)

Authors: Javad Mirzaei, Jeebak Mitra, Gwenael Poitau

With increased 5G deployments, network densification is higher than ever to
support the exponentially high throughput requirements. However, this has meant
a significant increase in energy consumption, leading to higher operational
expenditure (OpEx) for network operators creating an acute need for
improvements in network energy savings (NES). A key determinant of operational
efficacy in cellular networks is the user association (UA) policy, as it
affects critical aspects like spectral efficiency, load balancing etc. and
therefore impacts the overall energy consumption of the network directly.
Furthermore, with cellular network topologies lending themselves well to
graphical abstractions, use of graphs in network optimization has gained
significant prominence. In this work, we propose and analyze a graphical
abstraction based optimization for UA in cellular networks to improve NES by
determining when energy saving features like cell switch off can be activated.
A comparison with legacy approaches establishes the superiority of the proposed
approach.

### 4. [Smaller, Smarter, Closer: The Edge of Collaborative Generative AI](http://arxiv.org/pdf/2505.16499v1)

Authors: Roberto Morabito, SiYoung Jang

The rapid adoption of generative AI (GenAI), particularly Large Language
Models (LLMs), has exposed critical limitations of cloud-centric deployments,
including latency, cost, and privacy concerns. Meanwhile, Small Language Models
(SLMs) are emerging as viable alternatives for resource-constrained edge
environments, though they often lack the capabilities of their larger
counterparts. This article explores the potential of collaborative inference
systems that leverage both edge and cloud resources to address these
challenges. By presenting distinct cooperation strategies alongside practical
design principles and experimental insights, we offer actionable guidance for
deploying GenAI across the computing continuum.

### 5. [Edge-First Language Model Inference: Models, Metrics, and Tradeoffs](http://arxiv.org/pdf/2505.16508v1)

Authors: SiYoung Jang, Roberto Morabito

The widespread adoption of Language Models (LMs) across industries is driving
interest in deploying these services across the computing continuum, from the
cloud to the network edge. This shift aims to reduce costs, lower latency, and
improve reliability and privacy. Small Language Models (SLMs), enabled by
advances in model compression, are central to this shift, offering a path to
on-device inference on resource-constrained edge platforms. This work examines
the interplay between edge and cloud deployments, starting from detailed
benchmarking of SLM capabilities on single edge devices, and extending to
distributed edge clusters. We identify scenarios where edge inference offers
comparable performance with lower costs, and others where cloud fallback
becomes essential due to limits in scalability or model capacity. Rather than
proposing a one-size-fits-all solution, we present platform-level comparisons
and design insights for building efficient, adaptive LM inference systems
across heterogeneous environments.

### 6. [LLM-Based Emulation of the Radio Resource Control Layer: Towards AI-Native RAN Protocols](http://arxiv.org/pdf/2505.16821v1)

Authors: Ziming liu, Bryan Liu, Alvaro Valcarce, Xiaoli Chu

Integrating large AI models (LAMs) into 6G mobile networks promises to
redefine protocol design and control-plane intelligence by enabling autonomous,
cognitive network operations. While industry concepts, such as ETSI's
Experiential Networked Intelligence (ENI), envision LAM-driven agents for
adaptive network slicing and intent-based management, practical implementations
still face challenges in protocol literacy and real-world deployment. This
paper presents an end-to-end demonstration of a LAM that generates
standards-compliant, ASN.1-encoded Radio Resource Control (RRC) messages as
part of control-plane procedures inside a gNB. We treat RRC messaging as a
domain-specific language and fine-tune a decoder-only transformer model (LLaMA
class) using parameter-efficient Low-Rank Adaptation (LoRA) on RRC messages
linearized to retain their ASN.1 syntactic structure before standard byte-pair
encoding tokenization. This enables combinatorial generalization over RRC
protocol states while minimizing training overhead. On 30k field-test
request-response pairs, our 8 B model achieves a median cosine similarity of
0.97 with ground-truth messages on an edge GPU -- a 61 % relative gain over a
zero-shot LLaMA-3 8B baseline -- indicating substantially improved structural
and semantic RRC fidelity. Overall, our results show that LAMs, when augmented
with Radio Access Network (RAN)-specific reasoning, can directly orchestrate
control-plane procedures, representing a stepping stone toward the AI-native
air-interface paradigm. Beyond RRC emulation, this work lays the groundwork for
future AI-native wireless standards.

### Robotics

### 1. [Event-based Reconfiguration Control for Time-varying Formation of Robot Swarms in Narrow Spaces](http://arxiv.org/pdf/2505.16087v1)

Authors: Duy-Nam Bui, Manh Duong Phung, Hung Pham Duy

This study proposes an event-based reconfiguration control to navigate a
robot swarm through challenging environments with narrow passages such as
valleys, tunnels, and corridors. The robot swarm is modeled as an undirected
graph, where each node represents a robot capable of collecting real-time data
on the environment and the states of other robots in the formation. This data
serves as the input for the controller to provide dynamic adjustments between
the desired and straight-line configurations. The controller incorporates a set
of behaviors, designed using artificial potential fields, to meet the
requirements of goal-oriented motion, formation maintenance, tailgating, and
collision avoidance. The stability of the formation control is guaranteed via
the Lyapunov theorem. Simulation and comparison results show that the proposed
controller not only successfully navigates the robot swarm through narrow
spaces but also outperforms other established methods in key metrics including
the success rate, heading order, speed, travel time, and energy efficiency.
Software-in-the-loop tests have also been conducted to validate the
controller's applicability in practical scenarios. The source code of the
controller is available at https://github.com/duynamrcv/erc.

### 2. [Tactile-based Reinforcement Learning for Adaptive Grasping under Observation Uncertainties](http://arxiv.org/pdf/2505.16167v1)

Authors: Xiao Hu, Yang Ye

Robotic manipulation in industrial scenarios such as construction commonly
faces uncertain observations in which the state of the manipulating object may
not be accurately captured due to occlusions and partial observables. For
example, object status estimation during pipe assembly, rebar installation, and
electrical installation can be impacted by observation errors. Traditional
vision-based grasping methods often struggle to ensure robust stability and
adaptability. To address this challenge, this paper proposes a tactile
simulator that enables a tactile-based adaptive grasping method to enhance
grasping robustness. This approach leverages tactile feedback combined with the
Proximal Policy Optimization (PPO) reinforcement learning algorithm to
dynamically adjust the grasping posture, allowing adaptation to varying
grasping conditions under inaccurate object state estimations. Simulation
results demonstrate that the proposed method effectively adapts grasping
postures, thereby improving the success rate and stability of grasping tasks.

### 3. [Behavioral Safety Assessment towards Large-scale Deployment of Autonomous Vehicles](http://arxiv.org/pdf/2505.16214v1)

Authors: Henry X. Liu, Xintao Yan, Haowei Sun, Tinghan Wang, Zhijie Qiao, Haojie Zhu, Shengyin Shen, Shuo Feng, Greg Stevens, Greg McGuire

Autonomous vehicles (AVs) have significantly advanced in real-world
deployment in recent years, yet safety continues to be a critical barrier to
widespread adoption. Traditional functional safety approaches, which primarily
verify the reliability, robustness, and adequacy of AV hardware and software
systems from a vehicle-centric perspective, do not sufficiently address the
AV's broader interactions and behavioral impact on the surrounding traffic
environment. To overcome this limitation, we propose a paradigm shift toward
behavioral safety, a comprehensive approach focused on evaluating AV responses
and interactions within the traffic environment. To systematically assess
behavioral safety, we introduce a third-party AV safety assessment framework
comprising two complementary evaluation components: the Driver Licensing Test
and the Driving Intelligence Test. The Driver Licensing Test evaluates the AV's
reactive behaviors under controlled scenarios, ensuring basic behavioral
competency. In contrast, the Driving Intelligence Test assesses the AV's
interactive behaviors within naturalistic traffic conditions, quantifying the
frequency of safety-critical events to deliver statistically meaningful safety
metrics before large-scale deployment. We validated our proposed framework
using Autoware.Universe, an open-source Level 4 AV, tested both in simulated
environments and on the physical test track at the University of Michigan's
Mcity Testing Facility. The results indicate that Autoware.Universe passed 6
out of 14 scenarios and exhibited a crash rate of 3.01e-3 crashes per mile,
approximately 1,000 times higher than the average human driver crash rate.
During the tests, we also uncovered several unknown unsafe scenarios for
Autoware.Universe. These findings underscore the necessity of behavioral safety
evaluations for improving AV safety performance prior to widespread public
deployment.

### 4. [TacCompress: A Benchmark for Multi-Point Tactile Data Compression in Dexterous Manipulation](http://arxiv.org/pdf/2505.16289v1)

Authors: Yang Li, Yan Zhao, Zhengxue Cheng, Hengdi Zhang

Though robotic dexterous manipulation has progressed substantially recently,
challenges like in-hand occlusion still necessitate fine-grained tactile
perception, leading to the integration of more tactile sensors into robotic
hands. Consequently, the increased data volume imposes substantial bandwidth
pressure on signal transmission from the hand's controller. However, the
acquisition and compression of multi-point tactile signals based on the
dexterous hands' physical structures have not been thoroughly explored. In this
paper, our contributions are twofold. First, we introduce a Multi-Point Tactile
Dataset for Dexterous Hand Grasping (Dex-MPTD). This dataset captures tactile
signals from multiple contact sensors across various objects and grasping
poses, offering a comprehensive benchmark for advancing dexterous robotic
manipulation research. Second, we investigate both lossless and lossy
compression on Dex-MPTD by converting tactile data into images and applying six
lossless and five lossy image codecs for efficient compression. Experimental
results demonstrate that tactile data can be losslessly compressed to as low as
0.0364 bits per sub-sample (bpss), achieving approximately 200$\times$
compression ratio compared to the raw tactile data. Efficient lossy compressors
like HM and VTM can achieve about 1000x data reductions while preserving
acceptable data fidelity. The exploration of lossy compression also reveals
that screen-content-targeted coding tools outperform general-purpose codecs in
compressing tactile data.

### 5. [Unified Multi-Rate Model Predictive Control for a Jet-Powered Humanoid Robot](http://arxiv.org/pdf/2505.16478v1)

Authors: Davide Gorbani, Giuseppe L'Erario, Hosameldin Awadalla Omer Mohamed, Daniele Pucci

We propose a novel Model Predictive Control (MPC) framework for a jet-powered
flying humanoid robot. The controller is based on a linearised centroidal
momentum model to represent the flight dynamics, augmented with a second-order
nonlinear model to explicitly account for the slow and nonlinear dynamics of
jet propulsion. A key contribution is the introduction of a multi-rate MPC
formulation that handles the different actuation rates of the robot's joints
and jet engines while embedding the jet dynamics directly into the predictive
model. We validated the framework using the jet-powered humanoid robot iRonCub,
performing simulations in Mujoco; the simulation results demonstrate the
robot's ability to recover from external disturbances and perform stable,
non-abrupt flight manoeuvres, validating the effectiveness of the proposed
approach.

### 6. [D-LIO: 6DoF Direct LiDAR-Inertial Odometry based on Simultaneous Truncated Distance Field Mapping](http://arxiv.org/pdf/2505.16726v1)

Authors: Lucia Coto-Elena, J. E. Maese, L. Merino, F. Caballero

This paper presents a new approach for 6DoF Direct LiDAR-Inertial Odometry
(D-LIO) based on the simultaneous mapping of truncated distance fields on CPU.
Such continuous representation (in the vicinity of the points) enables working
with raw 3D LiDAR data online, avoiding the need of LiDAR feature selection and
tracking, simplifying the odometry pipeline and easily generalizing to many
scenarios. The method is based on the proposed Fast Truncated Distance Field
(Fast-TDF) method as a convenient tool to represent the environment. Such
representation enables i) solving the LiDAR point-cloud registration as a
nonlinear optimization process without the need of selecting/tracking LiDAR
features in the input data, ii) simultaneously producing an accurate truncated
distance field map of the environment, and iii) updating such map at constant
time independently of its size. The approach is tested using open datasets,
aerial and ground. It is also benchmarked against other state-of-the-art
odometry approaches, demonstrating the same or better level of accuracy with
the added value of an online-generated TDF representation of the environment,
that can be used for other robotics tasks as planning or collision avoidance.
The source code is publicly available at
https://anonymous.4open.science/r/D-LIO

### 7. [FlashBack: Consistency Model-Accelerated Shared Autonomy](http://arxiv.org/pdf/2505.16892v1)

Authors: Luzhe Sun, Jingtian Ji, Xiangshan Tan, Matthew R. Walter

Shared autonomy is an enabling technology that provides users with control
authority over robots that would otherwise be difficult if not impossible to
directly control. Yet, standard methods make assumptions that limit their
adoption in practice-for example, prior knowledge of the user's goals or the
objective (i.e., reward) function that they wish to optimize, knowledge of the
user's policy, or query-level access to the user during training.
Diffusion-based approaches to shared autonomy do not make such assumptions and
instead only require access to demonstrations of desired behaviors, while
allowing the user to maintain control authority. However, these advantages have
come at the expense of high computational complexity, which has made real-time
shared autonomy all but impossible. To overcome this limitation, we propose
Consistency Shared Autonomy (CSA), a shared autonomy framework that employs a
consistency model-based formulation of diffusion. Key to CSA is that it employs
the distilled probability flow of ordinary differential equations (PF ODE) to
generate high-fidelity samples in a single step. This results in inference
speeds significantly than what is possible with previous diffusion-based
approaches to shared autonomy, enabling real-time assistance in complex domains
with only a single function evaluation. Further, by intervening on flawed
actions at intermediate states of the PF ODE, CSA enables varying levels of
assistance. We evaluate CSA on a variety of challenging simulated and
real-world robot control problems, demonstrating significant improvements over
state-of-the-art methods both in terms of task performance and computational
efficiency.

### 8. [UAV See, UGV Do: Aerial Imagery and Virtual Teach Enabling Zero-Shot Ground Vehicle Repeat](http://arxiv.org/pdf/2505.16912v1)

Authors: Desiree Fisker, Alexander Krawciw, Sven Lilge, Melissa Greeff, Timothy D. Barfoot

This paper presents Virtual Teach and Repeat (VirT&R): an extension of the
Teach and Repeat (T&R) framework that enables GPS-denied, zero-shot autonomous
ground vehicle navigation in untraversed environments. VirT&R leverages aerial
imagery captured for a target environment to train a Neural Radiance Field
(NeRF) model so that dense point clouds and photo-textured meshes can be
extracted. The NeRF mesh is used to create a high-fidelity simulation of the
environment for piloting an unmanned ground vehicle (UGV) to virtually define a
desired path. The mission can then be executed in the actual target environment
by using NeRF-derived point cloud submaps associated along the path and an
existing LiDAR Teach and Repeat (LT&R) framework. We benchmark the
repeatability of VirT&R on over 12 km of autonomous driving data using physical
markings that allow a sim-to-real lateral path-tracking error to be obtained
and compared with LT&R. VirT&R achieved measured root mean squared errors
(RMSE) of 19.5 cm and 18.4 cm in two different environments, which are slightly
less than one tire width (24 cm) on the robot used for testing, and respective
maximum errors were 39.4 cm and 47.6 cm. This was done using only the
NeRF-derived teach map, demonstrating that VirT&R has similar closed-loop
path-tracking performance to LT&R but does not require a human to manually
teach the path to the UGV in the actual environment.

### 9. [3D Equivariant Visuomotor Policy Learning via Spherical Projection](http://arxiv.org/pdf/2505.16969v1)

Authors: Boce Hu, Dian Wang, David Klee, Heng Tian, Xupeng Zhu, Haojie Huang, Robert Platt, Robin Walters

Equivariant models have recently been shown to improve the data efficiency of
diffusion policy by a significant margin. However, prior work that explored
this direction focused primarily on point cloud inputs generated by multiple
cameras fixed in the workspace. This type of point cloud input is not
compatible with the now-common setting where the primary input modality is an
eye-in-hand RGB camera like a GoPro. This paper closes this gap by
incorporating into the diffusion policy model a process that projects features
from the 2D RGB camera image onto a sphere. This enables us to reason about
symmetries in SO(3) without explicitly reconstructing a point cloud. We perform
extensive experiments in both simulation and the real world that demonstrate
that our method consistently outperforms strong baselines in terms of both
performance and sample efficiency. Our work is the first SO(3)-equivariant
policy learning framework for robotic manipulation that works using only
monocular RGB inputs.

### 10. [RE-TRIP : Reflectivity Instance Augmented Triangle Descriptor for 3D Place Recognition](http://arxiv.org/pdf/2505.16165v1)

Authors: Yechan Park, Gyuhyeon Pak, Euntai Kim

While most people associate LiDAR primarily with its ability to measure
distances and provide geometric information about the environment (via point
clouds), LiDAR also captures additional data, including reflectivity or
intensity values. Unfortunately, when LiDAR is applied to Place Recognition
(PR) in mobile robotics, most previous works on LiDAR-based PR rely only on
geometric measurements, neglecting the additional reflectivity information that
LiDAR provides. In this paper, we propose a novel descriptor for 3D PR, named
RE-TRIP (REflectivity-instance augmented TRIangle descriPtor). This new
descriptor leverages both geometric measurements and reflectivity to enhance
robustness in challenging scenarios such as geometric degeneracy, high
geometric similarity, and the presence of dynamic objects. To implement RE-TRIP
in real-world applications, we further propose (1) a keypoint extraction
method, (2) a key instance segmentation method, (3) a RE-TRIP matching method,
and (4) a reflectivity-combined loop verification method. Finally, we conduct a
series of experiments to demonstrate the effectiveness of RE-TRIP. Applied to
public datasets (i.e., HELIPR, FusionPortable) containing diverse scenarios
such as long corridors, bridges, large-scale urban areas, and highly dynamic
environments -- our experimental results show that the proposed method
outperforms existing state-of-the-art methods in terms of Scan Context,
Intensity Scan Context, and STD.

### Software Engineering

### 1. [Rethinking Code Review Workflows with LLM Assistance: An Empirical Study](http://arxiv.org/pdf/2505.16339v1)

Authors: Fannar Steinn Aðalsteinsson, Björn Borgar Magnússon, Mislav Milicevic, Adam Nirving Davidsson, Chih-Hong Cheng

Code reviews are a critical yet time-consuming aspect of modern software
development, increasingly challenged by growing system complexity and the
demand for faster delivery. This paper presents a study conducted at
WirelessCar Sweden AB, combining an exploratory field study of current code
review practices with a field experiment involving two variations of an
LLM-assisted code review tool. The field study identifies key challenges in
traditional code reviews, including frequent context switching, insufficient
contextual information, and highlights both opportunities (e.g., automatic
summarization of complex pull requests) and concerns (e.g., false positives and
trust issues) in using LLMs. In the field experiment, we developed two
prototype variations: one offering LLM-generated reviews upfront and the other
enabling on-demand interaction. Both utilize a semantic search pipeline based
on retrieval-augmented generation to assemble relevant contextual information
for the review, thereby tackling the uncovered challenges. Developers evaluated
both variations in real-world settings: AI-led reviews are overall more
preferred, while still being conditional on the reviewers' familiarity with the
code base, as well as on the severity of the pull request.

### 2. [Web Element Relocalization in Evolving Web Applications: A Comparative Analysis and Extension Study](http://arxiv.org/pdf/2505.16424v1)

Authors: Anton Kluge, Andrea Stocco

Fragile web tests, primarily caused by locator breakages, are a persistent
challenge in web development. Hence, researchers have proposed techniques for
web-element re-identification in which algorithms utilize a range of element
properties to relocate elements on updated versions of websites based on
similarity scoring. In this paper, we replicate the original studies of the
most recent propositions in the literature, namely the Similo algorithm and its
successor, VON Similo. We also acknowledge and reconsider assumptions related
to threats to validity in the original studies, which prompted additional
analysis and the development of mitigation techniques. Our analysis revealed
that VON Similo, despite its novel approach, tends to produce more false
positives than Similo. We mitigated these issues through algorithmic
refinements and optimization algorithms that enhance parameters and comparison
methods across all Similo variants, improving the accuracy of Similo on its
original benchmark by 5.62%. Moreover, we extend the replicated studies by
proposing a larger evaluation benchmark (23x bigger than the original study) as
well as a novel approach that combines the strengths of both Similo and VON
Similo, called HybridSimilo. The combined approach achieved a gain comparable
to the improved Similo alone. Results on the extended benchmark show that
HybridSimilo locates 98.8% of elements with broken locators in realistic
testing scenarios.

### 3. [A Survey on the Application of Large Language Models in Scenario-Based Testing of Automated Driving Systems](http://arxiv.org/pdf/2505.16587v1)

Authors: Yongqi Zhao, Ji Zhou, Dong Bi, Tomislav Mihalj, Jia Hu, Arno Eichberger

The safety and reliability of Automated Driving Systems (ADSs) must be
validated prior to large-scale deployment. Among existing validation
approaches, scenario-based testing has been regarded as a promising method to
improve testing efficiency and reduce associated costs. Recently, the emergence
of Large Language Models (LLMs) has introduced new opportunities to reinforce
this approach. While an increasing number of studies have explored the use of
LLMs in the field of automated driving, a dedicated review focusing on their
application within scenario-based testing remains absent. This survey addresses
this gap by systematically categorizing the roles played by LLMs across various
phased of scenario-based testing, drawing from both academic research and
industrial practice. In addition, key characteristics of LLMs and corresponding
usage strategies are comprehensively summarized. The paper concludes by
outlining five open challenges and potential research directions. To support
ongoing research efforts, a continuously updated repository of recent
advancements and relevant open-source tools is made available at:
https://github.com/ftgTUGraz/LLM4ADSTest.

### 4. [Beyond LLMs: An Exploration of Small Open-source Language Models in Logging Statement Generation](http://arxiv.org/pdf/2505.16590v1)

Authors: Renyi Zhong, Yichen Li, Guangba Yu, Wenwei Gu, Jinxi Kuang, Yintong Huo, Michael R. Lyu

Effective software maintenance heavily relies on high-quality logging
statements, but manual logging is challenging, error-prone, and insufficiently
standardized, often leading to inconsistent log quality. While large language
models have shown promise in automatic logging, they introduce concerns
regarding privacy, resource intensity, and adaptability to specific enterprise
needs. To tackle these limitations, this paper empirically investigates whether
Small Open-source Language Models (SOLMs) could become a viable alternative via
proper exploitation. Specifically, we conduct a large-scale empirical study on
four prominent SOLMs, systematically evaluating the impacts of various
interaction strategies, parameter-efficient fine-tuning techniques, model
sizes, and model types in automatic logging. Our key findings reveal that
Retrieval-Augmented Generation significantly enhances performance, and LoRA is
a highly effective PEFT technique. While larger SOLMs tend to perform better,
this involves a trade-off with computational resources, and instruct-tuned
SOLMs generally surpass their base counterparts. Notably, fine-tuned SOLMs,
particularly Qwen2.5-coder-14B, outperformed existing specialized tools and LLM
baselines in accurately predicting logging locations and generating
high-quality statements, a conclusion supported by traditional evaluation
metrics and LLM-as-a-judge evaluations. Furthermore, SOLMs also demonstrated
robust generalization across diverse, unseen code repositories.

### 5. [Software Architecture Meets LLMs: A Systematic Literature Review](http://arxiv.org/pdf/2505.16697v1)

Authors: Larissa Schmid, Tobias Hey, Martin Armbruster, Sophie Corallo, Dominik Fuchß, Jan Keim, Haoyu Liu, Anne Koziolek

Large Language Models (LLMs) are used for many different software engineering
tasks. In software architecture, they have been applied to tasks such as
classification of design decisions, detection of design patterns, and
generation of software architecture design from requirements. However, there is
little overview on how well they work, what challenges exist, and what open
problems remain. In this paper, we present a systematic literature review on
the use of LLMs in software architecture. We analyze 18 research articles to
answer five research questions, such as which software architecture tasks LLMs
are used for, how much automation they provide, which models and techniques are
used, and how these approaches are evaluated. Our findings show that while LLMs
are increasingly applied to a variety of software architecture tasks and often
outperform baselines, some areas, such as generating source code from
architectural design, cloud-native computing and architecture, and checking
conformance remain underexplored. Although current approaches mostly use simple
prompting techniques, we identify a growing research interest in refining
LLM-based approaches by integrating advanced techniques.

### 6. [Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation](http://arxiv.org/pdf/2505.16222v1)

Authors: Jiwon Moon, Yerin Hwang, Dongryeol Lee, Taegwan Kang, Yongil Kim, Kyomin Jung

With the growing use of large language models(LLMs) as evaluators, their
application has expanded to code evaluation tasks, where they assess the
correctness of generated code without relying on reference implementations.
While this offers scalability and flexibility, it also raises a critical,
unresolved question: Can LLM judges fairly and robustly evaluate semantically
equivalent code with superficial variations? Functionally correct code often
exhibits variations-such as differences in variable names, comments, or
formatting-that should not influence its correctness. Yet, whether LLM judges
can reliably handle these variations remains unclear. We present the first
comprehensive study of this issue, defining six types of potential bias in code
evaluation and revealing their systematic impact on LLM judges. Across five
programming languages and multiple LLMs, we empirically demonstrate that all
tested LLM judges are susceptible to both positive and negative biases,
resulting in inflated or unfairly low scores. Moreover, we observe that LLM
judges remain vulnerable to these biases even when prompted to generate test
cases before scoring, highlighting the need for more robust code evaluation
methods.

### 7. [Multimodal Generative AI for Story Point Estimation in Software Development](http://arxiv.org/pdf/2505.16290v1)

Authors: Mohammad Rubyet Islam, Peter Sandborn

This research explores the application of Multimodal Generative AI to enhance
story point estimation in Agile software development. By integrating text,
image, and categorical data using advanced models like BERT, CNN, and XGBoost,
our approach surpasses the limitations of traditional single-modal estimation
methods. The results demonstrate strong accuracy for simpler story points,
while also highlighting challenges in more complex categories due to data
imbalance. This study further explores the impact of categorical data,
particularly severity, on the estimation process, emphasizing its influence on
model performance. Our findings emphasize the transformative potential of
multimodal data integration in refining AI-driven project management, paving
the way for more precise, adaptable, and domain-specific AI capabilities.
Additionally, this work outlines future directions for addressing data
variability and enhancing the robustness of AI in Agile methodologies.

### 8. [Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks](http://arxiv.org/pdf/2505.16901v1)

Authors: Hongyuan Tao, Ying Zhang, Zhenhao Tang, Hongen Peng, Xukun Zhu, Bingchang Liu, Yingguang Yang, Ziyin Zhang, Zhaogui Xu, Haipeng Zhang, Linchao Zhu, Rui Wang, Hang Yu, Jianguo Li, Peng Di

Recent advances in Large Language Models (LLMs) have shown promise in
function-level code generation, yet repository-level software engineering tasks
remain challenging. Current solutions predominantly rely on proprietary LLM
agents, which introduce unpredictability and limit accessibility, raising
concerns about data privacy and model customization. This paper investigates
whether open-source LLMs can effectively address repository-level tasks without
requiring agent-based approaches. We demonstrate this is possible by enabling
LLMs to comprehend functions and files within codebases through their semantic
information and structural dependencies. To this end, we introduce Code Graph
Models (CGMs), which integrate repository code graph structures into the LLM's
attention mechanism and map node attributes to the LLM's input space using a
specialized adapter. When combined with an agentless graph RAG framework, our
approach achieves a 43.00% resolution rate on the SWE-bench Lite benchmark
using the open-source Qwen2.5-72B model. This performance ranks first among
open weight models, second among methods with open-source systems, and eighth
overall, surpassing the previous best open-source model-based method by 12.33%.

### 9. [SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development](http://arxiv.org/pdf/2505.16975v1)

Authors: Yaxin Du, Yuzhu Cai, Yifan Zhou, Cheng Wang, Yu Qian, Xianghe Pang, Qian Liu, Yue Hu, Siheng Chen

Large Language Models (LLMs) have shown strong capability in diverse software
engineering tasks, e.g. code completion, bug fixing, and document generation.
However, feature-driven development (FDD), a highly prevalent real-world task
that involves developing new functionalities for large, existing codebases,
remains underexplored. We therefore introduce SWE-Dev, the first large-scale
dataset (with 14,000 training and 500 test samples) designed to evaluate and
train autonomous coding systems on real-world feature development tasks. To
ensure verifiable and diverse training, SWE-Dev uniquely provides all instances
with a runnable environment and its developer-authored executable unit tests.
This collection not only provides high-quality data for Supervised Fine-Tuning
(SFT), but also enables Reinforcement Learning (RL) by delivering accurate
reward signals from executable unit tests. Our extensive evaluations on
SWE-Dev, covering 17 chatbot LLMs, 10 reasoning models, and 10 Multi-Agent
Systems (MAS), reveal that FDD is a profoundly challenging frontier for current
AI (e.g., Claude-3.7-Sonnet achieves only 22.45\% Pass@3 on the hard test
split). Crucially, we demonstrate that SWE-Dev serves as an effective platform
for model improvement: fine-tuning on training set enabled a 7B model
comparable to GPT-4o on \textit{hard} split, underscoring the value of its
high-quality training data. Code is available here
\href{https://github.com/justLittleWhite/SWE-Dev}{https://github.com/justLittleWhite/SWE-Dev}.

### 10. [AutoMCQ -- Automatically Generate Code Comprehension Questions using GenAI](http://arxiv.org/pdf/2505.16430v1)

Authors: Martin Goodfellow, Robbie Booth, Andrew Fagan, Alasdair Lambert

Students often do not fully understand the code they have written. This
sometimes does not become evident until later in their education, which can
mean it is harder to fix their incorrect knowledge or misunderstandings. In
addition, being able to fully understand code is increasingly important in a
world where students have access to generative artificial intelligence (GenAI)
tools, such as GitHub Copilot. One effective solution is to utilise code
comprehension questions, where a marker asks questions about a submission to
gauge understanding, this can also have the side effect of helping to detect
plagiarism. However, this approach is time consuming and can be difficult
and/or expensive to scale. This paper introduces AutoMCQ, which uses GenAI for
the automatic generation of multiple-choice code comprehension questions. This
is integrated with the CodeRunner automated assessment platform.

### Social and Information Networks

### 1. [Novel Rewiring Mechanism for Restoration of the Fragmented Social Networks after Attacks](http://arxiv.org/pdf/2505.16233v1)

Authors: Rajesh Kumar, Suchi Kumari, Anubhav Mishra

Real-world complex systems exhibit intricate interconnections and
dependencies, especially social networks, technological infrastructures, and
communication networks. These networks are prone to disconnection due to random
failures or external attacks on their components. Therefore, managing the
security and resilience of such networks is a prime concern, particularly at
the time of disaster. Therefore, in this research work, network is
reconstructed by rewiring/addition of the edges and robustness of the networks
is measured. To this aim, two approaches namely (i) Strategic rewiring (ii)
budget constrained optimal rewiring are adopted. While current research often
assesses robustness by examining the size of the largest connected component,
this approach fails to capture the complete spectrum of vulnerability. The
failure of a small number of connections leads to a sparser network yet
connected network. Thus, the present research work delves deeper into
evaluating the robustness of the restored network by evaluating Laplacian
Energy to better comprehend the system's behavior during the restoration of the
network still considering the size of the largest connected component attacks.

### 2. [Filling in the Blanks? A Systematic Review and Theoretical Conceptualisation for Measuring WikiData Content Gaps](http://arxiv.org/pdf/2505.16383v1)

Authors: Marisa Ripoll, Neal Reeves, Anelia Kurteva, Elena Simperl, Albert Meroño Peñuela

Wikidata is a collaborative knowledge graph which provides machine-readable
structured data for Wikimedia projects including Wikipedia. Managed by a
community of volunteers, it has grown to become the most edited Wikimedia
project. However, it features a long-tail of items with limited data and a
number of systematic gaps within the available content. In this paper, we
present the results of a systematic literature review aimed to understand the
state of these content gaps within Wikidata. We propose a typology of gaps
based on prior research and contribute a theoretical framework intended to
conceptualise gaps and support their measurement. We also describe the methods
and metrics present used within the literature and classify them according to
our framework to identify overlooked gaps that might occur in Wikidata. We then
discuss the implications for collaboration and editor activity within Wikidata
as well as future research directions. Our results contribute to the
understanding of quality, completeness and the impact of systematic biases
within Wikidata and knowledge gaps more generally.

### 3. [Urban transport systems shape experiences of social segregation](http://arxiv.org/pdf/2505.16337v1)

Authors: Yitao Yang, Erjian Liu, Bin Jia, Ed Manley

Mobility is a fundamental feature of human life, and through it our
interactions with the world and people around us generate complex and
consequential social phenomena. Social segregation, one such process, is
increasingly acknowledged as a product of one's entire lived experience rather
than mere residential location. Increasingly granular sources of data on human
mobility have evidenced how segregation persists outside the home, in
workplaces, cafes, and on the street. Yet there remains only a weak evidential
link between the production of social segregation and urban policy. This study
addresses this gap through an assessment of the role of the urban
transportation systems in shaping social segregation. Using city-scale GPS
mobility data and a novel probabilistic mobility framework, we establish social
interactions at the scale of transportation infrastructure, by rail and bus
service segment, individual roads, and city blocks. The outcomes show how
social segregation is more than a single process in space, but varying by time
of day, urban design and structure, and service design. These findings
reconceptualize segregation as a product of likely encounters during one's
daily mobility practice. We then extend these findings through exploratory
simulations, highlighting how transportation policy to promote sustainable
transport may have potentially unforeseen impacts on segregation. The study
underscores that to understand social segregation and achieve positive social
change urban policymakers must consider the broadest impacts of their
interventions and seek to understand their impact on the daily lived experience
of their citizens.

### 4. [Modeling Inequality in Complex Networks of Strategic Agents using Iterative Game-Theoretic Transactions](http://arxiv.org/pdf/2505.16966v1)

Authors: Mayank Kejriwal, Yuesheng Luo

Transactions are an important aspect of human social life, and represent
dynamic flow of information, intangible values, such as trust, as well as
monetary and social capital. Although much research has been conducted on the
nature of transactions in fields ranging from the social sciences to game
theory, the systemic effects of different types of agents transacting in
real-world social networks (often following a scale-free distribution) are not
fully understood. A particular systemic measure that has not received adequate
attention in the complex networks and game theory communities, is the Gini
Coefficient, which is widely used in economics to quantify and understand
wealth inequality. In part, the problem is a lack of experimentation using a
replicable algorithm and publicly available data. Motivated by this problem,
this article proposes a model and simulation algorithm, based on game theory,
for quantifying the evolution of inequality in complex networks of strategic
agents. Our results shed light on several complex drivers of inequality, even
in simple, abstract settings, and exhibit consistency across networks with
different origins and descriptions.

### 5. [Scalable Graph Generative Modeling via Substructure Sequences](http://arxiv.org/pdf/2505.16130v1)

Authors: Zehong Wang, Zheyuan Zhang, Tianyi Ma, Chuxu Zhang, Yanfang Ye

Graph neural networks (GNNs) has been predominantly driven by
message-passing, where node representations are iteratively updated via local
neighborhood aggregation. Despite their success, message-passing suffers from
fundamental limitations -- including constrained expressiveness,
over-smoothing, over-squashing, and limited capacity to model long-range
dependencies. These issues hinder scalability: increasing data size or model
size often fails to yield improved performance, limiting the viability of GNNs
as backbones for graph foundation models. In this work, we explore pathways
beyond message-passing and introduce Generative Graph Pattern Machine
(G$^2$PM), a generative Transformer pre-training framework for graphs. G$^2$PM
represents graph instances (nodes, edges, or entire graphs) as sequences of
substructures, and employs generative pre-training over the sequences to learn
generalizable, transferable representations. Empirically, G$^2$PM demonstrates
strong scalability: on the ogbn-arxiv benchmark, it continues to improve with
model sizes up to 60M parameters, outperforming prior generative approaches
that plateau at significantly smaller scales (e.g., 3M). In addition, we
systematically analyze the model design space, highlighting key architectural
choices that contribute to its scalability and generalization. Across diverse
tasks -- including node classification, graph classification, and transfer
learning -- G$^2$PM consistently outperforms strong baselines, establishing a
compelling foundation for scalable graph learning. The code and dataset are
available at https://github.com/Zehong-Wang/G2PM.

### Systems and Control

### 1. [Partitioning and Observability in Linear Systems via Submodular Optimization](http://arxiv.org/pdf/2505.16169v1)

Authors: Mohamad H. Kazma, Ahmad F. Taha

Network partitioning has gained recent attention as a pathway to enable
decentralized operation and control in large-scale systems. This paper
addresses the interplay between partitioning, observability, and sensor
placement (SP) in dynamic networks. The problem, being computationally
intractable at scale, is largely unexplored in the literature. To that end, the
paper's objective is designing scalable partitioning of linear systems while
maximizing observability metrics of the subsystems. We show that the
partitioning problem can be posed as a submodular maximization problem -- and
the SP problem can subsequently be solved over the partitioned network.
Consequently, theoretical bounds are derived to compare observability metrics
of the original network with those of the resulting partitions, highlighting
the impact of partitioning on system observability. Case studies on networks of
varying sizes corroborate the derived theoretical bounds.

### 2. [Scalable and Efficient Aggregation of Energy-Constrained Flexible Loads](http://arxiv.org/pdf/2505.16374v1)

Authors: Julie Rousseau, Philipp Heer, Kristina Orehounig, Gabriela Hug

Loads represent a promising flexibility source to support the integration of
renewable energy sources, as they may shift their energy consumption over time.
By computing the aggregated flexibility of power and energy-constrained loads,
aggregators can communicate the group's flexibility without sharing individual
private information. However, this computation is, in practice, challenging.
Some studies suggest different inner approximations of aggregated flexibility
polytopes, but all suffer from large computational costs for realistic load
numbers and horizon lengths. In this paper, we develop a novel approximation of
the aggregated flexibility of loads based on the concept of worst-case energy
dispatch, i.e., if aggregated energy consumptions are assumed to be dispatched
in the worst manner possible. This leads to conservative piecewise linear
bounds that restrict the aggregated energy consumption only based on the
previous aggregated energy consumed. A comparative case study reveals that our
method can compute an approximation of the aggregation of thousands of loads
efficiently, while displaying an accuracy comparable to other approximation
techniques.

### 3. [Coverage Path Planning For Multi-view SAR-UAV Observation System Under Energy Constraint](http://arxiv.org/pdf/2505.16389v1)

Authors: Deyu Song, Xiangyin Zhang, Zipei Yu, Kaiyu Qin

Multi-view Synthetic Aperture Radar (SAR) imaging can effectively enhance the
performance of tasks such as automatic target recognition and image information
fusion. Unmanned aerial vehicles (UAVs) have the advantages of flexible
deployment and cost reduction. A swarm of UAVs equipped with synthetic aperture
radar imaging equipment is well suited to meet the functional requirements of
multi-view synthetic aperture radar imaging missions. However, to provide
optimal paths for SAR-UAVs from the base station to cover target viewpoints in
the mission area is of NP-hard computational complexity. In this work, the
coverage path planning problem for multi-view SAR-UAV observation systems is
studied. First, the coordinate of observation viewpoints is calculated based on
the location of targets and base station under a brief geometric model. Then,
the exact problem formulation is modeled in order to fully describe the
solution space and search for optimal paths that provide maximum coverage rate
for SAR-UAVs. Finally, an Adaptive Density Peak Clustering (ADPC) method is
proposed to overcome the additional energy consumption due to the viewpoints
being far away from the base station. The Particle Swarm Optimization (PSO)
algorithm is introduced for optimal path generation. Experimental results
demonstrate the effectiveness and computational efficiency of the proposed
approach.

### 4. [Trajectory-Independent Flexibility Envelopes of Energy-Constrained Systems with State-Dependent Losses](http://arxiv.org/pdf/2505.16396v1)

Authors: Julie Rousseau, Carlo Tajoli, Hanmin Cai, Philipp Heer, Kristina Orehounig, Gabriela Hug

As non-dispatchable renewable power units become prominent in electric power
grids, demand-side flexibility appears as a key element of future power
systems' operation. Power and energy bounds are intuitive metrics to describe
the flexibility of energy-constrained loads. However, to be used in operation,
any power consumption trajectory fulfilling the power and energy bounds must
necessarily fulfill the load's constraints. In this paper, we demonstrate that
energy bounds defined as the minimum and maximum energy consumption potential
of a load with state-dependent losses are Trajectory-Dependent (TD), i.e., for
any energy value in the bounds a feasible power trajectory exists, but not all
power trajectories enclosed in the energy envelopes satisfy the load's
constraints. To guarantee the satisfaction of load constraints for all
trajectories, we define Trajectory-Independent (TI) energy bounds. We present
TI envelope formulations for individual loads, as well as physically coupled
loads and assess the proposed formulations in a building heating system, a
system with state-dependent losses. We find that using a TD envelope as energy
bounds in operation may yield room temperature up to 3.8{\deg}C higher and
3.4{\deg}C lower than admissible. Overall, poorly insulated buildings observe a
TI energy envelope that differs significantly from their TD envelope.

### 5. [Robust Look-ahead Pursuit Control for Three-Dimensional Path Following within Finite-Time Stability Guarantee](http://arxiv.org/pdf/2505.16407v1)

Authors: Zimao Sheng, Hong an Yang, ZiRui Yu, Jiakang Wang

This paper addresses the challenging problem of robust path following for
fixed-wing unmanned aerial vehicles (UAVs) in complex environments with bounded
external disturbances and non-smooth predefined paths. Due to the unique
aerodynamic characteristics and flight constraints of fixed-wing UAVs,
achieving accurate and stable path following remains difficult, especially in
low-altitude mountainous terrains, urban landscapes, and under wind
disturbances. Traditional path-following guidance laws often struggle with
rapid stabilization and constrained input commands under unknown disturbances
while maintaining robustness. To overcome these limitations, we propose a
robust nonlinear path-following guidance law that considers the flight path
angle and track angle, and dynamically adjusts controller parameters to achieve
optimal compensation for acceleration increments. The proposed guidance law
guarantees finite-time stability, reduced sensitivity to constrained
uncertainties, and consistent behavior compared to traditional asymptotic
convergence controllers. Additionally, it ensures that the UAV approaches
mobile virtual target points in the shortest possible time while adhering to
input constrained conditions. Our contributions include a thorough analysis of
the conditions for robust stability, the derivation of the guidance law, and
simulations demonstrating its effectiveness. The results show that the proposed
guidance law significantly improves path-following performance under external
disturbances, making it a promising solution for autonomous missions execution
of fixed-wing UAVs.

### 6. [Data Center Model for Transient Stability Analysis of Power Systems](http://arxiv.org/pdf/2505.16575v1)

Authors: Alberto Jimenez-Ruiz, Federico Milano

The rising demand of computing power leads to the installation of a large
number of Data Centers (DCs). Their Fault-Ride-Through (FRT) behavior and their
unique power characteristics, especially for DCs catered to Artificial
Intelligence (AI) workloads, pose a threat to the stability of power systems.
To ensure its stability, it is required accurate models of the loads involved.
Here we propose a dynamic load model that properly captures the behaviour of
DCs. Its three most defining features are the use of an Uninterrupted Power
Supply (UPS) which sits between the server load and the grid, the cooling load
represented by an induction motor, and a pulsing load that represents the
transients caused by contemporary DCs with significant AI workloads. The
features of the proposed model and its impact on the dynamic performance of
transmission systems are illustrated through a model of the all-island Irish
transmission system and real-world data of the DCs currently connected to this
system.

### 7. [On the Deployment of RIS-mounted UAV Networks](http://arxiv.org/pdf/2505.16841v1)

Authors: Anupam Mondal, Priyadarshi Mukherjee, Sasthi C. Ghosh

Reconfigurable intelligent surfaces (RIS) enable smart wireless environments
by dynamically controlling signal propagation to enhance communication and
localization. Unmanned aerial vehicles (UAVs) can act as flying base stations
and thus, improve system performance by avoiding signal blockages. In this
paper, we propose a gradient ascent and coordinate search based method to
determine the optimal location for a system that consists of a UAV and a RIS,
where the UAV serves cellular users (CUs) and the RIS serves device-to-device
(D2D) pairs. In particular, by optimizing the net throughput for both the D2D
pairs and the CUs, the suggested method establishes the ideal location for the
RIS-mounted UAV. We consider both line of sight (LoS) and non-LoS paths for the
RIS and UAV to calculate the throughput while accounting for blockages in the
system. The numerical results show that the proposed method performs better
than the existing approaches in terms of both the net throughput and the user
fairness.

### 8. [Modeling and Constraint-Aware Control of Pressure Dynamics in Water Electrolysis Systems](http://arxiv.org/pdf/2505.16935v1)

Authors: Mostafaali Ayubirad, Madiha Akbar, Hamid R. Ossareh

This paper addresses the challenge of pressure constraint violations in water
electrolysis systems operating under dynamic power conditions, a problem common
to both Proton Exchange Membrane and alkaline technologies. To investigate this
issue, a control-oriented model of an alkaline electrolyzer is developed,
capturing key pressure and flow dynamics. To manage rapid power fluctuations
that may cause pressure to exceed manufacturer-defined operational boundaries,
a model-based constraint-aware power governor based on the Reference Governor
(RG) framework is proposed. Simulation results show that the strategy
effectively maintains pressure within the specified operating range,
outperforming conventional filtering methods while enhancing hydrogen
production and reducing auxiliary energy consumption.

### 9. [Offline Guarded Safe Reinforcement Learning for Medical Treatment Optimization Strategies](http://arxiv.org/pdf/2505.16242v1)

Authors: Runze Yan, Xun Shen, Akifumi Wachi, Sebastien Gros, Anni Zhao, Xiao Hu

When applying offline reinforcement learning (RL) in healthcare scenarios,
the out-of-distribution (OOD) issues pose significant risks, as inappropriate
generalization beyond clinical expertise can result in potentially harmful
recommendations. While existing methods like conservative Q-learning (CQL)
attempt to address the OOD issue, their effectiveness is limited by only
constraining action selection by suppressing uncertain actions. This
action-only regularization imitates clinician actions that prioritize
short-term rewards, but it fails to regulate downstream state trajectories,
thereby limiting the discovery of improved long-term treatment strategies. To
safely improve policy beyond clinician recommendations while ensuring that
state-action trajectories remain in-distribution, we propose \textit{Offline
Guarded Safe Reinforcement Learning} ($\mathsf{OGSRL}$), a theoretically
grounded model-based offline RL framework. $\mathsf{OGSRL}$ introduces a novel
dual constraint mechanism for improving policy with reliability and safety.
First, the OOD guardian is established to specify clinically validated regions
for safe policy exploration. By constraining optimization within these regions,
it enables the reliable exploration of treatment strategies that outperform
clinician behavior by leveraging the full patient state history, without
drifting into unsupported state-action trajectories. Second, we introduce a
safety cost constraint that encodes medical knowledge about physiological
safety boundaries, providing domain-specific safeguards even in areas where
training data might contain potentially unsafe interventions. Notably, we
provide theoretical guarantees on safety and near-optimality: policies that
satisfy these constraints remain in safe and reliable regions and achieve
performance close to the best possible policy supported by the data.

### 10. [Performance Guaranteed Poisoning Attacks in Federated Learning: A Sliding Mode Approach](http://arxiv.org/pdf/2505.16403v1)

Authors: Huazi Pan, Yanjun Zhang, Leo Yu Zhang, Scott Adams, Abbas Kouzani, Suiyang Khoo

Manipulation of local training data and local updates, i.e., the poisoning
attack, is the main threat arising from the collaborative nature of the
federated learning (FL) paradigm. Most existing poisoning attacks aim to
manipulate local data/models in a way that causes denial-of-service (DoS)
issues. In this paper, we introduce a novel attack method, named Federated
Learning Sliding Attack (FedSA) scheme, aiming at precisely introducing the
extent of poisoning in a subtle controlled manner. It operates with a
predefined objective, such as reducing global model's prediction accuracy by
10\%. FedSA integrates robust nonlinear control-Sliding Mode Control (SMC)
theory with model poisoning attacks. It can manipulate the updates from
malicious clients to drive the global model towards a compromised state,
achieving this at a controlled and inconspicuous rate. Additionally, leveraging
the robust control properties of FedSA allows precise control over the
convergence bounds, enabling the attacker to set the global accuracy of the
poisoned model to any desired level. Experimental results demonstrate that
FedSA can accurately achieve a predefined global accuracy with fewer malicious
clients while maintaining a high level of stealth and adjustable learning
rates.

### Machine Learning (Statistics Category)

### 1. [Exponential Convergence of CAVI for Bayesian PCA](http://arxiv.org/pdf/2505.16145v1)

Authors: Arghya Datta, Philippe Gagnon, Florian Maire

Probabilistic principal component analysis (PCA) and its Bayesian variant
(BPCA) are widely used for dimension reduction in machine learning and
statistics. The main advantage of probabilistic PCA over the traditional
formulation is allowing uncertainty quantification. The parameters of BPCA are
typically learned using mean-field variational inference, and in particular,
the coordinate ascent variational inference (CAVI) algorithm. So far, the
convergence speed of CAVI for BPCA has not been characterized. In our paper, we
fill this gap in the literature. Firstly, we prove a precise exponential
convergence result in the case where the model uses a single principal
component (PC). Interestingly, this result is established through a connection
with the classical $\textit{power iteration algorithm}$ and it indicates that
traditional PCA is retrieved as points estimates of the BPCA parameters.
Secondly, we leverage recent tools to prove exponential convergence of CAVI for
the model with any number of PCs, thus leading to a more general result, but
one that is of a slightly different flavor. To prove the latter result, we
additionally needed to introduce a novel lower bound for the symmetric
Kullback--Leibler divergence between two multivariate normal distributions,
which, we believe, is of independent interest in information theory.

### 2. [Integral Imprecise Probability Metrics](http://arxiv.org/pdf/2505.16156v1)

Authors: Siu Lun Chau, Michele Caprio, Krikamol Muandet

Quantifying differences between probability distributions is fundamental to
statistics and machine learning, primarily for comparing statistical
uncertainty. In contrast, epistemic uncertainty (EU) -- due to incomplete
knowledge -- requires richer representations than those offered by classical
probability. Imprecise probability (IP) theory offers such models, capturing
ambiguity and partial belief. This has driven growing interest in imprecise
probabilistic machine learning (IPML), where inference and decision-making rely
on broader uncertainty models -- highlighting the need for metrics beyond
classical probability. This work introduces the Integral Imprecise Probability
Metric (IIPM) framework, a Choquet integral-based generalisation of classical
Integral Probability Metric (IPM) to the setting of capacities -- a broad class
of IP models encompassing many existing ones, including lower probabilities,
probability intervals, belief functions, and more. Theoretically, we establish
conditions under which IIPM serves as a valid metric and metrises a form of
weak convergence of capacities. Practically, IIPM not only enables comparison
across different IP models but also supports the quantification of epistemic
uncertainty within a single IP model. In particular, by comparing an IP model
with its conjugate, IIPM gives rise to a new class of EU measures -- Maximum
Mean Imprecision -- which satisfy key axiomatic properties proposed in the
Uncertainty Quantification literature. We validate MMI through selective
classification experiments, demonstrating strong empirical performance against
established EU measures, and outperforming them when classical methods struggle
to scale to a large number of classes. Our work advances both theory and
practice in IPML, offering a principled framework for comparing and quantifying
epistemic uncertainty under imprecision.

### 3. [Graph-Smoothed Bayesian Black-Box Shift Estimator and Its Information Geometry](http://arxiv.org/pdf/2505.16251v1)

Authors: Masanari Kimura

Label shift adaptation aims to recover target class priors when the labelled
source distribution $P$ and the unlabelled target distribution $Q$ share $P(X
\mid Y) = Q(X \mid Y)$ but $P(Y) \neq Q(Y)$. Classical black-box shift
estimators invert an empirical confusion matrix of a frozen classifier,
producing a brittle point estimate that ignores sampling noise and similarity
among classes. We present Graph-Smoothed Bayesian BBSE (GS-B$^3$SE), a fully
probabilistic alternative that places Laplacian-Gaussian priors on both target
log-priors and confusion-matrix columns, tying them together on a
label-similarity graph. The resulting posterior is tractable with HMC or a fast
block Newton-CG scheme. We prove identifiability, $N^{-1/2}$ contraction,
variance bounds that shrink with the graph's algebraic connectivity, and
robustness to Laplacian misspecification. We also reinterpret GS-B$^3$SE
through information geometry, showing that it generalizes existing shift
estimators.

### 4. [Higher-Order Asymptotics of Test-Time Adaptation for Batch Normalization Statistics](http://arxiv.org/pdf/2505.16257v1)

Authors: Masanari Kimura

This study develops a higher-order asymptotic framework for test-time
adaptation (TTA) of Batch Normalization (BN) statistics under distribution
shift by integrating classical Edgeworth expansion and saddlepoint
approximation techniques with a novel one-step M-estimation perspective. By
analyzing the statistical discrepancy between training and test distributions,
we derive an Edgeworth expansion for the normalized difference in BN means and
obtain an optimal weighting parameter that minimizes the mean-squared error of
the adapted statistic. Reinterpreting BN TTA as a one-step M-estimator allows
us to derive higher-order local asymptotic normality results, which incorporate
skewness and other higher moments into the estimator's behavior. Moreover, we
quantify the trade-offs among bias, variance, and skewness in the adaptation
process and establish a corresponding generalization bound on the model risk.
The refined saddlepoint approximations further deliver uniformly accurate
density and tail probability estimates for the BN TTA statistic. These
theoretical insights provide a comprehensive understanding of how higher-order
corrections and robust one-step updating can enhance the reliability and
performance of BN layers in adapting to changing data distributions.

### 5. [Better Rates for Private Linear Regression in the Proportional Regime via Aggressive Clipping](http://arxiv.org/pdf/2505.16329v1)

Authors: Simone Bombari, Inbar Seroussi, Marco Mondelli

Differentially private (DP) linear regression has received significant
attention in the recent theoretical literature, with several works aimed at
obtaining improved error rates. A common approach is to set the clipping
constant much larger than the expected norm of the per-sample gradients. While
simplifying the analysis, this is however in sharp contrast with what empirical
evidence suggests to optimize performance. Our work bridges this gap between
theory and practice: we provide sharper rates for DP stochastic gradient
descent (DP-SGD) by crucially operating in a regime where clipping happens
frequently. Specifically, we consider the setting where the data is
multivariate Gaussian, the number of training samples $n$ is proportional to
the input dimension $d$, and the algorithm guarantees constant-order zero
concentrated DP. Our method relies on establishing a deterministic equivalent
for the trajectory of DP-SGD in terms of a family of ordinary differential
equations (ODEs). As a consequence, the risk of DP-SGD is bounded between two
ODEs, with upper and lower bounds matching for isotropic data. By studying
these ODEs when $n / d$ is large enough, we demonstrate the optimality of
aggressive clipping, and we uncover the benefits of decaying learning rate and
private noise scheduling.

### 6. [Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling](http://arxiv.org/pdf/2505.16481v1)

Authors: Xinxing Shi, Xiaoyu Jiang, Mauricio A. Álvarez

Gaussian Process (GP) Variational Autoencoders (VAEs) extend standard VAEs by
replacing the fully factorised Gaussian prior with a GP prior, thereby
capturing richer correlations among latent variables. However, performing exact
GP inference in large-scale GPVAEs is computationally prohibitive, often
forcing existing approaches to rely on restrictive kernel assumptions or large
sets of inducing points. In this work, we propose a neighbour-driven
approximation strategy that exploits local adjacencies in the latent space to
achieve scalable GPVAE inference. By confining computations to the nearest
neighbours of each data point, our method preserves essential latent
dependencies, allowing more flexible kernel choices and mitigating the need for
numerous inducing points. Through extensive experiments on tasks including
representation learning, data imputation, and conditional generation, we
demonstrate that our approach outperforms other GPVAE variants in both
predictive performance and computational efficiency.

### 7. [Incremental Sequence Classification with Temporal Consistency](http://arxiv.org/pdf/2505.16548v1)

Authors: Lucas Maystre, Gabriel Barello, Tudor Berariu, Aleix Cambray, Rares Dolga, Alvaro Ortega Gonzalez, Andrei Nica, David Barber

We address the problem of incremental sequence classification, where
predictions are updated as new elements in the sequence are revealed. Drawing
on temporal-difference learning from reinforcement learning, we identify a
temporal-consistency condition that successive predictions should satisfy. We
leverage this condition to develop a novel loss function for training
incremental sequence classifiers. Through a concrete example, we demonstrate
that optimizing this loss can offer substantial gains in data efficiency. We
apply our method to text classification tasks and show that it improves
predictive accuracy over competing approaches on several benchmark datasets. We
further evaluate our approach on the task of verifying large language model
generations for correctness in grade-school math problems. Our results show
that models trained with our method are better able to distinguish promising
generations from unpromising ones after observing only a few tokens.

### 8. [How high is `high'? Rethinking the roles of dimensionality in topological data analysis and manifold learning](http://arxiv.org/pdf/2505.16879v1)

Authors: Hannah Sansford, Nick Whiteley, Patrick Rubin-Delanchy

We present a generalised Hanson-Wright inequality and use it to establish new
statistical insights into the geometry of data point-clouds. In the setting of
a general random function model of data, we clarify the roles played by three
notions of dimensionality: ambient intrinsic dimension $p_{\mathrm{int}}$,
which measures total variability across orthogonal feature directions;
correlation rank, which measures functional complexity across samples; and
latent intrinsic dimension, which is the dimension of manifold structure hidden
in data. Our analysis shows that in order for persistence diagrams to reveal
latent homology and for manifold structure to emerge it is sufficient that
$p_{\mathrm{int}}\gg \log n$, where $n$ is the sample size. Informed by these
theoretical perspectives, we revisit the ground-breaking neuroscience discovery
of toroidal structure in grid-cell activity made by Gardner et al. (Nature,
2022): our findings reveal, for the first time, evidence that this structure is
in fact isometric to physical space, meaning that grid cell activity conveys a
geometrically faithful representation of the real world.

### 9. [Statistical Test for Saliency Maps of Graph Neural Networks via Selective Inference](http://arxiv.org/pdf/2505.16893v1)

Authors: Shuichi Nishino, Tomohiro Shiraishi, Teruyuki Katsuoka, Ichiro Takeuchi

Graph Neural Networks (GNNs) have gained prominence for their ability to
process graph-structured data across various domains. However, interpreting GNN
decisions remains a significant challenge, leading to the adoption of saliency
maps for identifying influential nodes and edges. Despite their utility, the
reliability of GNN saliency maps has been questioned, particularly in terms of
their robustness to noise. In this study, we propose a statistical testing
framework to rigorously evaluate the significance of saliency maps. Our main
contribution lies in addressing the inflation of the Type I error rate caused
by double-dipping of data, leveraging the framework of Selective Inference. Our
method provides statistically valid $p$-values while controlling the Type I
error rate, ensuring that identified salient subgraphs contain meaningful
information rather than random artifacts. To demonstrate the effectiveness of
our method, we conduct experiments on both synthetic and real-world datasets,
showing its effectiveness in assessing the reliability of GNN interpretations.

### 10. [TULiP: Test-time Uncertainty Estimation via Linearization and Weight Perturbation](http://arxiv.org/pdf/2505.16923v1)

Authors: Yuhui Zhang, Dongshen Wu, Yuichiro Wada, Takafumi Kanamori

A reliable uncertainty estimation method is the foundation of many modern
out-of-distribution (OOD) detectors, which are critical for safe deployments of
deep learning models in the open world. In this work, we propose TULiP, a
theoretically-driven post-hoc uncertainty estimator for OOD detection. Our
approach considers a hypothetical perturbation applied to the network before
convergence. Based on linearized training dynamics, we bound the effect of such
perturbation, resulting in an uncertainty score computable by perturbing model
parameters. Ultimately, our approach computes uncertainty from a set of sampled
predictions. We visualize our bound on synthetic regression and classification
datasets. Furthermore, we demonstrate the effectiveness of TULiP using
large-scale OOD detection benchmarks for image classification. Our method
exhibits state-of-the-art performance, particularly for near-distribution
samples.

