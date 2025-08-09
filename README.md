# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-08 17:00:25.630375 PST.

### Artificial Intelligence

### 1. [The Docking Game: Loop Self-Play for Fast, Dynamic, and Accurate Prediction of Flexible Protein--Ligand Binding](http://arxiv.org/pdf/2508.05006v1)

Authors: Youzhi Zhang, Yufei Li, Gaofeng Meng, Hongbin Liu, Jiebo Luo

Molecular docking is a crucial aspect of drug discovery, as it predicts the
binding interactions between small-molecule ligands and protein pockets.
However, current multi-task learning models for docking often show inferior
performance in ligand docking compared to protein pocket docking. This
disparity arises largely due to the distinct structural complexities of ligands
and proteins. To address this issue, we propose a novel game-theoretic
framework that models the protein-ligand interaction as a two-player game
called the Docking Game, with the ligand docking module acting as the ligand
player and the protein pocket docking module as the protein player. To solve
this game, we develop a novel Loop Self-Play (LoopPlay) algorithm, which
alternately trains these players through a two-level loop. In the outer loop,
the players exchange predicted poses, allowing each to incorporate the other's
structural predictions, which fosters mutual adaptation over multiple
iterations. In the inner loop, each player dynamically refines its predictions
by incorporating its own predicted ligand or pocket poses back into its model.
We theoretically show the convergence of LoopPlay, ensuring stable
optimization. Extensive experiments conducted on public benchmark datasets
demonstrate that LoopPlay achieves approximately a 10\% improvement in
predicting accurate binding modes compared to previous state-of-the-art
methods. This highlights its potential to enhance the accuracy of molecular
docking in drug discovery.

### 2. [MedMKEB: A Comprehensive Knowledge Editing Benchmark for Medical Multimodal Large Language Models](http://arxiv.org/pdf/2508.05083v1)

Authors: Dexuan Xu, Jieyi Wang, Zhongyan Chai, Yongzhi Cao, Hanpin Wang, Huamin Zhang, Yu Huang

Recent advances in multimodal large language models (MLLMs) have
significantly improved medical AI, enabling it to unify the understanding of
visual and textual information. However, as medical knowledge continues to
evolve, it is critical to allow these models to efficiently update outdated or
incorrect information without retraining from scratch. Although textual
knowledge editing has been widely studied, there is still a lack of systematic
benchmarks for multimodal medical knowledge editing involving image and text
modalities. To fill this gap, we present MedMKEB, the first comprehensive
benchmark designed to evaluate the reliability, generality, locality,
portability, and robustness of knowledge editing in medical multimodal large
language models. MedMKEB is built on a high-quality medical visual
question-answering dataset and enriched with carefully constructed editing
tasks, including counterfactual correction, semantic generalization, knowledge
transfer, and adversarial robustness. We incorporate human expert validation to
ensure the accuracy and reliability of the benchmark. Extensive single editing
and sequential editing experiments on state-of-the-art general and medical
MLLMs demonstrate the limitations of existing knowledge-based editing
approaches in medicine, highlighting the need to develop specialized editing
strategies. MedMKEB will serve as a standard benchmark to promote the
development of trustworthy and efficient medical knowledge editing algorithms.

### 3. [EasySize: Elastic Analog Circuit Sizing via LLM-Guided Heuristic Search](http://arxiv.org/pdf/2508.05113v1)

Authors: Xinyue Wu, Fan Hu, Shaik Jani Babu, Yi Zhao, Xinfei Guo

Analog circuit design is a time-consuming, experience-driven task in chip
development. Despite advances in AI, developing universal, fast, and stable
gate sizing methods for analog circuits remains a significant challenge. Recent
approaches combine Large Language Models (LLMs) with heuristic search
techniques to enhance generalizability, but they often depend on large model
sizes and lack portability across different technology nodes. To overcome these
limitations, we propose EasySize, the first lightweight gate sizing framework
based on a finetuned Qwen3-8B model, designed for universal applicability
across process nodes, design specifications, and circuit topologies. EasySize
exploits the varying Ease of Attainability (EOA) of performance metrics to
dynamically construct task-specific loss functions, enabling efficient
heuristic search through global Differential Evolution (DE) and local Particle
Swarm Optimization (PSO) within a feedback-enhanced flow. Although finetuned
solely on 350nm node data, EasySize achieves strong performance on 5
operational amplifier (Op-Amp) netlists across 180nm, 45nm, and 22nm technology
nodes without additional targeted training, and outperforms AutoCkt, a
widely-used Reinforcement Learning based sizing framework, on 86.67\% of tasks
with more than 96.67\% of simulation resources reduction. We argue that
EasySize can significantly reduce the reliance on human expertise and
computational resources in gate sizing, thereby accelerating and simplifying
the analog circuit design process. EasySize will be open-sourced at a later
date.

### 4. [Graph-based Event Log Repair](http://arxiv.org/pdf/2508.05145v1)

Authors: Sebastiano Dissegna, Chiara Di Francescomarino, Massimiliano Ronzani

The quality of event logs in Process Mining is crucial when applying any form
of analysis to them. In real-world event logs, the acquisition of data can be
non-trivial (e.g., due to the execution of manual activities and related manual
recording or to issues in collecting, for each event, all its attributes), and
often may end up with events recorded with some missing information. Standard
approaches to the problem of trace (or log) reconstruction either require the
availability of a process model that is used to fill missing values by
leveraging different reasoning techniques or employ a Machine Learning/Deep
Learning model to restore the missing values by learning from similar cases. In
recent years, a new type of Deep Learning model that is capable of handling
input data encoded as graphs has emerged, namely Graph Neural Networks. Graph
Neural Network models, and even more so Heterogeneous Graph Neural Networks,
offer the advantage of working with a more natural representation of complex
multi-modal sequences like the execution traces in Process Mining, allowing for
more expressive and semantically rich encodings.
  In this work, we focus on the development of a Heterogeneous Graph Neural
Network model that, given a trace containing some incomplete events, will
return the full set of attributes missing from those events. We evaluate our
work against a state-of-the-art approach leveraging autoencoders on two
synthetic logs and four real event logs, on different types of missing values.
Different from state-of-the-art model-free approaches, which mainly focus on
repairing a subset of event attributes, the proposed approach shows very good
performance in reconstructing all different event attributes.

### 5. [An Explainable Natural Language Framework for Identifying and Notifying Target Audiences In Enterprise Communication](http://arxiv.org/pdf/2508.05267v1)

Authors: Vítor N. Lourenço, Mohnish Dubey, Yunfei Bai, Audrey Depeige, Vivek Jain

In large-scale maintenance organizations, identifying subject matter experts
and managing communications across complex entities relationships poses
significant challenges -- including information overload and longer response
times -- that traditional communication approaches fail to address effectively.
We propose a novel framework that combines RDF graph databases with LLMs to
process natural language queries for precise audience targeting, while
providing transparent reasoning through a planning-orchestration architecture.
Our solution enables communication owners to formulate intuitive queries
combining concepts such as equipment, manufacturers, maintenance engineers, and
facilities, delivering explainable results that maintain trust in the system
while improving communication efficiency across the organization.

### 6. [NomicLaw: Emergent Trust and Strategic Argumentation in LLMs During Collaborative Law-Making](http://arxiv.org/pdf/2508.05344v1)

Authors: Asutosh Hota, Jussi P. P. Jokinen

Recent advancements in large language models (LLMs) have extended their
capabilities from basic text processing to complex reasoning tasks, including
legal interpretation, argumentation, and strategic interaction. However,
empirical understanding of LLM behavior in open-ended, multi-agent settings
especially those involving deliberation over legal and ethical dilemmas remains
limited. We introduce NomicLaw, a structured multi-agent simulation where LLMs
engage in collaborative law-making, responding to complex legal vignettes by
proposing rules, justifying them, and voting on peer proposals. We
quantitatively measure trust and reciprocity via voting patterns and
qualitatively assess how agents use strategic language to justify proposals and
influence outcomes. Experiments involving homogeneous and heterogeneous LLM
groups demonstrate how agents spontaneously form alliances, betray trust, and
adapt their rhetoric to shape collective decisions. Our results highlight the
latent social reasoning and persuasive capabilities of ten open-source LLMs and
provide insights into the design of future AI systems capable of autonomous
negotiation, coordination and drafting legislation in legal settings.

### 7. [StructVRM: Aligning Multimodal Reasoning with Structured and Verifiable Reward Models](http://arxiv.org/pdf/2508.05383v1)

Authors: Xiangxiang Zhang, Jingxuan Wei, Donghong Zhong, Qi Chen, Caijun Jia, Cheng Tan, Jinming Gu, Xiaobo Qin, Zhiping Liu, Liang Hu, Tong Sun, Yuchen Wu, Zewei Sun, Chenwei Lou, Hua Zheng, Tianyang Zhan, Changbao Wang, Shuangzhi Wu, Zefa Lin, Chang Guo, Sihang Yuan, Riwei Chen, Shixiong Zhao, Yingping Zhang, Gaowei Wu, Bihui Yu, Jiahui Wu, Zhehui Zhao, Qianqian Liu, Ruofeng Tang, Xingyue Huang, Bing Zhao, Mengyang Zhang, Youqiang Zhou

Existing Vision-Language Models often struggle with complex, multi-question
reasoning tasks where partial correctness is crucial for effective learning.
Traditional reward mechanisms, which provide a single binary score for an
entire response, are too coarse to guide models through intricate problems with
multiple sub-parts. To address this, we introduce StructVRM, a method that
aligns multimodal reasoning with Structured and Verifiable Reward Models. At
its core is a model-based verifier trained to provide fine-grained,
sub-question-level feedback, assessing semantic and mathematical equivalence
rather than relying on rigid string matching. This allows for nuanced, partial
credit scoring in previously intractable problem formats. Extensive experiments
demonstrate the effectiveness of StructVRM. Our trained model, Seed-StructVRM,
achieves state-of-the-art performance on six out of twelve public multimodal
benchmarks and our newly curated, high-difficulty STEM-Bench. The success of
StructVRM validates that training with structured, verifiable rewards is a
highly effective approach for advancing the capabilities of multimodal models
in complex, real-world reasoning domains.

### 8. [An Explainable Machine Learning Framework for Railway Predictive Maintenance using Data Streams from the Metro Operator of Portugal](http://arxiv.org/pdf/2508.05388v1)

Authors: Silvia García-Méndez, Francisco de Arriba-Pérez, Fátima Leal, Bruno Veloso, Benedita Malheiro, Juan Carlos Burguillo-Rial

This work contributes to a real-time data-driven predictive maintenance
solution for Intelligent Transportation Systems. The proposed method implements
a processing pipeline comprised of sample pre-processing, incremental
classification with Machine Learning models, and outcome explanation. This
novel online processing pipeline has two main highlights: (i) a dedicated
sample pre-processing module, which builds statistical and frequency-related
features on the fly, and (ii) an explainability module. This work is the first
to perform online fault prediction with natural language and visual
explainability. The experiments were performed with the MetroPT data set from
the metro operator of Porto, Portugal. The results are above 98 % for F-measure
and 99 % for accuracy. In the context of railway predictive maintenance,
achieving these high values is crucial due to the practical and operational
implications of accurate failure prediction. In the specific case of a high
F-measure, this ensures that the system maintains an optimal balance between
detecting the highest possible number of real faults and minimizing false
alarms, which is crucial for maximizing service availability. Furthermore, the
accuracy obtained enables reliability, directly impacting cost reduction and
increased safety. The analysis demonstrates that the pipeline maintains high
performance even in the presence of class imbalance and noise, and its
explanations effectively reflect the decision-making process. These findings
validate the methodological soundness of the approach and confirm its practical
applicability for supporting proactive maintenance decisions in real-world
railway operations. Therefore, by identifying the early signs of failure, this
pipeline enables decision-makers to understand the underlying problems and act
accordingly swiftly.

### 9. [DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning](http://arxiv.org/pdf/2508.05405v1)

Authors: Xinrun Xu, Pi Bu, Ye Wang, Börje F. Karlsson, Ziming Wang, Tengtao Song, Qi Zhu, Jun Song, Zhiming Ding, Bo Zheng

Although Vision Language Models (VLMs) exhibit strong perceptual abilities
and impressive visual reasoning, they struggle with attention to detail and
precise action planning in complex, dynamic environments, leading to subpar
performance. Real-world tasks typically require complex interactions, advanced
spatial reasoning, long-term planning, and continuous strategy refinement,
usually necessitating understanding the physics rules of the target scenario.
However, evaluating these capabilities in real-world scenarios is often
prohibitively expensive. To bridge this gap, we introduce DeepPHY, a novel
benchmark framework designed to systematically evaluate VLMs' understanding and
reasoning about fundamental physical principles through a series of challenging
simulated environments. DeepPHY integrates multiple physical reasoning
environments of varying difficulty levels and incorporates fine-grained
evaluation metrics. Our evaluation finds that even state-of-the-art VLMs
struggle to translate descriptive physical knowledge into precise, predictive
control.

### 10. [Large Language Models Transform Organic Synthesis From Reaction Prediction to Automation](http://arxiv.org/pdf/2508.05427v1)

Authors: Kartar Kumar Lohana Tharwani, Rajesh Kumar, Sumita, Numan Ahmed, Yong Tang

Large language models (LLMs) are beginning to reshape how chemists plan and
run reactions in organic synthesis. Trained on millions of reported
transformations, these text-based models can propose synthetic routes, forecast
reaction outcomes and even instruct robots that execute experiments without
human supervision. Here we survey the milestones that turned LLMs from
speculative tools into practical lab partners. We show how coupling LLMs with
graph neural networks, quantum calculations and real-time spectroscopy shrinks
discovery cycles and supports greener, data-driven chemistry. We discuss
limitations, including biased datasets, opaque reasoning and the need for
safety gates that prevent unintentional hazards. Finally, we outline community
initiatives open benchmarks, federated learning and explainable interfaces that
aim to democratize access while keeping humans firmly in control. These
advances chart a path towards rapid, reliable and inclusive molecular
innovation powered by artificial intelligence and automation.

### Hardware Architecture

### 1. [relOBI: A Reliable Low-latency Interconnect for Tightly-Coupled On-chip Communication](http://arxiv.org/pdf/2508.05354v1)

Authors: Michael Rogenmoser, Angelo Garofalo, Luca Benini

On-chip communication is a critical element of modern systems-on-chip (SoCs),
allowing processor cores to interact with memory and peripherals. Interconnects
require special care in radiation-heavy environments, as any soft error within
the SoC interconnect is likely to cause a functional failure of the whole SoC.
This work proposes relOBI, an extension to Open Bus Interface (OBI) combining
triple modular redundancy (TMR) for critical handshake signals with error
correction codes (ECC) protection on other signals for complete reliability.
Implementing and testing a fully reliable crossbar shows improved reliability
to injected faults from a vulnerability of 34.85 % to 0 % compared to a
reference design, with an area increase of 2.6x and 1.4x timing impact. The
area overhead is 1.8x lower than that reported in the literature for
fine-grained triplication and voting.

### 2. [Understanding and Mitigating Errors of LLM-Generated RTL Code](http://arxiv.org/pdf/2508.05266v1)

Authors: Jiazheng Zhang, Cheng Liu, Huawei Li

Despite the promising potential of large language model (LLM) based
register-transfer-level (RTL) code generation, the overall success rate remains
unsatisfactory. Errors arise from various factors, with limited understanding
of specific failure causes hindering improvement. To address this, we conduct a
comprehensive error analysis and manual categorization. Our findings reveal
that most errors stem not from LLM reasoning limitations, but from insufficient
RTL programming knowledge, poor understanding of circuit concepts, ambiguous
design descriptions, or misinterpretation of complex multimodal inputs.
Leveraging in-context learning, we propose targeted error correction
techniques. Specifically, we construct a domain-specific knowledge base and
employ retrieval-augmented generation (RAG) to supply necessary RTL knowledge.
To mitigate ambiguity errors, we introduce design description rules and
implement a rule-checking mechanism. For multimodal misinterpretation, we
integrate external tools to convert inputs into LLM-compatible meta-formats.
For remaining errors, we adopt an iterative debugging loop (simulation-error
localization-correction). Integrating these techniques into an LLM-based
framework significantly improves performance. We incorporate these error
correction techniques into a foundational LLM-based RTL code generation
framework, resulting in significantly improved performance. Experimental
results show that our enhanced framework achieves 91.0\% accuracy on the
VerilogEval benchmark, surpassing the baseline code generation approach by
32.7\%, demonstrating the effectiveness of our methods.

### Computational Complexity

### 1. [Minimal Model Reasoning in Description Logics: Don't Try This at Home!](http://arxiv.org/pdf/2508.05350v1)

Authors: Federica Di Stefano, Quentin Manière, Magdalena Ortiz, Mantas Šimkus

Reasoning with minimal models has always been at the core of many knowledge
representation techniques, but we still have only a limited understanding of
this problem in Description Logics (DLs). Minimization of some selected
predicates, letting the remaining predicates vary or be fixed, as proposed in
circumscription, has been explored and exhibits high complexity. The case of
`pure' minimal models, where the extension of all predicates must be minimal,
has remained largely uncharted. We address this problem in popular DLs and
obtain surprisingly negative results: concept satisfiability in minimal models
is undecidable already for $\mathcal{EL}$. This undecidability also extends to
a very restricted fragment of tuple-generating dependencies. To regain
decidability, we impose acyclicity conditions on the TBox that bring the
worst-case complexity below double exponential time and allow us to establish a
connection with the recently studied pointwise circumscription; we also derive
results in data complexity. We conclude with a brief excursion to the DL-Lite
family, where a positive result was known for DL-Lite$_{\text{core}}$, but our
investigation establishes ExpSpace-hardness already for its extension
DL-Lite$_{\text{horn}}$.

### 2. [NP-Hardness and ETH-Based Inapproximability of Communication Complexity via Relaxed Interlacing](http://arxiv.org/pdf/2508.05597v1)

Authors: Serge Gaspers, Zixu He, Simon Mackenzie

We prove that computing the deterministic communication complexity D(f) of a
Boolean function is NP-hard, even when protocols are limited to a constant
number of alternations, resolving a question first posed by Yao (1979). Our
reduction builds and expands on a suite of structural "interlacing" lemmas
introduced by Mackenzie and Saffidine (arXiv:2411.19003); these lemmas can be
reused as black boxes in future lower-bound constructions.
  The instances produced by our reduction admit optimal protocols that use only
constant alternations, so NP-hardness holds under stronger restrictions than
those considered in concurrent and independent work by Hirahara, Ilango, and
Loff (arXiv:2507.10426), whose proof requires unbounded alternations.
  Because the gadgets in our construction are self-similar, they can be
recursively embedded. We sketch how this yields, under the Exponential-Time
Hypothesis, an additive inapproximability gap that grows without bound, and we
outline a route toward NP-hardness of approximating D(f) within a fixed
constant additive error. Full details of the ETH-based inapproximability
results will appear in a future version.
  Beyond settling the complexity of deterministic communication complexity
itself, the modular framework we develop opens the door to a wider class of
reductions and, we believe, will prove useful in tackling other long-standing
questions in communication complexity.

### Computational Engineering

### 1. [Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha](http://arxiv.org/pdf/2508.04975v1)

Authors: Qizhao Chen, Hiroaki Kawashima

Traditionally, traders and quantitative analysts address alpha decay by
manually crafting formulaic alphas, mathematical expressions that identify
patterns or signals in financial data, through domain expertise and
trial-and-error. This process is often time-consuming and difficult to scale.
With recent advances in large language models (LLMs), it is now possible to
automate the generation of such alphas by leveraging the reasoning capabilities
of LLMs. This paper introduces a novel framework that integrates a prompt-based
LLM with a Transformer model for stock price prediction. The LLM first
generates diverse and adaptive alphas using structured inputs such as
historical stock features (Close, Open, High, Low, Volume), technical
indicators, sentiment scores of both target and related companies. These
alphas, instead of being used directly for trading, are treated as high-level
features that capture complex dependencies within the financial data. To
evaluate the effectiveness of these LLM-generated formulaic alphas, the alpha
features are then fed into prediction models such as Transformer, LSTM, TCN,
SVR, and Random Forest to forecast future stock prices. Experimental results
demonstrate that the LLM-generated alphas significantly improve predictive
accuracy. Moreover, the accompanying natural language reasoning provided by the
LLM enhances the interpretability and transparency of the predictions,
supporting more informed financial decision-making.

### 2. [Fuzzy Decisions on Fluid Instabilities: Autoencoder-Based Reconstruction meets Rule-Based Anomaly Classification](http://arxiv.org/pdf/2508.05418v1)

Authors: Bharadwaj Dogga, Gibin M. Raju, Wilhelm Louw, Kelly Cohen

Shockwave classification in shadowgraph imaging is challenging due to limited
labeled data and complex flow structures. This study presents a hybrid
framework that combines unsupervised autoencoder models with a fuzzy inference
system to generate and interpret anomaly maps. Among the evaluated methods, the
hybrid $\beta$-VAE autoencoder with a fuzzy rule-based system most effectively
captured coherent shock features, integrating spatial context to enhance
anomaly classification. The resulting approach enables interpretable,
unsupervised classification of flow disruptions and lays the groundwork for
real-time, physics-informed diagnostics in experimental and industrial fluid
applications.

### 3. [Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation](http://arxiv.org/pdf/2508.05425v1)

Authors: Aluffi Pietro Alessandro, Brandi Jess, Marya Bazzi, Kate Kennedy, Matt Arderne, Daniel Rodrigues, Martin Lotz

Despite their significant economic contributions, Small and Medium
Enterprises (SMEs) face persistent barriers to securing traditional financing
due to information asymmetries. Cash flow lending has emerged as a promising
alternative, but its effectiveness depends on accurate modelling of
transaction-level data. The main challenge in SME transaction analysis lies in
the unstructured nature of textual descriptions, characterised by extreme
abbreviations, limited context, and imbalanced label distributions. While
consumer transaction descriptions often show significant commonalities across
individuals, SME transaction descriptions are typically nonstandard and
inconsistent across businesses and industries. To address some of these
challenges, we propose a bank categorisation pipeline that leverages synthetic
data generation to augment existing transaction data sets. Our approach
comprises three core components: (1) a synthetic data generation module that
replicates transaction properties while preserving context and semantic
meaning; (2) a fine-tuned classification model trained on this enriched
dataset; and (3) a calibration methodology that aligns model outputs with
real-world label distributions. Experimental results demonstrate that our
approach achieves 73.49% (+-5.09) standard accuracy on held-out data, with
high-confidence predictions reaching 90.36% (+-6.52) accuracy. The model
exhibits robust generalisation across different types of SMEs and transactions,
which makes it suitable for practical deployment in cash-flow lending
applications. By addressing core data challenges, namely, scarcity, noise, and
imbalance, our framework provides a practical solution to build robust
classification systems in data-sparse SME lending contexts.

### 4. [Latent Space Diffusion for Topology Optimization](http://arxiv.org/pdf/2508.05624v1)

Authors: Aaron Lutheran, Srijan Das, Alireza Tabarraei

Topology optimization enables the automated design of efficient structures by
optimally distributing material within a defined domain. However, traditional
gradient-based methods often scale poorly with increasing resolution and
dimensionality due to the need for repeated finite element analyses and
sensitivity evaluations. In this work, we propose a novel framework that
combines latent diffusion models (LDMs) with variational autoencoders (VAEs) to
enable fast, conditional generation of optimized topologies. Unlike prior
approaches, our method conditions the generative process on physically
meaningful fields, specifically von Mises stress, strain energy density, volume
fraction, and loading information, embedded as dense input channels. To further
guide the generation process, we introduce auxiliary loss functions that
penalize floating material, load imbalance, and volume fraction deviation,
thereby encouraging physically realistic and manufacturable designs. Numerical
experiments on a large synthetic dataset demonstrate that our VAE-LDM framework
outperforms existing diffusion-based methods in compliance accuracy, volume
control, and structural connectivity, providing a robust and scalable
alternative to conventional

### 5. [Task-Based Programming for Adaptive Mesh Refinement in Compressible Flow Simulations](http://arxiv.org/pdf/2508.05020v1)

Authors: Anjiang Wei, Hang Song, Mert Hidayetoglu, Elliott Slaughter, Sanjiva K. Lele, Alex Aiken

High-order solvers for compressible flows are vital in scientific
applications. Adaptive mesh refinement (AMR) is a key technique for reducing
computational cost by concentrating resolution in regions of interest. In this
work, we develop an AMR-based numerical solver using Regent, a high-level
programming language for the Legion programming model. We address several
challenges associated with implementing AMR in Regent. These include dynamic
data structures for patch refinement/coarsening, mesh validity enforcement, and
reducing task launch overhead via task fusion. Experimental results show that
task fusion achieves 18x speedup, while automated GPU kernel generation via
simple annotations yields 9.7x speedup for the targeted kernel. We demonstrate
our approach through simulations of two canonical compressible flow problems
governed by the Euler equations.

### 6. [Echo State Networks for Bitcoin Time Series Prediction](http://arxiv.org/pdf/2508.05416v1)

Authors: Mansi Sharma, Enrico Sartor, Marc Cavazza, Helmut Prendinger

Forecasting stock and cryptocurrency prices is challenging due to high
volatility and non-stationarity, influenced by factors like economic changes
and market sentiment. Previous research shows that Echo State Networks (ESNs)
can effectively model short-term stock market movements, capturing nonlinear
patterns in dynamic data. To the best of our knowledge, this work is among the
first to explore ESNs for cryptocurrency forecasting, especially during extreme
volatility. We also conduct chaos analysis through the Lyapunov exponent in
chaotic periods and show that our approach outperforms existing machine
learning methods by a significant margin. Our findings are consistent with the
Lyapunov exponent analysis, showing that ESNs are robust during chaotic periods
and excel under high chaos compared to Boosting and Na\"ive methods.

### 7. [Deconstructing the Crystal Ball: From Ad-Hoc Prediction to Principled Startup Evaluation with the SAISE Framework](http://arxiv.org/pdf/2508.05491v1)

Authors: Seyed Mohammad Ali Jafari, Ali Mobini Dehkordi, Ehsan Chitsaz, Yadollah Yaghoobzadeh

The integration of Artificial Intelligence (AI) into startup evaluation
represents a significant technological shift, yet the academic research
underpinning this transition remains methodologically fragmented. Existing
studies often employ ad-hoc approaches, leading to a body of work with
inconsistent definitions of success, atheoretical features, and a lack of
rigorous validation. This fragmentation severely limits the comparability,
reliability, and practical utility of current predictive models.
  To address this critical gap, this paper presents a comprehensive systematic
literature review of 57 empirical studies. We deconstruct the current
state-of-the-art by systematically mapping the features, algorithms, data
sources, and evaluation practices that define the AI-driven startup prediction
landscape. Our synthesis reveals a field defined by a central paradox: a strong
convergence on a common toolkit -- venture databases and tree-based ensembles
-- but a stark divergence in methodological rigor. We identify four
foundational weaknesses: a fragmented definition of "success," a divide between
theory-informed and data-driven feature engineering, a chasm between common and
best-practice model validation, and a nascent approach to data ethics and
explainability.
  In response to these findings, our primary contribution is the proposal of
the Systematic AI-driven Startup Evaluation (SAISE) Framework. This novel,
five-stage prescriptive roadmap is designed to guide researchers from ad-hoc
prediction toward principled evaluation. By mandating a coherent, end-to-end
methodology that emphasizes stage-aware problem definition, theory-informed
data synthesis, principled feature engineering, rigorous validation, and
risk-aware interpretation, the SAISE framework provides a new standard for
conducting more comparable, robust, and practically relevant research in this
rapidly maturing domain

### Computational Geometry

### 1. [An Improved Physically-Based Surface Triangulation Method](http://arxiv.org/pdf/2508.05099v1)

Authors: Lei Shangyu, Fan Wei, Ren Hui

This paper proposes improvements to the physically-based surface
triangulation method, bubble meshing. The method simulates physical bubbles to
automatically generate mesh vertices, resulting in high-quality Delaunay
triangles. Despite its flexibility in local mesh size control and the advantage
of local re-meshing, bubble meshing is constrained by high computational costs
and slow convergence on complex surfaces. The proposed approach employs
conformal mapping to simplify surface bubble packing by flattening the surface
onto a plane. Surface triangulation is induced from the planar mesh, avoiding
direct bubble movement on the surface. Optimizing bubble quantity control and
separating it from the relaxation process accelerates convergence, cutting
computation time by over 70%. The enhanced method enables efficient
triangulation of disk topology surfaces, supports local size control, curvature
adaptation, and re-meshing of discrete surfaces. Keywords: Adaptive
triangulation, Surface remeshing, Bubble meshing, Conformal parameterization,
Algorithm efficiency

### 2. [GASP: A Gradient-Aware Shortest Path Algorithm for Boundary-Confined Visualization of 2-Manifold Reeb Graphs](http://arxiv.org/pdf/2508.05524v1)

Authors: Sefat Rahman, Tushar M. Athawale, Paul Rosen

Reeb graphs are an important tool for abstracting and representing the
topological structure of a function defined on a manifold. We have identified
three properties for faithfully representing Reeb graphs in a visualization.
Namely, they should be constrained to the boundary, compact, and aligned with
the function gradient. Existing algorithms for drawing Reeb graphs are agnostic
to or violate these properties. In this paper, we introduce an algorithm to
generate Reeb graph visualizations, called \textit{GASP}, that is cognizant of
these properties, thereby producing visualizations that are more representative
of the underlying data. To demonstrate the improvements, the resulting Reeb
graphs are evaluated both qualitatively and quantitatively against the
geometric barycenter algorithm, using its implementation available in the
Topology ToolKit (TTK), a widely adopted tool for calculating and visualizing
Reeb graphs.

### Computation and Language

### 1. [Multimodal Fact Checking with Unified Visual, Textual, and Contextual Representations](http://arxiv.org/pdf/2508.05097v1)

Authors: Aditya Kishore, Gaurav Kumar, Jasabanta Patro

The growing rate of multimodal misinformation, where claims are supported by
both text and images, poses significant challenges to fact-checking systems
that rely primarily on textual evidence. In this work, we have proposed a
unified framework for fine-grained multimodal fact verification called
"MultiCheck", designed to reason over structured textual and visual signals.
Our architecture combines dedicated encoders for text and images with a fusion
module that captures cross-modal relationships using element-wise interactions.
A classification head then predicts the veracity of a claim, supported by a
contrastive learning objective that encourages semantic alignment between
claim-evidence pairs in a shared latent space. We evaluate our approach on the
Factify 2 dataset, achieving a weighted F1 score of 0.84, substantially
outperforming the baseline. These results highlight the effectiveness of
explicit multimodal reasoning and demonstrate the potential of our approach for
scalable and interpretable fact-checking in complex, real-world scenarios.

### 2. [BEE-RAG: Balanced Entropy Engineering for Retrieval-Augmented Generation](http://arxiv.org/pdf/2508.05100v1)

Authors: Yuhao Wang, Ruiyang Ren, Yucheng Wang, Jing Liu, Wayne Xin Zhao, Hua Wu, Haifeng Wang

With the rapid advancement of large language models (LLMs),
retrieval-augmented generation (RAG) has emerged as a critical approach to
supplement the inherent knowledge limitations of LLMs. However, due to the
typically large volume of retrieved information, RAG tends to operate with long
context lengths. From the perspective of entropy engineering, we identify
unconstrained entropy growth and attention dilution due to long retrieval
context as significant factors affecting RAG performance. In this paper, we
propose the balanced entropy-engineered RAG (BEE-RAG) framework, which improves
the adaptability of RAG systems to varying context lengths through the
principle of entropy invariance. By leveraging balanced context entropy to
reformulate attention dynamics, BEE-RAG separates attention sensitivity from
context length, ensuring a stable entropy level. Building upon this, we
introduce a zero-shot inference strategy for multi-importance estimation and a
parameter-efficient adaptive fine-tuning mechanism to obtain the optimal
balancing factor for different settings. Extensive experiments across multiple
RAG tasks demonstrate the effectiveness of BEE-RAG.

### 3. [ATLANTIS at SemEval-2025 Task 3: Detecting Hallucinated Text Spans in Question Answering](http://arxiv.org/pdf/2508.05179v1)

Authors: Catherine Kobus, François Lancelot, Marion-Cécile Martin, Nawal Ould Amer

This paper presents the contributions of the ATLANTIS team to SemEval-2025
Task 3, focusing on detecting hallucinated text spans in question answering
systems. Large Language Models (LLMs) have significantly advanced Natural
Language Generation (NLG) but remain susceptible to hallucinations, generating
incorrect or misleading content. To address this, we explored methods both with
and without external context, utilizing few-shot prompting with a LLM,
token-level classification or LLM fine-tuned on synthetic data. Notably, our
approaches achieved top rankings in Spanish and competitive placements in
English and German. This work highlights the importance of integrating relevant
context to mitigate hallucinations and demonstrate the potential of fine-tuned
models and prompt engineering.

### 4. [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](http://arxiv.org/pdf/2508.05242v1)

Authors: Sijie Wang, Quanjiang Guo, Kai Zhao, Yawei Zhang, Xin Li, Xiang Li, Siqi Li, Rui She, Shangshu Yu, Wee Peng Tay

Code large language models (LLMs) have become indispensable tools for
building efficient and automated coding pipelines. Existing models are
typically post-trained using reinforcement learning (RL) from general-purpose
LLMs using "human instruction-final answer" pairs, where the instructions are
usually from manual annotations. However, collecting high-quality coding
instructions is both labor-intensive and difficult to scale. On the other hand,
code snippets are abundantly available from various sources. This imbalance
presents a major bottleneck in instruction-based post-training. We propose
CodeBoost, a post-training framework that enhances code LLMs purely from code
snippets, without relying on human-annotated instructions. CodeBoost introduces
the following key components: (1) maximum-clique curation, which selects a
representative and diverse training corpus from code; (2) bi-directional
prediction, which enables the model to learn from both forward and backward
prediction objectives; (3) error-aware prediction, which incorporates learning
signals from both correct and incorrect outputs; (4) heterogeneous
augmentation, which diversifies the training distribution to enrich code
semantics; and (5) heterogeneous rewarding, which guides model learning through
multiple reward types including format correctness and execution feedback from
both successes and failures. Extensive experiments across several code LLMs and
benchmarks verify that CodeBoost consistently improves performance,
demonstrating its effectiveness as a scalable and effective training pipeline.

### 5. [ASCoT: An Adaptive Self-Correction Chain-of-Thought Method for Late-Stage Fragility in LLMs](http://arxiv.org/pdf/2508.05282v1)

Authors: Dongxu Zhang, Ning Yang, Jihua Zhu, Jinnan Yang, Miao Xin, Baoliang Tian

Chain-of-Thought (CoT) prompting has significantly advanced the reasoning
capabilities of Large Language Models (LLMs), yet the reliability of these
reasoning chains remains a critical challenge. A widely held "cascading
failure" hypothesis suggests that errors are most detrimental when they occur
early in the reasoning process. This paper challenges that assumption through
systematic error-injection experiments, revealing a counter-intuitive
phenomenon we term "Late-Stage Fragility": errors introduced in the later
stages of a CoT chain are significantly more likely to corrupt the final answer
than identical errors made at the beginning. To address this specific
vulnerability, we introduce the Adaptive Self-Correction Chain-of-Thought
(ASCoT) method. ASCoT employs a modular pipeline in which an Adaptive
Verification Manager (AVM) operates first, followed by the Multi-Perspective
Self-Correction Engine (MSCE). The AVM leverages a Positional Impact Score
function I(k) that assigns different weights based on the position within the
reasoning chains, addressing the Late-Stage Fragility issue by identifying and
prioritizing high-risk, late-stage steps. Once these critical steps are
identified, the MSCE applies robust, dual-path correction specifically to the
failure parts. Extensive experiments on benchmarks such as GSM8K and MATH
demonstrate that ASCoT achieves outstanding accuracy, outperforming strong
baselines, including standard CoT. Our work underscores the importance of
diagnosing specific failure modes in LLM reasoning and advocates for a shift
from uniform verification strategies to adaptive, vulnerability-aware
correction mechanisms.

### 6. [Decision-Making with Deliberation: Meta-reviewing as a Document-grounded Dialogue](http://arxiv.org/pdf/2508.05283v1)

Authors: Sukannya Purkayastha, Nils Dycke, Anne Lauscher, Iryna Gurevych

Meta-reviewing is a pivotal stage in the peer-review process, serving as the
final step in determining whether a paper is recommended for acceptance. Prior
research on meta-reviewing has treated this as a summarization problem over
review reports. However, complementary to this perspective, meta-reviewing is a
decision-making process that requires weighing reviewer arguments and placing
them within a broader context. Prior research has demonstrated that
decision-makers can be effectively assisted in such scenarios via dialogue
agents. In line with this framing, we explore the practical challenges for
realizing dialog agents that can effectively assist meta-reviewers. Concretely,
we first address the issue of data scarcity for training dialogue agents by
generating synthetic data using Large Language Models (LLMs) based on a
self-refinement strategy to improve the relevance of these dialogues to expert
domains. Our experiments demonstrate that this method produces higher-quality
synthetic data and can serve as a valuable resource towards training
meta-reviewing assistants. Subsequently, we utilize this data to train dialogue
agents tailored for meta-reviewing and find that these agents outperform
\emph{off-the-shelf} LLM-based assistants for this task. Finally, we apply our
agents in real-world meta-reviewing scenarios and confirm their effectiveness
in enhancing the efficiency of meta-reviewing.\footnote{Code and Data:
https://github.com/UKPLab/arxiv2025-meta-review-as-dialog

### 7. [SONAR-LLM: Autoregressive Transformer that Thinks in Sentence Embeddings and Speaks in Tokens](http://arxiv.org/pdf/2508.05305v1)

Authors: Nikita Dragunov, Temurbek Rahmatullaev, Elizaveta Goncharova, Andrey Kuznetsov, Anton Razzhigaev

The recently proposed Large Concept Model (LCM) generates text by predicting
a sequence of sentence-level embeddings and training with either mean-squared
error or diffusion objectives. We present SONAR-LLM, a decoder-only transformer
that "thinks" in the same continuous SONAR embedding space, yet is supervised
through token-level cross-entropy propagated via the frozen SONAR decoder. This
hybrid objective retains the semantic abstraction of LCM while eliminating its
diffusion sampler and restoring a likelihood-based training signal. Across
model sizes from 39M to 1.3B parameters, SONAR-LLM attains competitive
generation quality. We report scaling trends, ablations, benchmark results, and
release the complete training code and all pretrained checkpoints to foster
reproducibility and future research.

### 8. [Can Language Models Critique Themselves? Investigating Self-Feedback for Retrieval Augmented Generation at BioASQ 2025](http://arxiv.org/pdf/2508.05366v1)

Authors: Samy Ateia, Udo Kruschwitz

Agentic Retrieval Augmented Generation (RAG) and 'deep research' systems aim
to enable autonomous search processes where Large Language Models (LLMs)
iteratively refine outputs. However, applying these systems to domain-specific
professional search, such as biomedical research, presents challenges, as
automated systems may reduce user involvement and misalign with expert
information needs. Professional search tasks often demand high levels of user
expertise and transparency. The BioASQ CLEF 2025 challenge, using
expert-formulated questions, can serve as a platform to study these issues. We
explored the performance of current reasoning and nonreasoning LLMs like
Gemini-Flash 2.0, o3-mini, o4-mini and DeepSeek-R1. A key aspect of our
methodology was a self-feedback mechanism where LLMs generated, evaluated, and
then refined their outputs for query expansion and for multiple answer types
(yes/no, factoid, list, ideal). We investigated whether this iterative
self-correction improves performance and if reasoning models are more capable
of generating useful feedback. Preliminary results indicate varied performance
for the self-feedback strategy across models and tasks. This work offers
insights into LLM self-correction and informs future work on comparing the
effectiveness of LLM-generated feedback with direct human expert input in these
search systems.

### 9. [The TUB Sign Language Corpus Collection](http://arxiv.org/pdf/2508.05374v1)

Authors: Eleftherios Avramidis, Vera Czehmann, Fabian Deckert, Lorenz Hufe, Aljoscha Lipski, Yuni Amaloa Quintero Villalobos, Tae Kwon Rhee, Mengqian Shi, Lennart Stölting, Fabrizio Nunnari, Sebastian Möller

We present a collection of parallel corpora of 12 sign languages in video
format, together with subtitles in the dominant spoken languages of the
corresponding countries. The entire collection includes more than 1,300 hours
in 4,381 video files, accompanied by 1,3~M subtitles containing 14~M tokens.
Most notably, it includes the first consistent parallel corpora for 8 Latin
American sign languages, whereas the size of the German Sign Language corpora
is ten times the size of the previously available corpora. The collection was
created by collecting and processing videos of multiple sign languages from
various online sources, mainly broadcast material of news shows, governmental
bodies and educational channels. The preparation involved several stages,
including data collection, informing the content creators and seeking usage
approvals, scraping, and cropping. The paper provides statistics on the
collection and an overview of the methods used to collect the data.

### 10. [LLMEval-3: A Large-Scale Longitudinal Study on Robust and Fair Evaluation of Large Language Models](http://arxiv.org/pdf/2508.05452v1)

Authors: Ming Zhang, Yujiong Shen, Jingyi Deng, Yuhui Wang, Yue Zhang, Junzhe Wang, Shichun Liu, Shihan Dou, Huayu Sha, Qiyuan Peng, Changhao Jiang, Jingqi Tong, Yilong Wu, Zhihao Zhang, Mingqi Wu, Zhiheng Xi, Mingxu Chai, Tao Liang, Zhihui Fei, Zhen Wang, Mingyang Wan, Guojun Ma, Tao Gui, Qi Zhang, Xuanjing Huang

Existing evaluation of Large Language Models (LLMs) on static benchmarks is
vulnerable to data contamination and leaderboard overfitting, critical issues
that obscure true model capabilities. To address this, we introduce LLMEval-3,
a framework for dynamic evaluation of LLMs. LLMEval-3 is built on a proprietary
bank of 220k graduate-level questions, from which it dynamically samples unseen
test sets for each evaluation run. Its automated pipeline ensures integrity via
contamination-resistant data curation, a novel anti-cheating architecture, and
a calibrated LLM-as-a-judge process achieving 90% agreement with human experts,
complemented by a relative ranking system for fair comparison. An 20-month
longitudinal study of nearly 50 leading models reveals a performance ceiling on
knowledge memorization and exposes data contamination vulnerabilities
undetectable by static benchmarks. The framework demonstrates exceptional
robustness in ranking stability and consistency, providing strong empirical
validation for the dynamic evaluation paradigm. LLMEval-3 offers a robust and
credible methodology for assessing the true capabilities of LLMs beyond
leaderboard scores, promoting the development of more trustworthy evaluation
standards.

### Cryptography and Security

### 1. [On the Classical Hardness of the Semidirect Discrete Logarithm Problem in Finite Groups](http://arxiv.org/pdf/2508.05048v1)

Authors: Mohammad Ferry Husnil Arif, Muhammad Imran

The semidirect discrete logarithm problem (SDLP) in finite groups was
proposed as a foundation for post-quantum cryptographic protocols, based on the
belief that its non-abelian structure would resist quantum attacks. However,
recent results have shown that SDLP in finite groups admits efficient quantum
algorithms, undermining its quantum resistance. This raises a fundamental
question: does the SDLP offer any computational advantages over the standard
discrete logarithm problem (DLP) against classical adversaries? In this work,
we investigate the classical hardness of SDLP across different finite group
platforms. We establish that the group-case SDLP can be reformulated as a
generalized discrete logarithm problem, enabling adaptation of classical
algorithms to study its complexity. We present a concrete adaptation of the
Baby-Step Giant-Step algorithm for SDLP, achieving time and space complexity
$O(\sqrt{r})$ where $r$ is the period of the underlying cycle structure.
Through theoretical analysis and experimental validation in SageMath, we
demonstrate that the classical hardness of SDLP is highly platform-dependent
and does not uniformly exceed that of standard DLP. In finite fields
$\mathbb{F}_p^*$, both problems exhibit comparable complexity. Surprisingly, in
elliptic curves $E(\mathbb{F}_p)$, the SDLP becomes trivial due to the bounded
automorphism group, while in elementary abelian groups $\mathbb{F}_p^n$, the
SDLP can be harder than DLP, with complexity varying based on the eigenvalue
structure of the automorphism. Our findings reveal that the non-abelian
structure of semidirect products does not inherently guarantee increased
classical hardness, suggesting that the search for classically hard problems
for cryptographic applications requires more careful consideration of the
underlying algebraic structures.

### 2. [An Overview of 7726 User Reports: Uncovering SMS Scams and Scammer Strategies](http://arxiv.org/pdf/2508.05276v1)

Authors: Sharad Agarwal, Guillermo Suarez-Tangil, Marie Vasek

Mobile network operators implement firewalls to stop illicit messages, but
scammers find ways to evade detection. Previous work has looked into SMS texts
that are blocked by these firewalls. However, there is little insight into SMS
texts that bypass them and reach users. To this end, we collaborate with a
major mobile network operator to receive 1.35m user reports submitted over four
months. We find 89.16% of user reports comprise text messages, followed by
reports of suspicious calls and URLs. Using our methodological framework, we
identify 35.12% of the unique text messages reported by users as spam, while
40.27% are scam text messages. This is the first paper that investigates SMS
reports submitted by users and differentiates between spam and scams. Our paper
classifies the identified scam text messages into 12 scam types, of which the
most popular is 'wrong number' scams. We explore the various infrastructure
services that scammers abuse to conduct SMS scams, including mobile network
operators and hosting infrastructure, and analyze the text of the scam messages
to understand how scammers lure victims into providing them with their personal
or financial details.

### 3. [ShikkhaChain: A Blockchain-Powered Academic Credential Verification System for Bangladesh](http://arxiv.org/pdf/2508.05334v1)

Authors: Ahsan Farabi, Israt Khandaker, Nusrat Jahan, Ibrahim Khalil Shanto

Academic credential fraud threatens educational integrity, especially in
developing countries like Bangladesh, where verification methods are primarily
manual and inefficient. To address this challenge, we present ShikkhaChain, a
blockchain-powered certificate management platform designed to securely issue,
verify, and revoke academic credentials in a decentralized and tamper-proof
manner. Built on Ethereum smart contracts and utilizing IPFS for off-chain
storage, the platform offers a transparent, scalable solution accessible
through a React-based DApp with MetaMask integration. ShikkhaChain enables
role-based access for governments, regulators, institutions, and public
verifiers, allowing QR-based validation and on-chain revocation tracking. Our
prototype demonstrates enhanced trust, reduced verification time, and improved
international credibility for Bangladeshi degrees, promoting a more reliable
academic and employment ecosystem.

### 4. [Grouped k-threshold random grid-based visual cryptography scheme](http://arxiv.org/pdf/2508.05394v1)

Authors: Xiaoli Zhuo, Xuehu Yan, Wei Yan

Visual cryptography schemes (VCSs) belong to a category of secret image
sharing schemes that do not require cryptographic knowledge for decryption,
instead relying directly on the human visual system. Among VCSs, random
grid-based VCS (RGVCS) has garnered widespread attention as it avoids pixel
expansion while requiring no basic matrices design. Contrast, a core metric for
RGVCS, directly determines the visual quality of recovered images, rendering
its optimization a critical research objective. However, existing $(k,n)$
RGVCSs still fail to attain theoretical upper bounds on contrast, highlighting
the urgent need for higher-contrast constructions. In this paper, we propose a
novel sharing paradigm for RGVCS that constructs $(k,n)$-threshold schemes from
arbitrary $(k,n')$-threshold schemes $(k \leq n'\leq n)$, termed
\emph{$n'$-grouped $(k,n)$ RGVCS}. This paradigm establishes hierarchical
contrast characteristics: participants within the same group achieve optimal
recovery quality, while inter-group recovery shows a hierarchical contrast. We
further introduce a new contrast calculation formula tailored to the new
paradigm. Then, we propose a contrast-enhanced $(k,n)$ RGVCS by setting $n'=
k$, achieving the highest contrast value documented in the existing literature.
Theoretical analysis and experimental results demonstrate the superiority of
our proposed scheme in terms of contrast.

### 5. [Local Distance Query with Differential Privacy](http://arxiv.org/pdf/2508.05518v1)

Authors: Weihong Sheng, Jiajun Chen, Bin Cai, Chunqiang Hu, Meng Han, Jiguo Yu

Differential Privacy (DP) is commonly employed to safeguard graph analysis or
publishing. Distance, a critical factor in graph analysis, is typically handled
using curator DP, where a trusted curator holds the complete neighbor lists of
all vertices and answers queries privately. However, in many real-world
scenarios, such a curator may not be present, posing a significant challenge
for implementing differentially private distance queries under Local
Differential Privacy (LDP). This paper proposes two approaches to address this
challenge. The first approach generates a synthetic graph by randomizing
responses and applies bitwise operations to reduce noise interference. However,
like other synthetic graph methods, this approach suffers from low utility. To
overcome this limitation, we propose a second approach, the first LDP method
specifically designed for distance queries, which captures the global graph
structure by continuously aggregating local distance vectors from neighboring
vertices. This process enables the accurate updating of global distances. We
demonstrate the effectiveness of our method through comprehensive theoretical
analysis and experimental evaluations on real-world datasets.

### 6. [PRvL: Quantifying the Capabilities and Risks of Large Language Models for PII Redaction](http://arxiv.org/pdf/2508.05545v1)

Authors: Leon Garza, Anantaa Kotal, Aritran Piplai, Lavanya Elluri, Prajit Das, Aman Chadha

Redacting Personally Identifiable Information (PII) from unstructured text is
critical for ensuring data privacy in regulated domains. While earlier
approaches have relied on rule-based systems and domain-specific Named Entity
Recognition (NER) models, these methods fail to generalize across formats and
contexts. Recent advances in Large Language Models (LLMs) offer a promising
alternative, yet the effect of architectural and training choices on redaction
performance remains underexplored. LLMs have demonstrated strong performance in
tasks that require contextual language understanding, including the redaction
of PII in free-form text. Prior work suggests that with appropriate adaptation,
LLMs can become effective contextual privacy learners. However, the
consequences of architectural and training choices for PII Redaction remain
underexplored. In this work, we present a comprehensive analysis of LLMs as
privacy-preserving PII Redaction systems. We evaluate a range of LLM
architectures and training strategies for their effectiveness in PII Redaction.
Our analysis measures redaction performance, semantic preservation, and PII
leakage, and compares these outcomes against latency and computational cost.
The results provide practical guidance for configuring LLM-based redactors that
are accurate, efficient, and privacy-aware. To support reproducibility and
real-world deployment, we release PRvL, an open-source suite of fine-tuned
models, and evaluation tools for general-purpose PII Redaction. PRvL is built
entirely on open-source LLMs and supports multiple inference settings for
flexibility and compliance. It is designed to be easily customized for
different domains and fully operable within secure, self-managed environments.
This enables data owners to perform redactions without relying on third-party
services or exposing sensitive content beyond their own infrastructure.

### 7. [Incident Response Planning Using a Lightweight Large Language Model with Reduced Hallucination](http://arxiv.org/pdf/2508.05188v1)

Authors: Kim Hammar, Tansu Alpcan, Emil C. Lupu

Timely and effective incident response is key to managing the growing
frequency of cyberattacks. However, identifying the right response actions for
complex systems is a major technical challenge. A promising approach to
mitigate this challenge is to use the security knowledge embedded in large
language models (LLMs) to assist security operators during incident handling.
Recent research has demonstrated the potential of this approach, but current
methods are mainly based on prompt engineering of frontier LLMs, which is
costly and prone to hallucinations. We address these limitations by presenting
a novel way to use an LLM for incident response planning with reduced
hallucination. Our method includes three steps: fine-tuning, information
retrieval, and lookahead planning. We prove that our method generates response
plans with a bounded probability of hallucination and that this probability can
be made arbitrarily small at the expense of increased planning time under
certain assumptions. Moreover, we show that our method is lightweight and can
run on commodity hardware. We evaluate our method on logs from incidents
reported in the literature. The experimental results show that our method a)
achieves up to 22% shorter recovery times than frontier LLMs and b) generalizes
to a broad range of incident types and response actions.

### 8. [Secure and practical Quantum Digital Signatures](http://arxiv.org/pdf/2508.05355v1)

Authors: Federico Grasselli, Gaetano Russo, Massimiliano Proietti

Digital signatures represent a crucial cryptographic asset that must be
protected against quantum adversaries. Quantum Digital Signatures (QDS) can
offer solutions that are information-theoretically (IT) secure and thus immune
to quantum attacks. In this work, we analyze three existing practical QDS
protocols based on preshared secure keys (e.g., established with quantum key
distribution) and universal hashing families. For each protocol, we make
amendments to close potential loopholes and prove their IT security while
accounting for the failure of IT-secure authenticated communication. We then
numerically optimize the protocol parameters to improve efficiency in terms of
preshared bit consumption and signature length, allowing us to identify the
most efficient protocol.

### 9. [Optimizing IoT Threat Detection with Kolmogorov-Arnold Networks (KANs)](http://arxiv.org/pdf/2508.05591v1)

Authors: Natalia Emelianova, Carlos Kamienski, Ronaldo C. Prati

The exponential growth of the Internet of Things (IoT) has led to the
emergence of substantial security concerns, with IoT networks becoming the
primary target for cyberattacks. This study examines the potential of
Kolmogorov-Arnold Networks (KANs) as an alternative to conventional machine
learning models for intrusion detection in IoT networks. The study demonstrates
that KANs, which employ learnable activation functions, outperform traditional
MLPs and achieve competitive accuracy compared to state-of-the-art models such
as Random Forest and XGBoost, while offering superior interpretability for
intrusion detection in IoT networks.

### 10. [Non-omniscient backdoor injection with a single poison sample: Proving the one-poison hypothesis for linear regression and linear classification](http://arxiv.org/pdf/2508.05600v1)

Authors: Thorsten Peinemann, Paula Arnold, Sebastian Berndt, Thomas Eisenbarth, Esfandiar Mohammadi

Backdoor injection attacks are a threat to machine learning models that are
trained on large data collected from untrusted sources; these attacks enable
attackers to inject malicious behavior into the model that can be triggered by
specially crafted inputs. Prior work has established bounds on the success of
backdoor attacks and their impact on the benign learning task, however, an open
question is what amount of poison data is needed for a successful backdoor
attack. Typical attacks either use few samples, but need much information about
the data points or need to poison many data points.
  In this paper, we formulate the one-poison hypothesis: An adversary with one
poison sample and limited background knowledge can inject a backdoor with zero
backdooring-error and without significantly impacting the benign learning task
performance. Moreover, we prove the one-poison hypothesis for linear regression
and linear classification. For adversaries that utilize a direction that is
unused by the benign data distribution for the poison sample, we show that the
resulting model is functionally equivalent to a model where the poison was
excluded from training. We build on prior work on statistical backdoor learning
to show that in all other cases, the impact on the benign learning task is
still limited. We also validate our theoretical results experimentally with
realistic benchmark data sets.

### Computer Vision and Pattern Recognition

### 1. [Accelerating Conditional Prompt Learning via Masked Image Modeling for Vision-Language Models](http://arxiv.org/pdf/2508.04942v1)

Authors: Phuoc-Nguyen Bui, Khanh-Binh Nguyen, Hyunseung Choo

Vision-language models (VLMs) like CLIP excel in zero-shot learning but often
require resource-intensive training to adapt to new tasks. Prompt learning
techniques, such as CoOp and CoCoOp, offer efficient adaptation but tend to
overfit to known classes, limiting generalization to unseen categories. We
introduce ProMIM, a plug-and-play framework that enhances conditional prompt
learning by integrating masked image modeling (MIM) into existing VLM
pipelines. ProMIM leverages a simple yet effective masking strategy to generate
robust, instance-conditioned prompts, seamlessly augmenting methods like CoOp
and CoCoOp without altering their core architectures. By masking only visible
image patches and using these representations to guide prompt generation,
ProMIM improves feature robustness and mitigates overfitting, all while
introducing negligible additional computational cost. Extensive experiments
across zero-shot and few-shot classification tasks demonstrate that ProMIM
consistently boosts generalization performance when plugged into existing
approaches, providing a practical, lightweight solution for real-world
vision-language applications.

### 2. [CSRAP: Enhanced Canvas Attention Scheduling for Real-Time Mission Critical Perception](http://arxiv.org/pdf/2508.04976v1)

Authors: Md Iftekharul Islam Sakib, Yigong Hu, Tarek Abdelzaher

Real-time perception on edge platforms faces a core challenge: executing
high-resolution object detection under stringent latency constraints on limited
computing resources. Canvas-based attention scheduling was proposed in earlier
work as a mechanism to reduce the resource demands of perception subsystems. It
consolidates areas of interest in an input data frame onto a smaller area,
called a canvas frame, that can be processed at the requisite frame rate. This
paper extends prior canvas-based attention scheduling literature by (i)
allowing for variable-size canvas frames and (ii) employing selectable canvas
frame rates that may depart from the original data frame rate. We evaluate our
solution by running YOLOv11, as the perception module, on an NVIDIA Jetson Orin
Nano to inspect video frames from the Waymo Open Dataset. Our results show that
the additional degrees of freedom improve the attainable quality/cost
trade-offs, thereby allowing for a consistently higher mean average precision
(mAP) and recall with respect to the state of the art.

### 3. [Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression](http://arxiv.org/pdf/2508.04979v1)

Authors: Zheng Chen, Mingde Zhou, Jinpei Guo, Jiale Yuan, Yifei Ji, Yulun Zhang

Diffusion-based image compression has demonstrated impressive perceptual
performance. However, it suffers from two critical drawbacks: (1) excessive
decoding latency due to multi-step sampling, and (2) poor fidelity resulting
from over-reliance on generative priors. To address these issues, we propose
SODEC, a novel single-step diffusion image compression model. We argue that in
image compression, a sufficiently informative latent renders multi-step
refinement unnecessary. Based on this insight, we leverage a pre-trained
VAE-based model to produce latents with rich information, and replace the
iterative denoising process with a single-step decoding. Meanwhile, to improve
fidelity, we introduce the fidelity guidance module, encouraging output that is
faithful to the original image. Furthermore, we design the rate annealing
training strategy to enable effective training under extremely low bitrates.
Extensive experiments show that SODEC significantly outperforms existing
methods, achieving superior rate-distortion-perception performance. Moreover,
compared to previous diffusion-based compression models, SODEC improves
decoding speed by more than 20$\times$. Code is released at:
https://github.com/zhengchen1999/SODEC.

### 4. [Propagating Sparse Depth via Depth Foundation Model for Out-of-Distribution Depth Completion](http://arxiv.org/pdf/2508.04984v1)

Authors: Shenglun Chen, Xinzhu Ma, Hong Zhang, Haojie Li, Zhihui Wang

Depth completion is a pivotal challenge in computer vision, aiming at
reconstructing the dense depth map from a sparse one, typically with a paired
RGB image. Existing learning based models rely on carefully prepared but
limited data, leading to significant performance degradation in
out-of-distribution (OOD) scenarios. Recent foundation models have demonstrated
exceptional robustness in monocular depth estimation through large-scale
training, and using such models to enhance the robustness of depth completion
models is a promising solution. In this work, we propose a novel depth
completion framework that leverages depth foundation models to attain
remarkable robustness without large-scale training. Specifically, we leverage a
depth foundation model to extract environmental cues, including structural and
semantic context, from RGB images to guide the propagation of sparse depth
information into missing regions. We further design a dual-space propagation
approach, without any learnable parameters, to effectively propagates sparse
depth in both 3D and 2D spaces to maintain geometric structure and local
consistency. To refine the intricate structure, we introduce a learnable
correction module to progressively adjust the depth prediction towards the real
depth. We train our model on the NYUv2 and KITTI datasets as in-distribution
datasets and extensively evaluate the framework on 16 other datasets. Our
framework performs remarkably well in the OOD scenarios and outperforms
existing state-of-the-art depth completion methods. Our models are released in
https://github.com/shenglunch/PSD.

### 5. [Unified modality separation: A vision-language framework for unsupervised domain adaptation](http://arxiv.org/pdf/2508.04987v1)

Authors: Xinyao Li, Jingjing Li, Zhekai Du, Lei Zhu, Heng Tao Shen

Unsupervised domain adaptation (UDA) enables models trained on a labeled
source domain to handle new unlabeled domains. Recently, pre-trained
vision-language models (VLMs) have demonstrated promising zero-shot performance
by leveraging semantic information to facilitate target tasks. By aligning
vision and text embeddings, VLMs have shown notable success in bridging domain
gaps. However, inherent differences naturally exist between modalities, which
is known as modality gap. Our findings reveal that direct UDA with the presence
of modality gap only transfers modality-invariant knowledge, leading to
suboptimal target performance. To address this limitation, we propose a unified
modality separation framework that accommodates both modality-specific and
modality-invariant components. During training, different modality components
are disentangled from VLM features then handled separately in a unified manner.
At test time, modality-adaptive ensemble weights are automatically determined
to maximize the synergy of different components. To evaluate instance-level
modality characteristics, we design a modality discrepancy metric to categorize
samples into modality-invariant, modality-specific, and uncertain ones. The
modality-invariant samples are exploited to facilitate cross-modal alignment,
while uncertain ones are annotated to enhance model capabilities. Building upon
prompt tuning techniques, our methods achieve up to 9% performance gain with 9
times of computational efficiencies. Extensive experiments and analysis across
various backbones, baselines, datasets and adaptation settings demonstrate the
efficacy of our design.

### 6. [Modeling Rapid Contextual Learning in the Visual Cortex with Fast-Weight Deep Autoencoder Networks](http://arxiv.org/pdf/2508.04988v1)

Authors: Yue Li, Weifan Wang, Tai Sing Lee

Recent neurophysiological studies have revealed that the early visual cortex
can rapidly learn global image context, as evidenced by a sparsification of
population responses and a reduction in mean activity when exposed to familiar
versus novel image contexts. This phenomenon has been attributed primarily to
local recurrent interactions, rather than changes in feedforward or feedback
pathways, supported by both empirical findings and circuit-level modeling.
Recurrent neural circuits capable of simulating these effects have been shown
to reshape the geometry of neural manifolds, enhancing robustness and
invariance to irrelevant variations. In this study, we employ a Vision
Transformer (ViT)-based autoencoder to investigate, from a functional
perspective, how familiarity training can induce sensitivity to global context
in the early layers of a deep neural network. We hypothesize that rapid
learning operates via fast weights, which encode transient or short-term memory
traces, and we explore the use of Low-Rank Adaptation (LoRA) to implement such
fast weights within each Transformer layer. Our results show that (1) The
proposed ViT-based autoencoder's self-attention circuit performs a manifold
transform similar to a neural circuit model of the familiarity effect. (2)
Familiarity training aligns latent representations in early layers with those
in the top layer that contains global context information. (3) Familiarity
training broadens the self-attention scope within the remembered image context.
(4) These effects are significantly amplified by LoRA-based fast weights.
Together, these findings suggest that familiarity training introduces global
sensitivity to earlier layers in a hierarchical network, and that a hybrid
fast-and-slow weight architecture may provide a viable computational model for
studying rapid global context learning in the brain.

### 7. [Attribute Guidance With Inherent Pseudo-label For Occluded Person Re-identification](http://arxiv.org/pdf/2508.04998v1)

Authors: Rui Zhi, Zhen Yang, Haiyang Zhang

Person re-identification (Re-ID) aims to match person images across different
camera views, with occluded Re-ID addressing scenarios where pedestrians are
partially visible. While pre-trained vision-language models have shown
effectiveness in Re-ID tasks, they face significant challenges in occluded
scenarios by focusing on holistic image semantics while neglecting fine-grained
attribute information. This limitation becomes particularly evident when
dealing with partially occluded pedestrians or when distinguishing between
individuals with subtle appearance differences. To address this limitation, we
propose Attribute-Guide ReID (AG-ReID), a novel framework that leverages
pre-trained models' inherent capabilities to extract fine-grained semantic
attributes without additional data or annotations. Our framework operates
through a two-stage process: first generating attribute pseudo-labels that
capture subtle visual characteristics, then introducing a dual-guidance
mechanism that combines holistic and fine-grained attribute information to
enhance image feature extraction. Extensive experiments demonstrate that
AG-ReID achieves state-of-the-art results on multiple widely-used Re-ID
datasets, showing significant improvements in handling occlusions and subtle
attribute differences while maintaining competitive performance on standard
Re-ID scenarios.

### 8. [Multimodal Causal-Driven Representation Learning for Generalizable Medical Image Segmentation](http://arxiv.org/pdf/2508.05008v1)

Authors: Xusheng Liang, Lihua Zhou, Nianxin Li, Miao Xu, Ziyang Song, Dong Yi, Jinlin Wu, Hongbin Liu, Jiebo Luo, Zhen Lei

Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable
zero-shot capabilities in various computer vision tasks. However, their
application to medical imaging remains challenging due to the high variability
and complexity of medical data. Specifically, medical images often exhibit
significant domain shifts caused by various confounders, including equipment
differences, procedure artifacts, and imaging modes, which can lead to poor
generalization when models are applied to unseen domains. To address this
limitation, we propose Multimodal Causal-Driven Representation Learning
(MCDRL), a novel framework that integrates causal inference with the VLM to
tackle domain generalization in medical image segmentation. MCDRL is
implemented in two steps: first, it leverages CLIP's cross-modal capabilities
to identify candidate lesion regions and construct a confounder dictionary
through text prompts, specifically designed to represent domain-specific
variations; second, it trains a causal intervention network that utilizes this
dictionary to identify and eliminate the influence of these domain-specific
variations while preserving the anatomical structural information critical for
segmentation tasks. Extensive experiments demonstrate that MCDRL consistently
outperforms competing methods, yielding superior segmentation accuracy and
exhibiting robust generalizability.

### 9. [HAMoBE: Hierarchical and Adaptive Mixture of Biometric Experts for Video-based Person ReID](http://arxiv.org/pdf/2508.05038v1)

Authors: Yiyang Su, Yunping Shi, Feng Liu, Xiaoming Liu

Recently, research interest in person re-identification (ReID) has
increasingly focused on video-based scenarios, which are essential for robust
surveillance and security in varied and dynamic environments. However, existing
video-based ReID methods often overlook the necessity of identifying and
selecting the most discriminative features from both videos in a query-gallery
pair for effective matching. To address this issue, we propose a novel
Hierarchical and Adaptive Mixture of Biometric Experts (HAMoBE) framework,
which leverages multi-layer features from a pre-trained large model (e.g.,
CLIP) and is designed to mimic human perceptual mechanisms by independently
modeling key biometric features--appearance, static body shape, and dynamic
gait--and adaptively integrating them. Specifically, HAMoBE includes two
levels: the first level extracts low-level features from multi-layer
representations provided by the frozen large model, while the second level
consists of specialized experts focusing on long-term, short-term, and temporal
features. To ensure robust matching, we introduce a new dual-input decision
gating network that dynamically adjusts the contributions of each expert based
on their relevance to the input scenarios. Extensive evaluations on benchmarks
like MEVID demonstrate that our approach yields significant performance
improvements (e.g., +13.0% Rank-1 accuracy).

### 10. [Finding Needles in Images: Can Multimodal LLMs Locate Fine Details?](http://arxiv.org/pdf/2508.05053v1)

Authors: Parth Thakkar, Ankush Agarwal, Prasad Kasu, Pulkit Bansal, Chaitanya Devaguptapu

While Multi-modal Large Language Models (MLLMs) have shown impressive
capabilities in document understanding tasks, their ability to locate and
reason about fine-grained details within complex documents remains
understudied. Consider searching a restaurant menu for a specific nutritional
detail or identifying a disclaimer in a lengthy newspaper article tasks that
demand careful attention to small but significant details within a broader
narrative, akin to Finding Needles in Images (NiM). To address this gap, we
introduce NiM, a carefully curated benchmark spanning diverse real-world
documents including newspapers, menus, and lecture images, specifically
designed to evaluate MLLMs' capability in these intricate tasks. Building on
this, we further propose Spot-IT, a simple yet effective approach that enhances
MLLMs capability through intelligent patch selection and Gaussian attention,
motivated from how humans zoom and focus when searching documents. Our
extensive experiments reveal both the capabilities and limitations of current
MLLMs in handling fine-grained document understanding tasks, while
demonstrating the effectiveness of our approach. Spot-IT achieves significant
improvements over baseline methods, particularly in scenarios requiring precise
detail extraction from complex layouts.

### Computers and Society

### 1. [Resistance Technologies: Moving Beyond Alternative Designs](http://arxiv.org/pdf/2508.05223v1)

Authors: Iness Ben Guirat, Jan Tobias Mühlberg

The discourse about sustainable technology has emerged from the
acknowledgment of the environmental collapse we are facing. In this paper, we
argue that addressing this crisis requires more than the development of
sustainable alternatives to current online services or the optimization of
resources using various dashboards and AI. Rather, the focus must shift toward
designing technologies that protect us from the consequences of the
environmental damages. Among these consequences, wars, genocide and new forms
of colonialism are perhaps the most significant. We identify "protection" not
in terms of military defense as Western States like to argue, but as part of
sovereignty. We seek to define the term of "Resistance Technologies" for such
technologies, arguing further that anti-surveillance technologies are a
foundational component of sovereignty and must be part of future conversations
around sustainability. Finally, our paper seeks to open a discourse with the
Computing-within-Limits community and beyond, towards defining other essential
aspects or concepts of technologies that we see as core values of "Resistance
Technology".

### 2. [A Conceptual Model and Methodology for Sustainability-aware, IoT-enhanced Business Processes](http://arxiv.org/pdf/2508.05301v1)

Authors: Victoria Torres Bosch, Ronny Seiger, Manuela Albert Albiol, Antoni Mestre Gascon, Pedro Jose Valderas Aranda

The real-time data collection and automation capabilities offered by the
Internet of Things (IoT) are revolutionizing and transforming Business
Processes (BPs) into IoT-enhanced BPs, showing high potential for improving
sustainability. Although already studied in Business Process Management (BPM),
sustainability research has primarily focused on environmental concerns.
However, achieving a holistic and lasting impact requires a systematic approach
to address sustainability beyond the environmental dimension. This work
proposes a conceptual model and a structured methodology with the goal of
analyzing the potential of IoT to measure and improve the sustainability of
BPs. The conceptual model formally represents key sustainability concepts,
linking BPM and IoT by highlighting how IoT devices support and contribute to
sustainability. The methodology guides the systematic analysis of existing BPs,
identifies opportunities, and implements sustainability-aware, IoT-enhanced
BPs. The approach is illustrated through a running example from the tourism
domain and a case study in healthcare.

### 3. [The Term 'Agent' Has Been Diluted Beyond Utility and Requires Redefinition](http://arxiv.org/pdf/2508.05338v1)

Authors: Brinnae Bent

The term 'agent' in artificial intelligence has long carried multiple
interpretations across different subfields. Recent developments in AI
capabilities, particularly in large language model systems, have amplified this
ambiguity, creating significant challenges in research communication, system
evaluation and reproducibility, and policy development. This paper argues that
the term 'agent' requires redefinition. Drawing from historical analysis and
contemporary usage patterns, we propose a framework that defines clear minimum
requirements for a system to be considered an agent while characterizing
systems along a multidimensional spectrum of environmental interaction,
learning and adaptation, autonomy, goal complexity, and temporal coherence.
This approach provides precise vocabulary for system description while
preserving the term's historically multifaceted nature. After examining
potential counterarguments and implementation challenges, we provide specific
recommendations for moving forward as a field, including suggestions for
terminology standardization and framework adoption. The proposed approach
offers practical tools for improving research clarity and reproducibility while
supporting more effective policy development.

### 4. [Building Effective Safety Guardrails in AI Education Tools](http://arxiv.org/pdf/2508.05360v1)

Authors: Hannah-Beth Clark, Laura Benton, Emma Searle, Margaux Dowland, Matthew Gregory, Will Gayne, John Roberts

There has been rapid development in generative AI tools across the education
sector, which in turn is leading to increased adoption by teachers. However,
this raises concerns regarding the safety and age-appropriateness of the
AI-generated content that is being created for use in classrooms. This paper
explores Oak National Academy's approach to addressing these concerns within
the development of the UK Government's first publicly available generative AI
tool - our AI-powered lesson planning assistant (Aila). Aila is intended to
support teachers planning national curriculum-aligned lessons that are
appropriate for pupils aged 5-16 years. To mitigate safety risks associated
with AI-generated content we have implemented four key safety guardrails - (1)
prompt engineering to ensure AI outputs are generated within pedagogically
sound and curriculum-aligned parameters, (2) input threat detection to mitigate
attacks, (3) an Independent Asynchronous Content Moderation Agent (IACMA) to
assess outputs against predefined safety categories, and (4) taking a
human-in-the-loop approach, to encourage teachers to review generated content
before it is used in the classroom. Through our on-going evaluation of these
safety guardrails we have identified several challenges and opportunities to
take into account when implementing and testing safety guardrails. This paper
highlights ways to build more effective safety guardrails in generative AI
education tools including the on-going iteration and refinement of guardrails,
as well as enabling cross-sector collaboration through sharing both open-source
code, datasets and learnings.

### 5. [Whose Truth? Pluralistic Geo-Alignment for (Agentic) AI](http://arxiv.org/pdf/2508.05432v1)

Authors: Krzysztof Janowicz, Zilong Liu, Gengchen Mai, Zhangyu Wang, Ivan Majic, Alexandra Fortacz, Grant McKenzie, Song Gao

AI (super) alignment describes the challenge of ensuring (future) AI systems
behave in accordance with societal norms and goals. While a quickly evolving
literature is addressing biases and inequalities, the geographic variability of
alignment remains underexplored. Simply put, what is considered appropriate,
truthful, or legal can differ widely across regions due to cultural norms,
political realities, and legislation. Alignment measures applied to AI/ML
workflows can sometimes produce outcomes that diverge from statistical
realities, such as text-to-image models depicting balanced gender ratios in
company leadership despite existing imbalances. Crucially, some model outputs
are globally acceptable, while others, e.g., questions about Kashmir, depend on
knowing the user's location and their context. This geographic sensitivity is
not new. For instance, Google Maps renders Kashmir's borders differently based
on user location. What is new is the unprecedented scale and automation with
which AI now mediates knowledge, expresses opinions, and represents geographic
reality to millions of users worldwide, often with little transparency about
how context is managed. As we approach Agentic AI, the need for
spatio-temporally aware alignment, rather than one-size-fits-all approaches, is
increasingly urgent. This paper reviews key geographic research problems,
suggests topics for future work, and outlines methods for assessing alignment
sensitivity.

### 6. [Do Political Opinions Transfer Between Western Languages? An Analysis of Unaligned and Aligned Multilingual LLMs](http://arxiv.org/pdf/2508.05553v1)

Authors: Franziska Weeber, Tanise Ceron, Sebastian Padó

Public opinion surveys show cross-cultural differences in political opinions
between socio-cultural contexts. However, there is no clear evidence whether
these differences translate to cross-lingual differences in multilingual large
language models (MLLMs). We analyze whether opinions transfer between languages
or whether there are separate opinions for each language in MLLMs of various
sizes across five Western languages. We evaluate MLLMs' opinions by prompting
them to report their (dis)agreement with political statements from voting
advice applications. To better understand the interaction between languages in
the models, we evaluate them both before and after aligning them with more left
or right views using direct preference optimization and English alignment data
only. Our findings reveal that unaligned models show only very few significant
cross-lingual differences in the political opinions they reflect. The political
alignment shifts opinions almost uniformly across all five languages. We
conclude that in Western language contexts, political opinions transfer between
languages, demonstrating the challenges in achieving explicit socio-linguistic,
cultural, and political alignment of MLLMs.

### 7. [Everything You Need to Know About CS Education: Open Results from a Survey of More Than 18,000 Participants](http://arxiv.org/pdf/2508.05286v1)

Authors: Katsiaryna Dzialets, Aleksandra Makeeva, Ilya Vlasov, Anna Potriasaeva, Aleksei Rostovskii, Yaroslav Golubev, Anastasiia Birillo

Computer science education is a dynamic field with many aspects that
influence the learner's path. While these aspects are usually studied in depth
separately, it is also important to carry out broader large-scale studies that
touch on many topics, because they allow us to put different results into each
other's perspective. Past large-scale surveys have provided valuable insights,
however, the emergence of new trends (e.g., AI), new learning formats (e.g.,
in-IDE learning), and the increasing learner diversity highlight the need for
an updated comprehensive study. To address this, we conducted a survey with
18,032 learners from 173 countries, ensuring diverse representation and
exploring a wide range of topics - formal education, learning formats, AI
usage, challenges, motivation, and more. This paper introduces the results of
this survey as an open dataset, describes our methodology and the survey
questions, and highlights, as a motivating example, three possible research
directions within this data: challenges in learning, emerging formats, and
insights into the in-IDE format. The dataset aims to support further research
and foster advancements in computer education.

### Databases

### 1. [AgenticData: An Agentic Data Analytics System for Heterogeneous Data](http://arxiv.org/pdf/2508.05002v1)

Authors: Ji Sun, Guoliang Li, Peiyao Zhou, Yihui Ma, Jingzhe Xu, Yuan Li

Existing unstructured data analytics systems rely on experts to write code
and manage complex analysis workflows, making them both expensive and
time-consuming. To address these challenges, we introduce AgenticData, an
innovative agentic data analytics system that allows users to simply pose
natural language (NL) questions while autonomously analyzing data sources
across multiple domains, including both unstructured and structured data.
First, AgenticData employs a feedback-driven planning technique that
automatically converts an NL query into a semantic plan composed of relational
and semantic operators. We propose a multi-agent collaboration strategy by
utilizing a data profiling agent for discovering relevant data, a semantic
cross-validation agent for iterative optimization based on feedback, and a
smart memory agent for maintaining short-term context and long-term knowledge.
Second, we propose a semantic optimization model to refine and execute semantic
plans effectively. Our system, AgenticData, has been tested using three
benchmarks. Experimental results showed that AgenticData achieved superior
accuracy on both easy and difficult tasks, significantly outperforming
state-of-the-art methods.

### 2. [Theseus: A Distributed and Scalable GPU-Accelerated Query Processing Platform Optimized for Efficient Data Movement](http://arxiv.org/pdf/2508.05029v1)

Authors: Felipe Aramburú, William Malpica, Kaouther Abrougui, Amin Aramoon, Romulo Auccapuclla, Claude Brisson, Matthijs Brobbel, Colby Farrell, Pradeep Garigipati, Joost Hoozemans, Supun Kamburugamuve, Akhil Nair, Alexander Ocsa, Johan Peltenburg, Rubén Quesada López, Deepak Sihag, Ahmet Uyar, Dhruv Vats, Michael Wendt, Jignesh M. Patel, Rodrigo Aramburú

Online analytical processing of queries on datasets in the many-terabyte
range is only possible with costly distributed computing systems. To decrease
the cost and increase the throughput, systems can leverage accelerators such as
GPUs, which are now ubiquitous in the compute infrastructure. This introduces
many challenges, the majority of which are related to when, where, and how to
best move data around the system. We present Theseus -- a production-ready
enterprise-scale distributed accelerator-native query engine designed to
balance data movement, memory utilization, and computation in an
accelerator-based system context. Specialized asynchronous control mechanisms
are tightly coupled to the hardware resources for the purpose of network
communication, data pre-loading, data spilling across memories and storage, and
GPU compute tasks. The memory subsystem contains a mechanism for fixed-size
page-locked host memory allocations to increase throughput and reduce memory
fragmentation. For the TPC-H benchmarks at scale factors ranging from 1k to 30k
on cloud infrastructure, Theseus outperforms Databricks Photon by up to
$4\times$ at cost parity. Theseus is capable of processing all queries of the
TPC-H and TPC-DS benchmarks at scale factor 100k (100 TB scale) with as few as
2 DGX A100 640GB nodes.

### 3. [Data-Aware Socratic Query Refinement in Database Systems](http://arxiv.org/pdf/2508.05061v1)

Authors: Ruiyuan Zhang, Chrysanthi Kosyfaki, Xiaofang Zhou

In this paper, we propose Data-Aware Socratic Guidance (DASG), a
dialogue-based query enhancement framework that embeds \linebreak interactive
clarification as a first-class operator within database systems to resolve
ambiguity in natural language queries. DASG treats dialogue as an optimization
decision, asking clarifying questions only when the expected execution cost
reduction exceeds the interaction overhead. The system quantifies ambiguity
through linguistic fuzziness, schema grounding confidence, and projected costs
across relational and vector backends. Our algorithm selects the optimal
clarifications by combining semantic relevance, catalog-based information gain,
and potential cost reduction. We evaluate our proposed framework on three
datasets. The results show that DASG demonstrates improved query precision
while maintaining efficiency, establishing a cooperative analytics paradigm
where systems actively participate in query formulation rather than passively
translating user requests.

### 4. [Making Prompts First-Class Citizens for Adaptive LLM Pipelines](http://arxiv.org/pdf/2508.05012v1)

Authors: Ugur Cetintemel, Shu Chen, Alexander W. Lee, Deepti Raghavan

Modern LLM pipelines increasingly resemble data-centric systems: they
retrieve external context, compose intermediate outputs, validate results, and
adapt based on runtime feedback. Yet, the central element guiding this process
-- the prompt -- remains a brittle, opaque string, disconnected from the
surrounding dataflow. This disconnect limits reuse, optimization, and runtime
control.
  In this paper, we describe our vision and an initial design for SPEAR, a
language and runtime that fills this prompt management gap by making prompts
structured, adaptive, and first-class components of the execution model. SPEAR
enables (1) runtime prompt refinement -- modifying prompts dynamically in
response to execution-time signals such as confidence, latency, or missing
context; and (2) structured prompt management -- organizing prompt fragments
into versioned views with support for introspection and logging.
  SPEAR defines a prompt algebra that governs how prompts are constructed and
adapted within a pipeline. It supports multiple refinement modes (manual,
assisted, and automatic), giving developers a balance between control and
automation. By treating prompt logic as structured data, SPEAR enables
optimizations such as operator fusion, prefix caching, and view reuse.
Preliminary experiments quantify the behavior of different refinement modes
compared to static prompts and agentic retries, as well as the impact of
prompt-level optimizations such as operator fusion.

### Distributed, Parallel, and Cluster Computing

### 1. [Managing, Analyzing and Sharing Research Data with Gen3 Data Commons](http://arxiv.org/pdf/2508.04944v1)

Authors: Craig Barnes, Kyle Burton, Michael S. Fitzsimons, Hara Prasad Juvvala, Brienna Larrick, Christopher Meyer, Pauline Ribeyre, Ao Liu, Clint Malson, Noah Metoki-Shlubsky, Andrii Prokhorenkov, Jawad Qureshi, Radhika Reddy, L. Philip Schumm, Mingfei Shao, Trevar Simmons, Alexander VanTol, Peter Vassilatos, Aarti Venkat, Robert L. Grossman

Gen3 is an open-source data platform for building data commons. A data
commons is a cloud-based data platform for managing, analyzing, and sharing
data with a research community. Gen3 has been used to build over a dozen data
commons that in aggregate contain over 28 PB of data and 64 million FAIR data
objects. To set up a Gen3 data commons, you first define a data model. Gen3
then autogenerates 1) a data portal for searching and exploring data in the
commons; 2) a data portal for submitting data to the commons; and 3) FAIR APIs
for accessing the data programmatically. Gen3 is built over a small number of
standards-based software services, which are designed to support current and
future Gen3 components so that Gen3 can interoperate with other data platforms
and data ecosystems.

### 2. [Simulating LLM training workloads for heterogeneous compute and network infrastructure](http://arxiv.org/pdf/2508.05370v1)

Authors: Sumit Kumar, Arjun Temura, Naman Sharma, Ramanjeet Singh, Meet Dadhania, Praveen Tammana, Satananda Burla, Abed Mohammad Kamaluddin, Rinku Shah

The growing demand for large-scale GPU clusters in distributed model training
presents a significant barrier to innovation, particularly in model
optimization, performance tuning, and system-level enhancements. To address
this challenge, LLM training simulators are employed to estimate training time
and guide design decisions. However, the state-of-the-art LLM training
simulators assume homogeneous compute and network infrastructure. In practice,
device heterogeneity is inevitable due to resource sharing in cloud
environments, frequent shifts in device generations, and inherent intra-chip
interconnect heterogeneity. To address the gap between state-of-the-art and
practical requirements, we propose the design of a heterogeneity-aware
distributed LLM simulator capable of predicting training time while enabling
abstractions to specify custom configurations for device groups and
device-to-parallelism mapping. We present the design requirements and
challenges in building a heterogeneity-aware distributed ML training simulator,
and design components such as non-uniform workload partitioning. Our initial
simulation results demonstrate the impact of heterogeneity on the model
computation and communication time.

### 3. [Adaptive Parallel Downloader for Large Genomic Datasets](http://arxiv.org/pdf/2508.05511v1)

Authors: Rasman Mubtasim Swargo, Engin Arslan, Md Arifuzzaman

Modern next-generation sequencing (NGS) projects routinely generate terabytes
of data, which researchers commonly download from public repositories such as
SRA or ENA. Existing download tools often employ static concurrency settings,
leading to inefficient bandwidth utilization and prolonged download times due
to their inability to adapt to dynamic network conditions. We introduce
FastBioDL, a parallel file downloader designed for large biological datasets,
featuring an adaptive concurrency controller. FastBioDL frames the download
process as an online optimization problem, utilizing a utility function and
gradient descent to adjust the number of concurrent socket streams in real-time
dynamically. This approach maximizes download throughput while minimizing
resource overhead. Comprehensive evaluations on public genomic datasets
demonstrate that FastBioDL achieves up to $4x$ speedup over state-of-the-art
tools. Moreover, in high-speed network experiments, its adaptive design was up
to $2.1x$ faster than existing tools. By intelligently optimizing standard HTTP
or FTP downloads on the client side, FastBioDL provides a robust and efficient
solution for large-scale genomic data acquisition, democratizing
high-performance data retrieval for researchers without requiring specialized
commercial software or protocols.

### 4. [Modular Architecture for High-Performance and Low Overhead Data Transfers](http://arxiv.org/pdf/2508.05546v1)

Authors: Rasman Mubtasim Swargo, Engin Arslan, Md Arifuzzaman

High-performance applications necessitate rapid and dependable transfer of
massive datasets across geographically dispersed locations. Traditional file
transfer tools often suffer from resource underutilization and instability
because of fixed configurations or monolithic optimization methods. We propose
AutoMDT, a novel modular data transfer architecture that employs a deep
reinforcement learning based agent to simultaneously optimize concurrency
levels for read, network, and write operations. Our solution incorporates a
lightweight network-system simulator, enabling offline training of a Proximal
Policy Optimization (PPO) agent in approximately 45 minutes on average, thereby
overcoming the impracticality of lengthy online training in production
networks. AutoMDT's modular design decouples I/O and network tasks, allowing
the agent to capture complex buffer dynamics precisely and to adapt quickly to
changing system and network conditions. Evaluations on production-grade
testbeds show that AutoMDT achieves up to 8x faster convergence and a 68%
reduction in transfer completion times compared with state-of-the-art
solutions.

### 5. [Tesserae: Scalable Placement Policies for Deep Learning Workloads](http://arxiv.org/pdf/2508.04953v1)

Authors: Song Bian, Saurabh Agarwal, Md. Tareq Mahmood, Shivaram Venkataraman

Training deep learning (DL) models has become a dominant workload in
data-centers and improving resource utilization is a key goal of DL cluster
schedulers. In order to do this, schedulers typically incorporate placement
policies that govern where jobs are placed on the cluster. Existing placement
policies are either designed as ad-hoc heuristics or incorporated as
constraints within a complex optimization problem and thus either suffer from
suboptimal performance or poor scalability. Our key insight is that many
placement constraints can be formulated as graph matching problems and based on
that we design novel placement policies for minimizing job migration overheads
and job packing. We integrate these policies into Tesserae and describe how our
design leads to a scalable and effective GPU cluster scheduler. Our
experimental results show that Tesserae improves average JCT by up to 1.62x and
the Makespan by up to 1.15x compared with the existing schedulers.

### 6. [Theseus: A Distributed and Scalable GPU-Accelerated Query Processing Platform Optimized for Efficient Data Movement](http://arxiv.org/pdf/2508.05029v1)

Authors: Felipe Aramburú, William Malpica, Kaouther Abrougui, Amin Aramoon, Romulo Auccapuclla, Claude Brisson, Matthijs Brobbel, Colby Farrell, Pradeep Garigipati, Joost Hoozemans, Supun Kamburugamuve, Akhil Nair, Alexander Ocsa, Johan Peltenburg, Rubén Quesada López, Deepak Sihag, Ahmet Uyar, Dhruv Vats, Michael Wendt, Jignesh M. Patel, Rodrigo Aramburú

Online analytical processing of queries on datasets in the many-terabyte
range is only possible with costly distributed computing systems. To decrease
the cost and increase the throughput, systems can leverage accelerators such as
GPUs, which are now ubiquitous in the compute infrastructure. This introduces
many challenges, the majority of which are related to when, where, and how to
best move data around the system. We present Theseus -- a production-ready
enterprise-scale distributed accelerator-native query engine designed to
balance data movement, memory utilization, and computation in an
accelerator-based system context. Specialized asynchronous control mechanisms
are tightly coupled to the hardware resources for the purpose of network
communication, data pre-loading, data spilling across memories and storage, and
GPU compute tasks. The memory subsystem contains a mechanism for fixed-size
page-locked host memory allocations to increase throughput and reduce memory
fragmentation. For the TPC-H benchmarks at scale factors ranging from 1k to 30k
on cloud infrastructure, Theseus outperforms Databricks Photon by up to
$4\times$ at cost parity. Theseus is capable of processing all queries of the
TPC-H and TPC-DS benchmarks at scale factor 100k (100 TB scale) with as few as
2 DGX A100 640GB nodes.

### 7. [HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation](http://arxiv.org/pdf/2508.05135v1)

Authors: Thinh Nguyen, Trung Phan, Binh T. Nguyen, Khoa D Doan, Kok-Seng Wong

Federated Learning (FL) is a decentralized approach where multiple clients
collaboratively train a shared global model without sharing their raw data.
Despite its effectiveness, conventional FL faces scalability challenges due to
excessive computational and communication demands placed on a single central
server as the number of participating devices grows. Hierarchical Federated
Learning (HFL) addresses these issues by distributing model aggregation tasks
across intermediate nodes (stations), thereby enhancing system scalability and
robustness against single points of failure. However, HFL still suffers from a
critical yet often overlooked limitation: domain shift, where data
distributions vary significantly across different clients and stations,
reducing model performance on unseen target domains. While Federated Domain
Generalization (FedDG) methods have emerged to improve robustness to domain
shifts, their integration into HFL frameworks remains largely unexplored. In
this paper, we formally introduce Hierarchical Federated Domain Generalization
(HFedDG), a novel scenario designed to investigate domain shift within
hierarchical architectures. Specifically, we propose HFedATM, a hierarchical
aggregation method that first aligns the convolutional filters of models from
different stations through Filter-wise Optimal Transport Alignment and
subsequently merges aligned models using a Shrinkage-aware Regularized Mean
Aggregation. Our extensive experimental evaluations demonstrate that HFedATM
significantly boosts the performance of existing FedDG baselines across
multiple datasets and maintains computational and communication efficiency.
Moreover, theoretical analyses indicate that HFedATM achieves tighter
generalization error bounds compared to standard hierarchical averaging,
resulting in faster convergence and stable training behavior.

### 8. [Task-Based Programming for Adaptive Mesh Refinement in Compressible Flow Simulations](http://arxiv.org/pdf/2508.05020v1)

Authors: Anjiang Wei, Hang Song, Mert Hidayetoglu, Elliott Slaughter, Sanjiva K. Lele, Alex Aiken

High-order solvers for compressible flows are vital in scientific
applications. Adaptive mesh refinement (AMR) is a key technique for reducing
computational cost by concentrating resolution in regions of interest. In this
work, we develop an AMR-based numerical solver using Regent, a high-level
programming language for the Legion programming model. We address several
challenges associated with implementing AMR in Regent. These include dynamic
data structures for patch refinement/coarsening, mesh validity enforcement, and
reducing task launch overhead via task fusion. Experimental results show that
task fusion achieves 18x speedup, while automated GPU kernel generation via
simple annotations yields 9.7x speedup for the targeted kernel. We demonstrate
our approach through simulations of two canonical compressible flow problems
governed by the Euler equations.

### 9. [X-VFL: A New Vertical Federated Learning Framework with Cross Completion and Decision Subspace Alignment](http://arxiv.org/pdf/2508.05568v1)

Authors: Qinghua Yao, Xiangrui Xu, Zhize Li

Vertical Federated Learning (VFL) enables collaborative learning by
integrating disjoint feature subsets from multiple clients/parties. However,
VFL typically faces two key challenges: i) the requirement for perfectly
aligned data samples across all clients (missing features are not allowed); ii)
the requirement for joint collaborative inference/prediction involving all
clients (it does not support locally independent inference on a single client).
To address these challenges, we propose X-VFL, a new VFL framework designed to
deal with the non-aligned data samples with (partially) missing features and to
support locally independent inference of new data samples for each client. In
particular, we design two novel modules in X-VFL: Cross Completion (XCom) and
Decision Subspace Alignment (DS-Align). XCom can complete/reconstruct missing
features for non-aligned data samples by leveraging information from other
clients. DS-Align aligns local features with completed and global features
across all clients within the decision subspace, thus enabling locally
independent inference at each client. Moreover, we provide convergence theorems
for different algorithms used in training X-VFL, showing an $O(1/\sqrt{T})$
convergence rate for SGD-type algorithms and an $O(1/T)$ rate for PAGE-type
algorithms, where $T$ denotes the number of training update steps. Extensive
experiments on real-world datasets demonstrate that X-VFL significantly
outperforms existing methods, e.g., achieving a 15% improvement in accuracy on
the image CIFAR-10 dataset and a 43% improvement on the medical MIMIC-III
dataset. These results validate the practical effectiveness and superiority of
X-VFL, particularly in scenarios involving partially missing features and
locally independent inference.

### Digital Libraries

### 1. [Situated Epistemic Infrastructures: A Diagnostic Framework for Post-Coherence Knowledge](http://arxiv.org/pdf/2508.04995v1)

Authors: Matthew Kelly

Large Language Models (LLMs) such as ChatGPT have rendered visible the
fragility of contemporary knowledge infrastructures by simulating coherence
while bypassing traditional modes of citation, authority, and validation. This
paper introduces the Situated Epistemic Infrastructures (SEI) framework as a
diagnostic tool for analyzing how knowledge becomes authoritative across hybrid
human-machine systems under post-coherence conditions. Rather than relying on
stable scholarly domains or bounded communities of practice, SEI traces how
credibility is mediated across institutional, computational, and temporal
arrangements. Integrating insights from infrastructure studies, platform
theory, and epistemology, the framework foregrounds coordination over
classification, emphasizing the need for anticipatory and adaptive models of
epistemic stewardship. The paper contributes to debates on AI governance,
knowledge production, and the ethical design of information systems by offering
a robust alternative to representationalist models of scholarly communication.

### Discrete Mathematics

### 1. [Space-Efficient Hierholzer: Eulerian Cycles in O(m) Time and O(n) Space](http://arxiv.org/pdf/2508.05251v1)

Authors: Ziad Ismaili Alaoui, Detlef Plump, Sebastian Wild

We describe a simple variant of Hierholzer's algorithm that finds an Eulerian
cycle in a (multi)graph with $n$ vertices and $m$ edges using $\mathrm{O}(n \lg
m)$ bits of working memory. This substantially improves the working space
compared to standard implementations of Hierholzer's algorithm, which use
$\mathrm{O}(m \lg n)$ bits of space. Our algorithm runs in linear time, like
the classical versions, but avoids an $\mathrm{O}(m)$-size stack of vertices or
storing information for each edge. To our knowledge, this is the first
linear-time algorithm to achieve this space bound, and the method is very easy
to implement. The correctness argument, by contrast, is surprisingly subtle; we
give a detailed formal proof. The space savings are particularly relevant for
dense graphs or multigraphs with large edge multiplicities.

### 2. [Aircraft routing: periodicity and complexity](http://arxiv.org/pdf/2508.05532v1)

Authors: Frédéric Meunier, Axel Parmentier, Nour ElHouda Tellache

The aircraft routing problem is one of the most studied problems of
operations research applied to aircraft management. It involves assigning
flights to aircraft while ensuring regular visits to maintenance bases. This
paper examines two aspects of the problem.
  First, we explore the relationship between periodic instances, where flights
are the same every day, and periodic solutions. The literature has implicitly
assumed-without discussion-that periodic instances necessitate periodic
solutions, and even periodic solutions in a stronger form, where every two
airplanes perform either the exact same cyclic sequence of flights, or
completely disjoint cyclic sequences. However, enforcing such periodicity may
eliminate feasible solutions. We prove that, when regular maintenance is
required at most every four days, there always exist periodic solutions of this
form.
  Second, we consider the computational hardness of the problem. Even if many
papers in this area refer to the NP-hardness of the aircraft routing problem,
such a result is only available in the literature for periodic instances. We
establish its NP-hardness for a non-periodic version. Polynomiality of a
special but natural case is also proven.

### 3. [Improved lower bounds on the maximum size of graphs with girth 5](http://arxiv.org/pdf/2508.05562v1)

Authors: Jan Goedgebeur, Jorik Jooken, Gwenaël Joret, Tibo Van den Eede

We present a new algorithm for improving lower bounds on $ex(n;\{C_3,C_4\})$,
the maximum size (number of edges) of an $n$-vertex graph of girth at least 5.
The core of our algorithm is a variant of a hill-climbing heuristic introduced
by Exoo, McKay, Myrvold and Nadon (2011) to find small cages. Our algorithm
considers a range of values of $n$ in multiple passes. In each pass, the
hill-climbing heuristic for a specific value of $n$ is initialized with a few
graphs obtained by modifying near-extremal graphs previously found for
neighboring values of $n$, allowing to `propagate' good patterns that were
found. Focusing on the range $n\in \{74,75, \dots, 198\}$, which is currently
beyond the scope of exact methods, our approach yields improvements on existing
lower bounds for $ex(n;\{C_3,C_4\})$ for all $n$ in the range, except for two
values of $n$ ($n=96,97$).

### 4. [Balanced Steinhaus triangles](http://arxiv.org/pdf/2508.05159v1)

Authors: Jonathan Chappelon

A Steinhaus triangle modulo $m$ is a finite down-pointing triangle of
elements in the finite cyclic group $\mathbb{Z}/m\mathbb{Z}$ satisfying the
same local rule as the standard Pascal triangle modulo $m$. A Steinhaus
triangle modulo $m$ is said to be balanced if it contains all the elements of
$\mathbb{Z}/m\mathbb{Z}$ with the same multiplicity. In this paper, the
existence of infinitely many balanced Steinhaus triangles modulo $m$, for any
positive integer $m$, is shown. This is achieved by considering periodic
triangles generated from interlaced arithmetic progressions. This positively
answers a weak version of a problem, due to John C. Molluzzo in 1978, that has
remained unsolved to date for the even values of $m\geqslant 12$.

### 5. [Parameterized complexity of isometric path partition: treewidth and diameter](http://arxiv.org/pdf/2508.05448v1)

Authors: Dibyayan Chakraborty, Oscar Defrain, Florent Foucaud, Mathieu Mari, Prafullkumar Tale

We investigate the parameterized complexity of the Isometric Path Partition
problem when parameterized by the treewidth ($\mathrm{tw}$) of the input graph,
arguably one of the most widely studied parameters. Courcelle's theorem shows
that graph problems that are expressible as MSO formulas of constant size admit
FPT algorithms parameterized by the treewidth of the input graph. This
encompasses many natural graph problems. However, many metric-based graph
problems, where the solution is defined using some metric-based property of the
graph (often the distance) are not expressible as MSO formulas of constant
size. These types of problems, Isometric Path Partition being one of them,
require individual attention and often draw the boundary for the success story
of parameterization by treewidth.
  In this paper, we prove that Isometric Path Partition is $W[1]$-hard when
parameterized by treewidth (in fact, even pathwidth), answering the question by
Dumas et al. [SIDMA, 2024], Fernau et al. [CIAC, 2023], and confirming the
aforementioned tendency. We complement this hardness result by designing a
tailored dynamic programming algorithm running in $n^{O(\mathrm{tw})}$ time.
This dynamic programming approach also results in an algorithm running in time
$\textrm{diam}^{O(\mathrm{tw}^2)} \cdot n^{O(1)}$, where $\textrm{diam}$ is the
diameter of the graph. Note that the dependency on treewidth is unusually high,
as most problems admit algorithms running in time $2^{O(\mathrm{tw})}\cdot
n^{O(1)}$ or $2^{O(\mathrm{tw} \log (\mathrm{tw}))}\cdot n^{O(1)}$. However, we
rule out the possibility of a significantly faster algorithm by proving that
Isometric Path Partition does not admit an algorithm running in time
$\textrm{diam}^{o(\mathrm{tw}^2/(\log^3(\mathrm{tw})))} \cdot n^{O(1)}$, unless
the Randomized-ETH fails.

### Data Structures and Algorithms

### 1. [Text Indexing and Pattern Matching with Ephemeral Edits](http://arxiv.org/pdf/2508.05124v1)

Authors: Solon P. Pissis

A sequence $e_0,e_1,\ldots$ of edit operations in a string $T$ is called
ephemeral if operation $e_i$ constructing string $T^i$, for all $i=2k$ with
$k\in\mathbb{N}$, is reverted by operation $e_{i+1}$ that reconstructs $T$.
Such a sequence arises when processing a stream of independent edits or testing
hypothetical edits.
  We introduce text indexing with ephemeral substring edits, a new version of
text indexing. Our goal is to design a data structure over a given text that
supports subsequent pattern matching queries with ephemeral substring
insertions, deletions, or substitutions in the text; we require insertions and
substitutions to be of constant length. In particular, we preprocess a text
$T=T[0\mathinner{.\,.} n)$ over an integer alphabet $\Sigma=[0,\sigma)$ with
$\sigma=n^{\mathcal{O}(1)}$ in $\mathcal{O}(n)$ time. Then, we can preprocess
any arbitrary pattern $P=P[0\mathinner{.\,.} m)$ given online in
$\mathcal{O}(m\log\log m)$ time and $\mathcal{O}(m)$ space and allow any
ephemeral sequence of edit operations in $T$. Before reverting the $i$th
operation, we report all Occ occurrences of $P$ in $T^i$ in
$\mathcal{O}(\log\log n + \text{Occ})$ time.
  We also introduce pattern matching with ephemeral edits. In particular, we
preprocess two strings $T$ and $P$, each of length at most $n$, over an integer
alphabet $\Sigma=[0,\sigma)$ with $\sigma=n^{\mathcal{O}(1)}$ in
$\mathcal{O}(n)$ time. Then, we allow any ephemeral sequence of edit operations
in $T$. Before reverting the $i$th operation, we report all Occ occurrences of
$P$ in $T^i$ in the optimal $\mathcal{O}(\text{Occ})$ time. Along our way to
this result, we also give an optimal solution for pattern matching with
ephemeral block deletions.

### 2. [Parameterized Algorithms for Spanning Tree Isomorphism by Redundant Set Size](http://arxiv.org/pdf/2508.05351v1)

Authors: Fangjian Shen, Yicheng Zheng, Wushao Wen, Hankz Hankui Zhuo

In this paper, we present fixed-parameter tractability algorithms for both
the undirected and directed versions of the Spanning Tree Isomorphism Problem,
parameterized by the size $k$ of a redundant set. A redundant set is a
collection of edges whose removal transforms the graph into a spanning tree.
For the undirected version, our algorithm achieves a time complexity of $O(n^2
\log n \cdot 2^{k \log k})$. For the directed version, we propose a more
efficient algorithm with a time complexity of $O(n^2 \cdot 2^{4k-3})$, where
$n$ is the number of vertices.

### 3. [An Improved Approximation Algorithm for the Capacitated Arc Routing Problem](http://arxiv.org/pdf/2508.05471v1)

Authors: Jingyang Zhao, Mingyu Xiao

The Capacitated Arc Routing Problem (CARP), introduced by Golden and Wong in
1981, is an important arc routing problem in Operations Research, which
generalizes the famous Capacitated Vehicle Routing Problem (CVRP). When every
customer has a unit demand, the best known approximation ratio for CARP, given
by Jansen in 1993, remains $\frac{5}{2}-\frac{1.5}{k}$, where $k$ denotes the
vehicle capacity. Based on recent progress in approximating CVRP, we improve
this result by proposing a
$(\frac{5}{2}-\Theta(\frac{1}{\sqrt{k}}))$-approximation algorithm, which to
the best of our knowledge constitutes the first improvement over Jansen's
bound.

### 4. [Minimum-Weight Parity Factor Decoder for Quantum Error Correction](http://arxiv.org/pdf/2508.04969v1)

Authors: Yue Wu, Binghong Li, Kathleen Chang, Shruti Puri, Lin Zhong

Fast and accurate quantum error correction (QEC) decoding is crucial for
scalable fault-tolerant quantum computation. Most-Likely-Error (MLE) decoding,
while being near-optimal, is intractable on general quantum Low-Density
Parity-Check (qLDPC) codes and typically relies on approximation and
heuristics. We propose HyperBlossom, a unified framework that formulates MLE
decoding as a Minimum-Weight Parity Factor (MWPF) problem and generalizes the
blossom algorithm to hypergraphs via a similar primal-dual linear programming
model with certifiable proximity bounds. HyperBlossom unifies all the existing
graph-based decoders like (Hypergraph) Union-Find decoders and Minimum-Weight
Perfect Matching (MWPM) decoder, thus bridging the gap between heuristic and
certifying decoders.
  We implement HyperBlossom in software, namely Hyperion. Hyperion achieves a
4.8x lower logical error rate compared to the MWPM decoder on the distance-11
surface code and 1.6x lower logical error rate compared to a fine-tuned BPOSD
decoder on the $[[90, 8, 10]]$ bivariate bicycle code under code-capacity
noise. It also achieves an almost-linear average runtime scaling on both the
surface code and the color code, with numerical results up to sufficiently
large code distances of 99 and 31 for code-capacity noise and circuit-level
noise, respectively.

### 5. [Space-Efficient Hierholzer: Eulerian Cycles in O(m) Time and O(n) Space](http://arxiv.org/pdf/2508.05251v1)

Authors: Ziad Ismaili Alaoui, Detlef Plump, Sebastian Wild

We describe a simple variant of Hierholzer's algorithm that finds an Eulerian
cycle in a (multi)graph with $n$ vertices and $m$ edges using $\mathrm{O}(n \lg
m)$ bits of working memory. This substantially improves the working space
compared to standard implementations of Hierholzer's algorithm, which use
$\mathrm{O}(m \lg n)$ bits of space. Our algorithm runs in linear time, like
the classical versions, but avoids an $\mathrm{O}(m)$-size stack of vertices or
storing information for each edge. To our knowledge, this is the first
linear-time algorithm to achieve this space bound, and the method is very easy
to implement. The correctness argument, by contrast, is surprisingly subtle; we
give a detailed formal proof. The space savings are particularly relevant for
dense graphs or multigraphs with large edge multiplicities.

### 6. [Online Sparsification of Bipartite-Like Clusters in Graphs](http://arxiv.org/pdf/2508.05437v1)

Authors: Joyentanuj Das, Suranjan De, He Sun

Graph clustering is an important algorithmic technique for analysing massive
graphs, and has been widely applied in many research fields of data science.
While the objective of most graph clustering algorithms is to find a vertex set
of low conductance, a sequence of recent studies highlights the importance of
the inter-connection between vertex sets when analysing real-world datasets.
Following this line of research, in this work we study bipartite-like clusters
and present efficient and online sparsification algorithms that find such
clusters in both undirected graphs and directed ones. We conduct experimental
studies on both synthetic and real-world datasets, and show that our algorithms
significantly speedup the running time of existing clustering algorithms while
preserving their effectiveness.

### 7. [Necessity of Block Designs for Optimal Locally Private Distribution Estimation](http://arxiv.org/pdf/2508.05110v1)

Authors: Abigail Gentle

Local differential privacy represents the gold standard for preserving the
privacy of data before it leaves the device, and distribution estimation under
this model has been well studied. Recently, protocols built upon balanced
incomplete block designs were shown to achieve optimal error for this problem.
However, it remained unknown whether other constructions could also be optimal.
We resolve this question by proving that any protocol achieving optimal error
must correspond to some balanced incomplete block design. This result, combined
with prior work, completely characterises the set of optimal protocols for this
problem. As a consequence, the protocols that achieve optimal error and optimal
communication are only those based on symmetrical balanced incomplete block
designs.

### 8. [Parameterized complexity of isometric path partition: treewidth and diameter](http://arxiv.org/pdf/2508.05448v1)

Authors: Dibyayan Chakraborty, Oscar Defrain, Florent Foucaud, Mathieu Mari, Prafullkumar Tale

We investigate the parameterized complexity of the Isometric Path Partition
problem when parameterized by the treewidth ($\mathrm{tw}$) of the input graph,
arguably one of the most widely studied parameters. Courcelle's theorem shows
that graph problems that are expressible as MSO formulas of constant size admit
FPT algorithms parameterized by the treewidth of the input graph. This
encompasses many natural graph problems. However, many metric-based graph
problems, where the solution is defined using some metric-based property of the
graph (often the distance) are not expressible as MSO formulas of constant
size. These types of problems, Isometric Path Partition being one of them,
require individual attention and often draw the boundary for the success story
of parameterization by treewidth.
  In this paper, we prove that Isometric Path Partition is $W[1]$-hard when
parameterized by treewidth (in fact, even pathwidth), answering the question by
Dumas et al. [SIDMA, 2024], Fernau et al. [CIAC, 2023], and confirming the
aforementioned tendency. We complement this hardness result by designing a
tailored dynamic programming algorithm running in $n^{O(\mathrm{tw})}$ time.
This dynamic programming approach also results in an algorithm running in time
$\textrm{diam}^{O(\mathrm{tw}^2)} \cdot n^{O(1)}$, where $\textrm{diam}$ is the
diameter of the graph. Note that the dependency on treewidth is unusually high,
as most problems admit algorithms running in time $2^{O(\mathrm{tw})}\cdot
n^{O(1)}$ or $2^{O(\mathrm{tw} \log (\mathrm{tw}))}\cdot n^{O(1)}$. However, we
rule out the possibility of a significantly faster algorithm by proving that
Isometric Path Partition does not admit an algorithm running in time
$\textrm{diam}^{o(\mathrm{tw}^2/(\log^3(\mathrm{tw})))} \cdot n^{O(1)}$, unless
the Randomized-ETH fails.

### 9. [NP-Hardness and ETH-Based Inapproximability of Communication Complexity via Relaxed Interlacing](http://arxiv.org/pdf/2508.05597v1)

Authors: Serge Gaspers, Zixu He, Simon Mackenzie

We prove that computing the deterministic communication complexity D(f) of a
Boolean function is NP-hard, even when protocols are limited to a constant
number of alternations, resolving a question first posed by Yao (1979). Our
reduction builds and expands on a suite of structural "interlacing" lemmas
introduced by Mackenzie and Saffidine (arXiv:2411.19003); these lemmas can be
reused as black boxes in future lower-bound constructions.
  The instances produced by our reduction admit optimal protocols that use only
constant alternations, so NP-hardness holds under stronger restrictions than
those considered in concurrent and independent work by Hirahara, Ilango, and
Loff (arXiv:2507.10426), whose proof requires unbounded alternations.
  Because the gadgets in our construction are self-similar, they can be
recursively embedded. We sketch how this yields, under the Exponential-Time
Hypothesis, an additive inapproximability gap that grows without bound, and we
outline a route toward NP-hardness of approximating D(f) within a fixed
constant additive error. Full details of the ETH-based inapproximability
results will appear in a future version.
  Beyond settling the complexity of deterministic communication complexity
itself, the modular framework we develop opens the door to a wider class of
reductions and, we believe, will prove useful in tackling other long-standing
questions in communication complexity.

### Emerging Technologies

### 1. [Wave Computing based on Dynamical Networks: Applications in Optimization Problems](http://arxiv.org/pdf/2508.05014v1)

Authors: Yunwen Liu, Jiang Xiao

We develop a computing framework that leverages wave propagation within an
interconnected network, where nodes and edges possess wave manipulation
capabilities, such as frequency mixing or time delay. This computing paradigm
can not only achieve intrinsic parallelism like existing works by the
exploration of an exponential number of possibilities simultaneously with very
small number of hardware units, but also extend this unique characteristic to a
multidimensional space including spatial, temporal and frequency domains,
making it particularly effective for addressing NP-hard problems. The proposed
architecture has been validated through SPICE simulations, demonstrating its
potential capability in solving several NP-hard problems, such as the Number
Partitioning Problem, the 0/1 Knapsack Problem, and the Traveling Salesman
Problem.

### 2. [QFOR: A Fidelity-aware Orchestrator for Quantum Computing Environments using Deep Reinforcement Learning](http://arxiv.org/pdf/2508.04974v1)

Authors: Hoa T. Nguyen, Muhammad Usman, Rajkumar Buyya

Quantum cloud computing enables remote access to quantum processors, yet the
heterogeneity and noise of available quantum hardware create significant
challenges for efficient resource orchestration. These issues complicate the
optimization of quantum task allocation and scheduling, as existing heuristic
methods fall short in adapting to dynamic conditions or effectively balancing
execution fidelity and time. Here, we propose QFOR, a Quantum Fidelity-aware
Orchestration of tasks across heterogeneous quantum nodes in cloud-based
environments using Deep Reinforcement learning. We model the quantum task
orchestration as a Markov Decision Process and employ the Proximal Policy
Optimization algorithm to learn adaptive scheduling policies, using IBM quantum
processor calibration data for noise-aware performance estimation. Our
configurable framework balances overall quantum task execution fidelity and
time, enabling adaptation to different operational priorities. Extensive
evaluation demonstrates that QFOR is adaptive and achieves significant
performance with 29.5-84% improvements in relative fidelity performance over
heuristic baselines. Furthermore, it maintains comparable quantum execution
times, contributing to cost-efficient use of quantum computation resources.

### 3. [Salt-Rock Creep Deformation Forecasting Using Deep Neural Networks and Analytical Models for Subsurface Energy Storage Applications](http://arxiv.org/pdf/2508.05248v1)

Authors: Pradeep Kumar Shukla, Tanujit Chakraborty, Mustafa Sari, Joel Sarout, Partha Pratim Mandal

This study provides an in-depth analysis of time series forecasting methods
to predict the time-dependent deformation trend (also known as creep) of salt
rock under varying confining pressure conditions. Creep deformation assessment
is essential for designing and operating underground storage facilities for
nuclear waste, hydrogen energy, or radioactive materials. Salt rocks, known for
their mechanical properties like low porosity, low permeability, high
ductility, and exceptional creep and self-healing capacities, were examined
using multi-stage triaxial (MSTL) creep data. After resampling, axial strain
datasets were recorded at 5--10 second intervals under confining pressure
levels ranging from 5 to 35 MPa over 5.8--21 days. Initial analyses, including
Seasonal-Trend Decomposition (STL) and Granger causality tests, revealed
minimal seasonality and causality between axial strain and temperature data.
Further statistical tests, such as the Augmented Dickey-Fuller (ADF) test,
confirmed the stationarity of the data with p-values less than 0.05, and
wavelet coherence plot (WCP) analysis indicated repeating trends. A suite of
deep neural network (DNN) models (Neural Basis Expansion Analysis for Time
Series (N-BEATS), Temporal Convolutional Networks (TCN), Recurrent Neural
Networks (RNN), and Transformers (TF)) was utilized and compared against
statistical baseline models. Predictive performance was evaluated using Root
Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage
Error (MAPE), and Symmetric Mean Absolute Percentage Error (SMAPE). Results
demonstrated that N-BEATS and TCN models outperformed others across various
stress levels, respectively. DNN models, particularly N-BEATS and TCN, showed a
15--20\% improvement in accuracy over traditional analytical models,
effectively capturing complex temporal dependencies and patterns.

### Formal Languages and Automata Theory

### 1. [Overview of Controllability Definitions in Supervisory Control Theory](http://arxiv.org/pdf/2508.05177v1)

Authors: Jeroen J. A. Keiren, Michel A. Reniers

In the field of supervisory control theory, the literature often proposes
different definitions for the same concept, making it difficult to understand
how these definitions are related. This is definitely so for the fundamental
notion of controllability of a supervisor w.r.t. a plant. This paper lists
definitions of controllability found in the literature and studies their
relationships in settings of both deterministic and nondeterministic automata.
In the general context, where both the supervisor and the plant are allowed to
be nondeterministic, the notions of controllability as described by Flordal and
Malik, and uncontrollable event admissibility by Kushi and Takai are
equivalent. These are also the only notions that imply the traditional notion
of (language) controllability. From a practical perspective, one is often more
interested in controllability of a supervised plant w.r.t. a plant. In this
context, in addition to the previous two controllability notions, state
controllability by Zhou et al. implies language controllability.

### Graphics

### 1. [Open-world Point Cloud Semantic Segmentation: A Human-in-the-loop Framework](http://arxiv.org/pdf/2508.04962v1)

Authors: Peng Zhang, Songru Yang, Jinsheng Sun, Weiqing Li, Zhiyong Su

Open-world point cloud semantic segmentation (OW-Seg) aims to predict point
labels of both base and novel classes in real-world scenarios. However,
existing methods rely on resource-intensive offline incremental learning or
densely annotated support data, limiting their practicality. To address these
limitations, we propose HOW-Seg, the first human-in-the-loop framework for
OW-Seg. Specifically, we construct class prototypes, the fundamental
segmentation units, directly on the query data, avoiding the prototype bias
caused by intra-class distribution shifts between the support and query data.
By leveraging sparse human annotations as guidance, HOW-Seg enables
prototype-based segmentation for both base and novel classes. Considering the
lack of granularity of initial prototypes, we introduce a hierarchical
prototype disambiguation mechanism to refine ambiguous prototypes, which
correspond to annotations of different classes. To further enrich contextual
awareness, we employ a dense conditional random field (CRF) upon the refined
prototypes to optimize their label assignments. Through iterative human
feedback, HOW-Seg dynamically improves its predictions, achieving high-quality
segmentation for both base and novel classes. Experiments demonstrate that with
sparse annotations (e.g., one-novel-class-one-click), HOW-Seg matches or
surpasses the state-of-the-art generalized few-shot segmentation (GFS-Seg)
method under the 5-shot setting. When using advanced backbones (e.g.,
Stratified Transformer) and denser annotations (e.g., 10 clicks per sub-scene),
HOW-Seg achieves 85.27% mIoU on S3DIS and 66.37% mIoU on ScanNetv2,
significantly outperforming alternatives.

### 2. [Point cloud segmentation for 3D Clothed Human Layering](http://arxiv.org/pdf/2508.05531v1)

Authors: Davide Garavaso, Federico Masi, Pietro Musoni, Umberto Castellani

3D Cloth modeling and simulation is essential for avatars creation in several
fields, such as fashion, entertainment, and animation. Achieving high-quality
results is challenging due to the large variability of clothed body especially
in the generation of realistic wrinkles. 3D scan acquisitions provide more
accuracy in the representation of real-world objects but lack semantic
information that can be inferred with a reliable semantic reconstruction
pipeline. To this aim, shape segmentation plays a crucial role in identifying
the semantic shape parts. However, current 3D shape segmentation methods are
designed for scene understanding and interpretation and only few work is
devoted to modeling. In the context of clothed body modeling the segmentation
is a preliminary step for fully semantic shape parts reconstruction namely the
underlying body and the involved garments. These parts represent several layers
with strong overlap in contrast with standard segmentation methods that provide
disjoint sets. In this work we propose a new 3D point cloud segmentation
paradigm where each 3D point can be simultaneously associated to different
layers. In this fashion we can estimate the underlying body parts and the
unseen clothed regions, i.e., the part of a cloth occluded by the clothed-layer
above. We name this segmentation paradigm clothed human layering. We create a
new synthetic dataset that simulates very realistic 3D scans with the ground
truth of the involved clothing layers. We propose and evaluate different neural
network settings to deal with 3D clothing layering. We considered both coarse
and fine grained per-layer garment identification. Our experiments demonstrates
the benefit in introducing proper strategies for the segmentation on the
garment domain on both the synthetic and real-world scan datasets.

### 3. [Physically Controllable Relighting of Photographs](http://arxiv.org/pdf/2508.05626v1)

Authors: Chris Careaga, Yağız Aksoy

We present a self-supervised approach to in-the-wild image relighting that
enables fully controllable, physically based illumination editing. We achieve
this by combining the physical accuracy of traditional rendering with the
photorealistic appearance made possible by neural rendering. Our pipeline works
by inferring a colored mesh representation of a given scene using monocular
estimates of geometry and intrinsic components. This representation allows
users to define their desired illumination configuration in 3D. The scene under
the new lighting can then be rendered using a path-tracing engine. We send this
approximate rendering of the scene through a feed-forward neural renderer to
predict the final photorealistic relighting result. We develop a differentiable
rendering process to reconstruct in-the-wild scene illumination, enabling
self-supervised training of our neural renderer on raw image collections. Our
method represents a significant step in bringing the explicit physical control
over lights available in typical 3D computer graphics tools, such as Blender,
to in-the-wild relighting.

### 4. [Perceive-Sample-Compress: Towards Real-Time 3D Gaussian Splatting](http://arxiv.org/pdf/2508.04965v1)

Authors: Zijian Wang, Beizhen Zhao, Hao Wang

Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated remarkable
capabilities in real-time and photorealistic novel view synthesis. However,
traditional 3DGS representations often struggle with large-scale scene
management and efficient storage, particularly when dealing with complex
environments or limited computational resources. To address these limitations,
we introduce a novel perceive-sample-compress framework for 3D Gaussian
Splatting. Specifically, we propose a scene perception compensation algorithm
that intelligently refines Gaussian parameters at each level. This algorithm
intelligently prioritizes visual importance for higher fidelity rendering in
critical areas, while optimizing resource usage and improving overall visible
quality. Furthermore, we propose a pyramid sampling representation to manage
Gaussian primitives across hierarchical levels. Finally, to facilitate
efficient storage of proposed hierarchical pyramid representations, we develop
a Generalized Gaussian Mixed model compression algorithm to achieve significant
compression ratios without sacrificing visual fidelity. The extensive
experiments demonstrate that our method significantly improves memory
efficiency and high visual quality while maintaining real-time rendering speed.

### 5. [Laplacian Analysis Meets Dynamics Modelling: Gaussian Splatting for 4D Reconstruction](http://arxiv.org/pdf/2508.04966v1)

Authors: Yifan Zhou, Beizhen Zhao, Pengcheng Wu, Hao Wang

While 3D Gaussian Splatting (3DGS) excels in static scene modeling, its
extension to dynamic scenes introduces significant challenges. Existing dynamic
3DGS methods suffer from either over-smoothing due to low-rank decomposition or
feature collision from high-dimensional grid sampling. This is because of the
inherent spectral conflicts between preserving motion details and maintaining
deformation consistency at different frequency. To address these challenges, we
propose a novel dynamic 3DGS framework with hybrid explicit-implicit functions.
Our approach contains three key innovations: a spectral-aware Laplacian
encoding architecture which merges Hash encoding and Laplacian-based module for
flexible frequency motion control, an enhanced Gaussian dynamics attribute that
compensates for photometric distortions caused by geometric deformation, and an
adaptive Gaussian split strategy guided by KDTree-based primitive control to
efficiently query and optimize dynamic areas. Through extensive experiments,
our method demonstrates state-of-the-art performance in reconstructing complex
dynamic scenes, achieving better reconstruction fidelity.

### 6. [A Study of the Framework and Real-World Applications of Language Embedding for 3D Scene Understanding](http://arxiv.org/pdf/2508.05064v1)

Authors: Mahmoud Chick Zaouali, Todd Charter, Yehor Karpichev, Brandon Haworth, Homayoun Najjjaran

Gaussian Splatting has rapidly emerged as a transformative technique for
real-time 3D scene representation, offering a highly efficient and expressive
alternative to Neural Radiance Fields (NeRF). Its ability to render complex
scenes with high fidelity has enabled progress across domains such as scene
reconstruction, robotics, and interactive content creation. More recently, the
integration of Large Language Models (LLMs) and language embeddings into
Gaussian Splatting pipelines has opened new possibilities for text-conditioned
generation, editing, and semantic scene understanding. Despite these advances,
a comprehensive overview of this emerging intersection has been lacking. This
survey presents a structured review of current research efforts that combine
language guidance with 3D Gaussian Splatting, detailing theoretical
foundations, integration strategies, and real-world use cases. We highlight key
limitations such as computational bottlenecks, generalizability, and the
scarcity of semantically annotated 3D Gaussian data and outline open challenges
and future directions for advancing language-guided 3D scene understanding
using Gaussian Splatting.

### 7. [RAP: Real-time Audio-driven Portrait Animation with Video Diffusion Transformer](http://arxiv.org/pdf/2508.05115v1)

Authors: Fangyu Du, Taiqing Li, Ziwei Zhang, Qian Qiao, Tan Yu, Dingcheng Zhen, Xu Jia, Yang Yang, Shunshun Yin, Siyuan Liu

Audio-driven portrait animation aims to synthesize realistic and natural
talking head videos from an input audio signal and a single reference image.
While existing methods achieve high-quality results by leveraging
high-dimensional intermediate representations and explicitly modeling motion
dynamics, their computational complexity renders them unsuitable for real-time
deployment. Real-time inference imposes stringent latency and memory
constraints, often necessitating the use of highly compressed latent
representations. However, operating in such compact spaces hinders the
preservation of fine-grained spatiotemporal details, thereby complicating
audio-visual synchronization RAP (Real-time Audio-driven Portrait animation), a
unified framework for generating high-quality talking portraits under real-time
constraints. Specifically, RAP introduces a hybrid attention mechanism for
fine-grained audio control, and a static-dynamic training-inference paradigm
that avoids explicit motion supervision. Through these techniques, RAP achieves
precise audio-driven control, mitigates long-term temporal drift, and maintains
high visual fidelity. Extensive experiments demonstrate that RAP achieves
state-of-the-art performance while operating under real-time constraints.

### 8. [Refining Gaussian Splatting: A Volumetric Densification Approach](http://arxiv.org/pdf/2508.05187v1)

Authors: Mohamed Abdul Gafoor, Marius Preda, Titus Zaharia

Achieving high-quality novel view synthesis in 3D Gaussian Splatting (3DGS)
often depends on effective point primitive management. The underlying Adaptive
Density Control (ADC) process addresses this issue by automating densification
and pruning. Yet, the vanilla 3DGS densification strategy shows key
shortcomings. To address this issue, in this paper we introduce a novel density
control method, which exploits the volumes of inertia associated to each
Gaussian function to guide the refinement process. Furthermore, we study the
effect of both traditional Structure from Motion (SfM) and Deep Image Matching
(DIM) methods for point cloud initialization. Extensive experimental
evaluations on the Mip-NeRF 360 dataset demonstrate that our approach surpasses
3DGS in reconstruction quality, delivering encouraging performance across
diverse scenes.

### 9. [GASP: A Gradient-Aware Shortest Path Algorithm for Boundary-Confined Visualization of 2-Manifold Reeb Graphs](http://arxiv.org/pdf/2508.05524v1)

Authors: Sefat Rahman, Tushar M. Athawale, Paul Rosen

Reeb graphs are an important tool for abstracting and representing the
topological structure of a function defined on a manifold. We have identified
three properties for faithfully representing Reeb graphs in a visualization.
Namely, they should be constrained to the boundary, compact, and aligned with
the function gradient. Existing algorithms for drawing Reeb graphs are agnostic
to or violate these properties. In this paper, we introduce an algorithm to
generate Reeb graph visualizations, called \textit{GASP}, that is cognizant of
these properties, thereby producing visualizations that are more representative
of the underlying data. To demonstrate the improvements, the resulting Reeb
graphs are evaluated both qualitatively and quantitatively against the
geometric barycenter algorithm, using its implementation available in the
Topology ToolKit (TTK), a widely adopted tool for calculating and visualizing
Reeb graphs.

### Computer Science and Game Theory

### 1. [A New Three-Players Auction Bridge with Dynamic Opponents and Team Members](http://arxiv.org/pdf/2508.05582v1)

Authors: Sourish Sarkar, Aritrabha Majumdar, Moutushi Chatterjee

This article presents a new three-player version of the bridge playing card
game for the purpose of ending fixed partnerships so that the play can be more
dynamic and flexible. By dynamically redefining team makeup in real time, this
game design increases unpredictability and forces players to repeatedly update
strategy. A novel scoring system is introduced to reduce biases present in
conventional rule-based games by favoring fairness via reward systems that
enforce tactical decision making and risk assessment. Being subject to regular
bridge rules, this version tests players to collaborate without fixed
friendships, requiring fluid adjustment and adaptive bidding behavior in real
time. Strategic issues involve aggressive and defensive bidding, adaptable
playing styles, and loss-seeking strategies specific to the three-player
structure. The article discusses probabilistic issues of bidding, trump and
no-trump declarative effects, and algorithmic methods to trick-taking.
Simulation outcomes illustrate the efficiency of diverse strategies. The game's
architecture is ideal for competitions and possibly influential in broadening
entry pools for tournament card games.

### 2. [Toward Energy and Location-Aware Resource Allocation in Next Generation Networks](http://arxiv.org/pdf/2508.05109v1)

Authors: Mandar Datar, Mattia Merluzzi

Wireless networks are evolving from radio resource providers to complex
systems that also involve computing, with the latter being distributed across
edge and cloud facilities. Also, their optimization is shifting more and more
from a performance to a value-oriented paradigm. The two aspects shall be
balanced continuously, to maximize the utilities of Services Providers (SPs),
users quality of experience and fairness, while meeting global constraints in
terms of energy consumption and carbon footprint among others, with all these
heterogeneous resources contributing. In this paper, we tackle the problem of
communication and compute resource allocation under energy constraints, with
multiple SPs competing to get their preferred resource bundle by spending a a
fictitious currency budget. By modeling the network as a Fisher market, we
propose a low complexity solution able to achieve high utilities and guarantee
energy constraints, while also promoting fairness among SPs, as compared to a
social optimal solution. The market equilibrium is proved mathematically, and
numerical results show the multi-dimensional trade-off between utility and
energy at different locations, with communication and computation-intensive
services.

### 3. [Pairwise efficiency and monotonicity imply Pareto efficiency in (probabilistic) object allocation](http://arxiv.org/pdf/2508.05340v1)

Authors: Tom Demeulemeester, Bettina Klaus

We consider object allocation problems with capacities (see, e.g.,
Abdulkadiroglu and Sonmez, 1998; Basteck, 2025) where objects have to be
assigned to agents. We show that if a lottery rule satisfies ex-post
non-wastefulness and probabilistic (Maskin) monotonicity, then ex-post pairwise
efficiency is equivalent to ex-post Pareto efficiency. This result allows for a
strengthening of various existing characterization results, both for lottery
rules and deterministic rules, by replacing (ex-post) Pareto efficiency with
(ex-post) pairwise efficiency, e.g., for characterizations of the Random Serial
Dictatorship rule (Basteck, 2025), Trading Cycles rules (Pycia and Unver,
2017), and Hierarchical Exchange rules (Papai, 2000).

### Human-Computer Interaction

### 1. [Accessibility Beyond Accommodations: A Systematic Redesign of Introduction to Computer Science for Students with Visual Impairments](http://arxiv.org/pdf/2508.05056v1)

Authors: Vaanee Tripathi, Aalok Thakkar

Computer science education has evolved extensively; however, systemic
barriers still prevent students with visual impairments from fully
participating. While existing research has developed specialized programming
tools and assistive technologies, these solutions remain fragmented and often
require complex technical infrastructure, which limits their classroom
implementation. Current approaches treat accessibility as individual
accommodations rather than integral curriculum design, creating gaps in
holistic educational support. This paper presents a comprehensive framework for
redesigning introductory computer science curricula to provide equitable
learning experiences for students with visual impairments without requiring
specialized technical infrastructure. The framework outlines five key
components that together contribute a systematic approach to curriculum
accessibility: accessible learning resources with pre-distributed materials and
tactile diagrams, in-class learning kits with hands-on demonstrations,
structured support systems with dedicated teaching assistance, an online tool
repository, and psychosocial support for classroom participation. Unlike
existing tool-focused solutions, this framework addresses both technical and
pedagogical dimensions of inclusive education while emphasizing practical
implementation in standard university settings. The design is grounded in
universal design principles and validated through expert consultation with
accessibility specialists and disability services professionals, establishing
foundations for future empirical evaluation of learning outcomes and student
engagement while serving as a template for broader institutional adoption.

### 2. [A Desktop-Centric Design Space for Direct Object Examination and Visualization in Mixed-Reality Environments](http://arxiv.org/pdf/2508.05088v1)

Authors: Sam Johnson-Lacoss, Santiago V. Lombeyda, S. George Djorgovski

Mixed reality (MR) environments are bound to become ubiquitous as MR
technology becomes lighter, higher resolution, more affordable, and overall
becomes a seamless extension of our current work and living spaces. For
research scientists and clinicians focused on understanding 3D phenomena or
patient pathologies within the context of the larger human anatomy, that means
a necessary evolution of their workstations currently only utilizing 2D
interfaces for everyday communication, logistics and data analysis. MR
technologies bring forth immersive 3D representations coexisting in our natural
spaces, while allowing for richer interconnected information displays, where 3D
representations greatly aid in the detailed understanding of physical
structures, spatial relationships, and 3D contextualization of 2D measurements,
projections, abstractions, and other data details. We present a breakdown of
the different interaction zones and modalities into a design space that best
accommodates the creation of applications for users engaged through MR
technologies in precise object-centric data analysis within the ergonomic
confines of their desktop physical spaces.

### 3. [SparseEMG: Computational Design of Sparse EMG Layouts for Sensing Gestures](http://arxiv.org/pdf/2508.05098v1)

Authors: Anand Kumar, Antony Albert Raj Irudayaraj, Ishita Chandra, Adwait Sharma, Aditya Shekhar Nittala

Gesture recognition with electromyography (EMG) is a complex problem
influenced by gesture sets, electrode count and placement, and machine learning
parameters (e.g., features, classifiers). Most existing toolkits focus on
streamlining model development but overlook the impact of electrode selection
on classification accuracy. In this work, we present the first data-driven
analysis of how electrode selection and classifier choice affect both accuracy
and sparsity. Through a systematic evaluation of 28 combinations (4 selection
schemes, 7 classifiers), across six datasets, we identify an approach that
minimizes electrode count without compromising accuracy. The results show that
Permutation Importance (selection scheme) with Random Forest (classifier)
reduces the number of electrodes by 53.5\%. Based on these findings, we
introduce SparseEMG, a design tool that generates sparse electrode layouts
based on user-selected gesture sets, electrode constraints, and ML parameters
while also predicting classification performance. SparseEMG supports 50+ unique
gestures and is validated in three real-world applications using different
hardware setups. Results from our multi-dataset evaluation show that the
layouts generated from the SparseEMG design tool are transferable across users
with only minimal variation in gesture recognition performance.

### 4. [Metacognition and self-regulated learning in manipulative robotic problem-solving task](http://arxiv.org/pdf/2508.05112v1)

Authors: Margarida Romero, George Kalmpourtzis

Metacognition is an important aspect in creative problem solving (CPS) and
through this chapter we analyse the meta-reasoning aspects applied in the
different processes of monitoring the progress of learners' reasoning and CPS
activities. Meta-reasoning monitors the way that problem-solving processes
advance and regulate time and efforts towards a solution. In the context of an
ill-defined problem, exploration is required to develop a better-defined
problem space and advance towards the solution space. The way learners engage
in exploration and exploitations is regulated by the meta-reasoning within the
CPS activity. The objective of this chapter is to examine and identify the CPS
process with educational robots through a metacognitive and interactionist
approach. This chapter presents a case study, where, to solve a problem, a
participant had to explore a set of robot cubes to develop the technological
knowledge associated with each single component of the system, but also
conceptualize a system-level behaviour of the cubes when they are assembled.
The chapter presents the emergence of knowledge through the metacognitive
regulation of the process of exploration and exploitation of prior knowledge
and emergent knowledge until finding a solution

### 5. [AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study](http://arxiv.org/pdf/2508.05156v1)

Authors: Nikolaos Avouris

This paper focuses on AI tutors in foreign language learning, a field of
application of AI tutors with great development, especially during the last
years, when great advances in natural language understanding and processing in
real time, have been achieved. These tutors attempt to address needs for
improving language skills (speaking, or communicative competence,
understanding). In this paper, a mixed-methos empirical study on the use of
different kinds of state-of-the-art AI tutors for language learning is
reported. This study involves a user experience evaluation of typical such
tools, with special focus in their conversation functionality and an evaluation
of their quality, based on chat transcripts. This study can help establish
criteria for assessing the quality of such systems and inform the design of
future tools, including concerns about data privacy and secure handling of
learner information.

### 6. [A Methodological Framework and Questionnaire for Investigating Perceived Algorithmic Fairness](http://arxiv.org/pdf/2508.05281v1)

Authors: Ahmed Abdal Shafi Rasel, Ahmed Mustafa Amlan, Tasmim Shajahan Mim, Tanvir Hasan

This study explores perceptions of fairness in algorithmic decision-making
among users in Bangladesh through a comprehensive mixed-methods approach. By
integrating quantitative survey data with qualitative interview insights, we
examine how cultural, social, and contextual factors influence users'
understanding of fairness, transparency, and accountability in AI systems. Our
findings reveal nuanced attitudes toward human oversight, explanation
mechanisms, and contestability, highlighting the importance of culturally aware
design principles for equitable and trustworthy algorithmic systems. These
insights contribute to ongoing discussions on algorithmic fairness by
foregrounding perspectives from a non-Western context, thus broadening the
global dialogue on ethical AI deployment.

### 7. [Critical Design Strategy: a Method for Heuristically Evaluating Visualisation Designs](http://arxiv.org/pdf/2508.05325v1)

Authors: Jonathan C. Roberts, Hanan Alnjar, Aron E. Owen, Panagiotis D. Ritsos

We present the Critical Design Strategy (CDS) - a structured method designed
to facilitate the examination of visualisation designs through reflection and
critical thought. The CDS helps designers think critically and make informed
improvements using heuristic evaluation. When developing a visual tool or
pioneering a novel visualisation approach, identifying areas for enhancement
can be challenging. Critical thinking is particularly crucial for visualisation
designers and tool developers, especially those new to the field, such as
studying visualisation in higher education. The CDS consists of three stages
across six perspectives: Stage 1 captures the essence of the idea by assigning
an indicative title and selecting five adjectives (from twenty options) to form
initial impressions of the design. Stage 2 involves an in-depth critique using
30 heuristic questions spanning six key perspectives - user, environment,
interface, components, design, and visual marks. Stage 3 focuses on
synthesising insights, reflecting on design decisions, and determining the next
steps forward. We introduce the CDS and explore its use across three
visualisation modules in both undergraduate and postgraduate courses. Our
longstanding experience with the CDS has allowed us to refine and develop it
over time: from its initial creation through workshops in 2017/18 to
improvements in wording and the development of two applications by 2020,
followed by the expansion of support notes and refinement of heuristics through
2023; while using it in our teaching each year. This sustained use allows us to
reflect on its practical application and offer guidance on how others can
incorporate it into their own work.

### 8. [Implementation and Application of Multi-Format 3D Data Integration in a Cross-Device Commercial Metaverse Platform](http://arxiv.org/pdf/2508.05332v1)

Authors: Masanori Ibara, Yuichi Hiroi, Takushi Kamegai, Takefumi Hiraki

Traditionally, specialized 3D design data, such as BIM and CAD, have been
accessible only to a select group of experts, creating significant barriers
that prevent general users from participating in decision-making processes.
This paper provides a systematic overview of practical insights for utilizing
3D data in industrial and architectural domains by presenting implementation
cases of the industrial metaverse on Cluster, a commercial cross-device
metaverse platform. This paper analyzes the characteristics and constraints of
major data formats in the industrial and architectural fields and organizes
integration workflows for the metaverse. Through application cases utilizing 3D
data across multiple domains, we present practical examples of collaborative
decision-making support enabled by the fusion of metaverse and digital twin
technologies. Specifically, we demonstrate that multi-device access and
simultaneous multi-user participation capabilities foster democratic
environments in the industrial metaverse, which are challenging to achieve with
conventional, expert-dependent systems.

### 9. [Discrepancy-Aware Contrastive Adaptation in Medical Time Series Analysis](http://arxiv.org/pdf/2508.05572v1)

Authors: Yifan Wang, Hongfeng Ai, Ruiqi Li, Maowei Jiang, Ruiyuan Kang, Jiahua Dong, Cheng Jiang, Chenzhong Li

In medical time series disease diagnosis, two key challenges are identified.
First, the high annotation cost of medical data leads to overfitting in models
trained on label-limited, single-center datasets. To address this, we propose
incorporating external data from related tasks and leveraging AE-GAN to extract
prior knowledge, providing valuable references for downstream tasks. Second,
many existing studies employ contrastive learning to derive more generalized
medical sequence representations for diagnostic tasks, usually relying on
manually designed diverse positive and negative sample pairs. However, these
approaches are complex, lack generalizability, and fail to adaptively capture
disease-specific features across different conditions. To overcome this, we
introduce LMCF (Learnable Multi-views Contrastive Framework), a framework that
integrates a multi-head attention mechanism and adaptively learns
representations from different views through inter-view and intra-view
contrastive learning strategies. Additionally, the pre-trained AE-GAN is used
to reconstruct discrepancies in the target data as disease probabilities, which
are then integrated into the contrastive learning process. Experiments on three
target datasets demonstrate that our method consistently outperforms other
seven baselines, highlighting its significant impact on healthcare applications
such as the diagnosis of myocardial infarction, Alzheimer's disease, and
Parkinson's disease. We release the source code at xxxxx.

### 10. [Will You Be Aware? Eye Tracking-Based Modeling of Situational Awareness in Augmented Reality](http://arxiv.org/pdf/2508.05025v1)

Authors: Zhehan Qu, Tianyi Hu, Christian Fronk, Maria Gorlatova

Augmented Reality (AR) systems, while enhancing task performance through
real-time guidance, pose risks of inducing cognitive tunneling-a hyperfocus on
virtual content that compromises situational awareness (SA) in safety-critical
scenarios. This paper investigates SA in AR-guided cardiopulmonary
resuscitation (CPR), where responders must balance effective compressions with
vigilance to unpredictable hazards (e.g., patient vomiting). We developed an AR
app on a Magic Leap 2 that overlays real-time CPR feedback (compression depth
and rate) and conducted a user study with simulated unexpected incidents (e.g.,
bleeding) to evaluate SA, in which SA metrics were collected via observation
and questionnaires administered during freeze-probe events. Eye tracking
analysis revealed that higher SA levels were associated with greater saccadic
amplitude and velocity, and with reduced proportion and frequency of fixations
on virtual content. To predict SA, we propose FixGraphPool, a graph neural
network that structures gaze events (fixations, saccades) into spatiotemporal
graphs, effectively capturing dynamic attentional patterns. Our model achieved
83.0% accuracy (F1=81.0%), outperforming feature-based machine learning and
state-of-the-art time-series models by leveraging domain knowledge and
spatial-temporal information encoded in ET data. These findings demonstrate the
potential of eye tracking for SA modeling in AR and highlight its utility in
designing AR systems that ensure user safety and situational awareness.

### Information Retrieval

### 1. [An End-to-End Multi-objective Ensemble Ranking Framework for Video Recommendation](http://arxiv.org/pdf/2508.05093v1)

Authors: Tiantian He, Minzhi Xie, Runtong Li, Xiaoxiao Xu, Jiaqi Yu, Zixiu Wang, Lantao Hu, Han Li, Kun Gai

We propose a novel End-to-end Multi-objective Ensemble Ranking framework
(EMER) for the multi-objective ensemble ranking module, which is the most
critical component of the short video recommendation system. EMER enhances
personalization by replacing manually-designed heuristic formulas with an
end-to-end modeling paradigm. EMER introduces a meticulously designed loss
function to address the fundamental challenge of defining effective supervision
for ensemble ranking, where no single ground-truth signal can fully capture
user satisfaction. Moreover, EMER introduces novel sample organization method
and transformer-based network architecture to capture the comparative
relationships among candidates, which are critical for effective ranking.
Additionally, we have proposed an offline-online consistent evaluation system
to enhance the efficiency of offline model optimization, which is an
established yet persistent challenge within the multi-objective ranking domain
in industry. Abundant empirical tests are conducted on a real industrial
dataset, and the results well demonstrate the effectiveness of our proposed
framework. In addition, our framework has been deployed in the primary
scenarios of Kuaishou, a short video recommendation platform with hundreds of
millions of daily active users, achieving a 1.39% increase in overall App Stay
Time and a 0.196% increase in 7-day user Lifetime(LT7), which are substantial
improvements.

### 2. [FIRE: Faithful Interpretable Recommendation Explanations](http://arxiv.org/pdf/2508.05225v1)

Authors: S. M. F. Sani, Asal Meskin, Mohammad Amanlou, Hamid R. Rabiee

Natural language explanations in recommender systems are often framed as a
review generation task, leveraging user reviews as ground-truth supervision.
While convenient, this approach conflates a user's opinion with the system's
reasoning, leading to explanations that may be fluent but fail to reflect the
true logic behind recommendations. In this work, we revisit the core objective
of explainable recommendation: to transparently communicate why an item is
recommended by linking user needs to relevant item features. Through a
comprehensive analysis of existing methods across multiple benchmark datasets,
we identify common limitations-explanations that are weakly aligned with model
predictions, vague or inaccurate in identifying user intents, and overly
repetitive or generic. To overcome these challenges, we propose FIRE, a
lightweight and interpretable framework that combines SHAP-based feature
attribution with structured, prompt-driven language generation. FIRE produces
faithful, diverse, and user-aligned explanations, grounded in the actual
decision-making process of the model. Our results demonstrate that FIRE not
only achieves competitive recommendation accuracy but also significantly
improves explanation quality along critical dimensions such as alignment,
structure, and faithfulness. This work highlights the need to move beyond the
review-as-explanation paradigm and toward explanation methods that are both
accountable and interpretable.

### 3. [Difference Views for Visual Graph Query Building](http://arxiv.org/pdf/2508.05314v1)

Authors: Benedikt Kantz, Stefan Lengauer, Peter Waldert, Tobias Schreck

Knowledge Graphs (KGs) contain vast amounts of linked resources that encode
knowledge in various domains, which can be queried and searched for using
specialized languages like SPARQL, a query language developed to query KGs.
Existing visual query builders enable non-expert users to construct SPARQL
queries and utilize the knowledge contained in these graphs. Query building is,
however, an iterative and, often, visual process where the question of the user
can change and differ throughout the process, especially for explorative
search. Our visual querying interface communicates these change between
iterative steps in the query building process using graph differences to
contrast the changes and the evolution in the graph query. We also enable users
to formulate their evolving information needs using a natural language
interface directly integrated into the difference query view. We, furthermore,
communicate the change in results in the result view by contrasting the
differences in both result distribution and individual instances of the
prototype graph and demonstrate the system's applicability through case studies
on different ontologies and usage scenarios, illustrating how our system
fosters, both, data exploration and analysis of domain-specific graphs.

### 4. [RankArena: A Unified Platform for Evaluating Retrieval, Reranking and RAG with Human and LLM Feedback](http://arxiv.org/pdf/2508.05512v1)

Authors: Abdelrahman Abdallah, Mahmoud Abdalla, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, Adam Jatowt

Evaluating the quality of retrieval-augmented generation (RAG) and document
reranking systems remains challenging due to the lack of scalable,
user-centric, and multi-perspective evaluation tools. We introduce RankArena, a
unified platform for comparing and analysing the performance of retrieval
pipelines, rerankers, and RAG systems using structured human and LLM-based
feedback as well as for collecting such feedback. RankArena supports multiple
evaluation modes: direct reranking visualisation, blind pairwise comparisons
with human or LLM voting, supervised manual document annotation, and end-to-end
RAG answer quality assessment. It captures fine-grained relevance feedback
through both pairwise preferences and full-list annotations, along with
auxiliary metadata such as movement metrics, annotation time, and quality
ratings. The platform also integrates LLM-as-a-judge evaluation, enabling
comparison between model-generated rankings and human ground truth annotations.
All interactions are stored as structured evaluation datasets that can be used
to train rerankers, reward models, judgment agents, or retrieval strategy
selectors. Our platform is publicly available at https://rankarena.ngrok.io/,
and the Demo video is provided https://youtu.be/jIYAP4PaSSI.

### 5. [A Metric for MLLM Alignment in Large-scale Recommendation](http://arxiv.org/pdf/2508.04963v1)

Authors: Yubin Zhang, Yanhua Huang, Haiming Xu, Mingliang Qi, Chang Wang, Jiarui Jin, Xiangyuan Ren, Xiaodan Wang, Ruiwen Xu

Multimodal recommendation has emerged as a critical technique in modern
recommender systems, leveraging content representations from advanced
multimodal large language models (MLLMs). To ensure these representations are
well-adapted, alignment with the recommender system is essential. However,
evaluating the alignment of MLLMs for recommendation presents significant
challenges due to three key issues: (1) static benchmarks are inaccurate
because of the dynamism in real-world applications, (2) evaluations with online
system, while accurate, are prohibitively expensive at scale, and (3)
conventional metrics fail to provide actionable insights when learned
representations underperform. To address these challenges, we propose the
Leakage Impact Score (LIS), a novel metric for multimodal recommendation.
Rather than directly assessing MLLMs, LIS efficiently measures the upper bound
of preference data. We also share practical insights on deploying MLLMs with
LIS in real-world scenarios. Online A/B tests on both Content Feed and Display
Ads of Xiaohongshu's Explore Feed production demonstrate the effectiveness of
our proposed method, showing significant improvements in user spent time and
advertiser value.

### 6. [Data-Aware Socratic Query Refinement in Database Systems](http://arxiv.org/pdf/2508.05061v1)

Authors: Ruiyuan Zhang, Chrysanthi Kosyfaki, Xiaofang Zhou

In this paper, we propose Data-Aware Socratic Guidance (DASG), a
dialogue-based query enhancement framework that embeds \linebreak interactive
clarification as a first-class operator within database systems to resolve
ambiguity in natural language queries. DASG treats dialogue as an optimization
decision, asking clarifying questions only when the expected execution cost
reduction exceeds the interaction overhead. The system quantifies ambiguity
through linguistic fuzziness, schema grounding confidence, and projected costs
across relational and vector backends. Our algorithm selects the optimal
clarifications by combining semantic relevance, catalog-based information gain,
and potential cost reduction. We evaluate our proposed framework on three
datasets. The results show that DASG demonstrates improved query precision
while maintaining efficiency, establishing a cooperative analytics paradigm
where systems actively participate in query formulation rather than passively
translating user requests.

### 7. [Align-for-Fusion: Harmonizing Triple Preferences via Dual-oriented Diffusion for Cross-domain Sequential Recommendation](http://arxiv.org/pdf/2508.05074v1)

Authors: Yongfu Zha, Xinxin Dong, Haokai Ma, Yonghui Yang, Xiaodong Wang

Personalized sequential recommendation aims to predict appropriate items for
users based on their behavioral sequences. To alleviate data sparsity and
interest drift issues, conventional approaches typically incorporate auxiliary
behaviors from other domains via cross-domain transition. However, existing
cross-domain sequential recommendation (CDSR) methods often follow an
align-then-fusion paradigm that performs representation-level alignment across
multiple domains and combines them mechanically for recommendation, overlooking
the fine-grained fusion of domain-specific preferences. Inspired by recent
advances in diffusion models (DMs) for distribution matching, we propose an
align-for-fusion framework for CDSR to harmonize triple preferences via
dual-oriented DMs, termed HorizonRec. Specifically, we investigate the
uncertainty injection of DMs and identify stochastic noise as a key source of
instability in existing DM-based recommenders. To address this, we introduce a
mixed-conditioned distribution retrieval strategy that leverages distributions
retrieved from users' authentic behavioral logic as semantic bridges across
domains, enabling consistent multi-domain preference modeling. Furthermore, we
propose a dual-oriented preference diffusion method to suppress potential noise
and emphasize target-relevant interests during multi-domain user representation
fusion. Extensive experiments on four CDSR datasets from two distinct platforms
demonstrate the effectiveness and robustness of HorizonRec in fine-grained
triple-domain preference fusion.

### 8. [Community-Aware Social Community Recommendation](http://arxiv.org/pdf/2508.05107v1)

Authors: Runhao Jiang, Renchi Yang, Wenqing Lin

Social recommendation, which seeks to leverage social ties among users to
alleviate the sparsity issue of user-item interactions, has emerged as a
popular technique for elevating personalized services in recommender systems.
Despite being effective, existing social recommendation models are mainly
devised for recommending regular items such as blogs, images, and products, and
largely fail for community recommendations due to overlooking the unique
characteristics of communities. Distinctly, communities are constituted by
individuals, who present high dynamicity and relate to rich structural patterns
in social networks. To our knowledge, limited research has been devoted to
comprehensively exploiting this information for recommending communities.
  To bridge this gap, this paper presents CASO, a novel and effective model
specially designed for social community recommendation. Under the hood, CASO
harnesses three carefully-crafted encoders for user embedding, wherein two of
them extract community-related global and local structures from the social
network via social modularity maximization and social closeness aggregation,
while the third one captures user preferences using collaborative filtering
with observed user-community affiliations. To further eliminate feature
redundancy therein, we introduce a mutual exclusion between social and
collaborative signals. Finally, CASO includes a community detection loss in the
model optimization, thereby producing community-aware embeddings for
communities. Our extensive experiments evaluating CASO against nine strong
baselines on six real-world social networks demonstrate its consistent and
remarkable superiority over the state of the art in terms of community
recommendation performance.

### 9. [Navigating Through Paper Flood: Advancing LLM-based Paper Evaluation through Domain-Aware Retrieval and Latent Reasoning](http://arxiv.org/pdf/2508.05129v1)

Authors: Wuqiang Zheng, Yiyan Xu, Xinyu Lin, Chongming Gao, Wenjie Wang, Fuli Feng

With the rapid and continuous increase in academic publications, identifying
high-quality research has become an increasingly pressing challenge. While
recent methods leveraging Large Language Models (LLMs) for automated paper
evaluation have shown great promise, they are often constrained by outdated
domain knowledge and limited reasoning capabilities. In this work, we present
PaperEval, a novel LLM-based framework for automated paper evaluation that
addresses these limitations through two key components: 1) a domain-aware paper
retrieval module that retrieves relevant concurrent work to support
contextualized assessments of novelty and contributions, and 2) a latent
reasoning mechanism that enables deep understanding of complex motivations and
methodologies, along with comprehensive comparison against concurrently related
work, to support more accurate and reliable evaluation. To guide the reasoning
process, we introduce a progressive ranking optimization strategy that
encourages the LLM to iteratively refine its predictions with an emphasis on
relative comparison. Experiments on two datasets demonstrate that PaperEval
consistently outperforms existing methods in both academic impact and paper
quality evaluation. In addition, we deploy PaperEval in a real-world paper
recommendation system for filtering high-quality papers, which has gained
strong engagement on social media -- amassing over 8,000 subscribers and
attracting over 10,000 views for many filtered high-quality papers --
demonstrating the practical effectiveness of PaperEval.

### 10. [Tool Graph Retriever: Exploring Dependency Graph-based Tool Retrieval for Large Language Models](http://arxiv.org/pdf/2508.05152v1)

Authors: Linfeng Gao, Yaoxiang Wang, Minlong Peng, Jialong Tang, Yuzhe Shang, Mingming Sun, Jinsong Su

With the remarkable advancement of AI agents, the number of their equipped
tools is increasing rapidly. However, integrating all tool information into the
limited model context becomes impractical, highlighting the need for efficient
tool retrieval methods. In this regard, dominant methods primarily rely on
semantic similarities between tool descriptions and user queries to retrieve
relevant tools. However, they often consider each tool independently,
overlooking dependencies between tools, which may lead to the omission of
prerequisite tools for successful task execution. To deal with this defect, in
this paper, we propose Tool Graph Retriever (TGR), which exploits the
dependencies among tools to learn better tool representations for retrieval.
First, we construct a dataset termed TDI300K to train a discriminator for
identifying tool dependencies. Then, we represent all candidate tools as a tool
dependency graph and use graph convolution to integrate the dependencies into
their representations. Finally, these updated tool representations are employed
for online retrieval. Experimental results on several commonly used datasets
show that our TGR can bring a performance improvement to existing dominant
methods, achieving SOTA performance. Moreover, in-depth analyses also verify
the importance of tool dependencies and the effectiveness of our TGR.

### Machine Learning

### 1. [Self-Error Adjustment: Theory and Practice of Balancing Individual Performance and Diversity in Ensemble Learning](http://arxiv.org/pdf/2508.04948v1)

Authors: Rui Zou

Ensemble learning boosts performance by aggregating predictions from multiple
base learners. A core challenge is balancing individual learner accuracy with
diversity. Traditional methods like Bagging and Boosting promote diversity
through randomness but lack precise control over the accuracy-diversity
trade-off. Negative Correlation Learning (NCL) introduces a penalty to manage
this trade-off but suffers from loose theoretical bounds and limited adjustment
range. To overcome these limitations, we propose a novel framework called
Self-Error Adjustment (SEA), which decomposes ensemble errors into two distinct
components: individual performance terms, representing the self-error of each
base learner, and diversity terms, reflecting interactions among learners. This
decomposition allows us to introduce an adjustable parameter into the loss
function, offering precise control over the contribution of each component,
thus enabling finer regulation of ensemble performance. Compared to NCL and its
variants, SEA provides a broader range of effective adjustments and more
consistent changes in diversity. Furthermore, we establish tighter theoretical
bounds for adjustable ensemble methods and validate them through empirical
experiments. Experimental results on several public regression and
classification datasets demonstrate that SEA consistently outperforms baseline
methods across all tasks. Ablation studies confirm that SEA offers more
flexible adjustment capabilities and superior performance in fine-tuning
strategies.

### 2. [Disentangling Bias by Modeling Intra- and Inter-modal Causal Attention for Multimodal Sentiment Analysis](http://arxiv.org/pdf/2508.04999v1)

Authors: Menghua Jiang, Yuxia Lin, Baoliang Chen, Haifeng Hu, Yuncheng Jiang, Sijie Mai

Multimodal sentiment analysis (MSA) aims to understand human emotions by
integrating information from multiple modalities, such as text, audio, and
visual data. However, existing methods often suffer from spurious correlations
both within and across modalities, leading models to rely on statistical
shortcuts rather than true causal relationships, thereby undermining
generalization. To mitigate this issue, we propose a Multi-relational
Multimodal Causal Intervention (MMCI) model, which leverages the backdoor
adjustment from causal theory to address the confounding effects of such
shortcuts. Specifically, we first model the multimodal inputs as a
multi-relational graph to explicitly capture intra- and inter-modal
dependencies. Then, we apply an attention mechanism to separately estimate and
disentangle the causal features and shortcut features corresponding to these
intra- and inter-modal relations. Finally, by applying the backdoor adjustment,
we stratify the shortcut features and dynamically combine them with the causal
features to encourage MMCI to produce stable predictions under distribution
shifts. Extensive experiments on several standard MSA datasets and
out-of-distribution (OOD) test sets demonstrate that our method effectively
suppresses biases and improves performance.

### 3. [TANGO: Graph Neural Dynamics via Learned Energy and Tangential Flows](http://arxiv.org/pdf/2508.05070v1)

Authors: Moshe Eliasof, Eldad Haber, Carola-Bibiane Schönlieb

We introduce TANGO -- a dynamical systems inspired framework for graph
representation learning that governs node feature evolution through a learned
energy landscape and its associated descent dynamics. At the core of our
approach is a learnable Lyapunov function over node embeddings, whose gradient
defines an energy-reducing direction that guarantees convergence and stability.
To enhance flexibility while preserving the benefits of energy-based dynamics,
we incorporate a novel tangential component, learned via message passing, that
evolves features while maintaining the energy value. This decomposition into
orthogonal flows of energy gradient descent and tangential evolution yields a
flexible form of graph dynamics, and enables effective signal propagation even
in flat or ill-conditioned energy regions, that often appear in graph learning.
Our method mitigates oversquashing and is compatible with different graph
neural network backbones. Empirically, TANGO achieves strong performance across
a diverse set of node and graph classification and regression benchmarks,
demonstrating the effectiveness of jointly learned energy functions and
tangential flows for graph neural networks.

### 4. [ULU: A Unified Activation Function](http://arxiv.org/pdf/2508.05073v1)

Authors: Simin Huo

We propose \textbf{ULU}, a novel non-monotonic, piecewise activation function
defined as $\{f(x;\alpha_1),x<0; f(x;\alpha_2),x>=0 \}$, where
$f(x;\alpha)=0.5x(tanh(\alpha x)+1),\alpha >0$. ULU treats positive and
negative inputs differently. Extensive experiments demonstrate ULU
significantly outperforms ReLU and Mish across image classification and object
detection tasks. Its variant Adaptive ULU (\textbf{AULU}) is expressed as
$\{f(x;\beta_1^2),x<0; f(x;\beta_2^2),x>=0 \}$, where $\beta_1$ and $\beta_2$
are learnable parameters, enabling it to adapt its response separately for
positive and negative inputs. Additionally, we introduce the LIB (Like
Inductive Bias) metric from AULU to quantitatively measure the inductive bias
of the model.

### 5. [Cold Start Active Preference Learning in Socio-Economic Domains](http://arxiv.org/pdf/2508.05090v1)

Authors: Mojtaba Fayaz-Bakhsh, Danial Ataee, MohammadAmin Fazli

Active preference learning is a powerful paradigm for efficiently modeling
preferences, yet it suffers from the cold-start problem: a significant drop in
performance when no initial labeled data is available. This challenge is
particularly acute in computational social systems and economic analysis, where
labeled data is often scarce, expensive, and subject to expert noise. To
address this gap, we propose a novel framework for cold-start active preference
learning. Our method initiates the learning process through a self-supervised
pre-training phase, utilizing Principal Component Analysis (PCA) to derive
initial pseudo-labels from the data's inherent structure, thereby creating a
cold-start model without any initial oracle interaction. Subsequently, the
model is refined through an active learning loop that strategically queries a
simulated noisy oracle for labels. We conduct extensive experiments on diverse
datasets from different domains, including financial credibility, career
success rate, and socio-economic status. The results demonstrate that our
cold-start approach outperforms standard active learning strategies that begin
from a blank slate, achieving higher accuracy with substantially fewer labeled
pairs. Our framework offers a practical and effective solution to mitigate the
cold-start problem, enhancing the sample efficiency and applicability of
preference learning in data-constrained environments. We release our code at
https://github.com/Dan-A2/cold-start-preference-learning

### 6. [Learning from Similarity-Confidence and Confidence-Difference](http://arxiv.org/pdf/2508.05108v1)

Authors: Tomoya Tate, Kosuke Sugiyama, Masato Uchida

In practical machine learning applications, it is often challenging to assign
accurate labels to data, and increasing the number of labeled instances is
often limited. In such cases, Weakly Supervised Learning (WSL), which enables
training with incomplete or imprecise supervision, provides a practical and
effective solution. However, most existing WSL methods focus on leveraging a
single type of weak supervision. In this paper, we propose a novel WSL
framework that leverages complementary weak supervision signals from multiple
relational perspectives, which can be especially valuable when labeled data is
limited. Specifically, we introduce SconfConfDiff Classification, a method that
integrates two distinct forms of weaklabels: similarity-confidence and
confidence-difference, which are assigned to unlabeled data pairs. To implement
this method, we derive two types of unbiased risk estimators for
classification: one based on a convex combination of existing estimators, and
another newly designed by modeling the interaction between two weak labels. We
prove that both estimators achieve optimal convergence rates with respect to
estimation error bounds. Furthermore, we introduce a risk correction approach
to mitigate overfitting caused by negative empirical risk, and provide
theoretical analysis on the robustness of the proposed method against
inaccurate class prior probability and label noise. Experimental results
demonstrate that the proposed method consistently outperforms existing
baselines across a variety of settings.

### 7. [PSEO: Optimizing Post-hoc Stacking Ensemble Through Hyperparameter Tuning](http://arxiv.org/pdf/2508.05144v1)

Authors: Beicheng Xu, Wei Liu, Keyao Ding, Yupeng Lu, Bin Cui

The Combined Algorithm Selection and Hyperparameter Optimization (CASH)
problem is fundamental in Automated Machine Learning (AutoML). Inspired by the
success of ensemble learning, recent AutoML systems construct post-hoc
ensembles for final predictions rather than relying on the best single model.
However, while most CASH methods conduct extensive searches for the optimal
single model, they typically employ fixed strategies during the ensemble phase
that fail to adapt to specific task characteristics. To tackle this issue, we
propose PSEO, a framework for post-hoc stacking ensemble optimization. First,
we conduct base model selection through binary quadratic programming, with a
trade-off between diversity and performance. Furthermore, we introduce two
mechanisms to fully realize the potential of multi-layer stacking. Finally,
PSEO builds a hyperparameter space and searches for the optimal post-hoc
ensemble strategy within it. Empirical results on 80 public datasets show that
\sys achieves the best average test rank (2.96) among 16 methods, including
post-hoc designs in recent AutoML systems and state-of-the-art ensemble
learning methods.

### 8. [pFedDSH: Enabling Knowledge Transfer in Personalized Federated Learning through Data-free Sub-Hypernetwork](http://arxiv.org/pdf/2508.05157v1)

Authors: Thinh Nguyen, Le Huy Khiem, Van-Tuan Tran, Khoa D Doan, Nitesh V Chawla, Kok-Seng Wong

Federated Learning (FL) enables collaborative model training across
distributed clients without sharing raw data, offering a significant privacy
benefit. However, most existing Personalized Federated Learning (pFL) methods
assume a static client participation, which does not reflect real-world
scenarios where new clients may continuously join the federated system (i.e.,
dynamic client onboarding). In this paper, we explore a practical scenario in
which a new batch of clients is introduced incrementally while the learning
task remains unchanged. This dynamic environment poses various challenges,
including preserving performance for existing clients without retraining and
enabling efficient knowledge transfer between client batches. To address these
issues, we propose Personalized Federated Data-Free Sub-Hypernetwork (pFedDSH),
a novel framework based on a central hypernetwork that generates personalized
models for each client via embedding vectors. To maintain knowledge stability
for existing clients, pFedDSH incorporates batch-specific masks, which activate
subsets of neurons to preserve knowledge. Furthermore, we introduce a data-free
replay strategy motivated by DeepInversion to facilitate backward transfer,
enhancing existing clients' performance without compromising privacy. Extensive
experiments conducted on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate
that pFedDSH outperforms the state-of-the-art pFL and Federated Continual
Learning baselines in our investigation scenario. Our approach achieves robust
performance stability for existing clients, as well as adaptation for new
clients and efficient utilization of neural resources.

### 9. [S$^2$M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection](http://arxiv.org/pdf/2508.05164v1)

Authors: Jiaqi Wang, Zhengyu Ma, Xiongri Shen, Chenlin Zhou, Leilei Zhao, Han Zhang, Yi Zhong, Siqi Cai, Zhenxi Song, Zhiguo Zhang

Auditory attention detection (AAD) aims to decode listeners' focus in complex
auditory environments from electroencephalography (EEG) recordings, which is
crucial for developing neuro-steered hearing devices. Despite recent
advancements, EEG-based AAD remains hindered by the absence of synergistic
frameworks that can fully leverage complementary EEG features under
energy-efficiency constraints. We propose S$^2$M-Former, a novel spiking
symmetric mixing framework to address this limitation through two key
innovations: i) Presenting a spike-driven symmetric architecture composed of
parallel spatial and frequency branches with mirrored modular design,
leveraging biologically plausible token-channel mixers to enhance complementary
learning across branches; ii) Introducing lightweight 1D token sequences to
replace conventional 3D operations, reducing parameters by 14.7$\times$. The
brain-inspired spiking architecture further reduces power consumption,
achieving a 5.8$\times$ energy reduction compared to recent ANN methods, while
also surpassing existing SNN baselines in terms of parameter efficiency and
performance. Comprehensive experiments on three AAD benchmarks (KUL, DTU and
AV-GC-AAD) across three settings (within-trial, cross-trial and cross-subject)
demonstrate that S$^2$M-Former achieves comparable state-of-the-art (SOTA)
decoding accuracy, making it a promising low-power, high-performance solution
for AAD tasks.

### 10. [Human Activity Recognition from Smartphone Sensor Data for Clinical Trials](http://arxiv.org/pdf/2508.05175v1)

Authors: Stefania Russo, Rafał Klimas, Marta Płonka, Hugo Le Gall, Sven Holm, Dimitar Stanev, Florian Lipsmeier, Mattia Zanon, Lito Kriara

We developed a ResNet-based human activity recognition (HAR) model with
minimal overhead to detect gait versus non-gait activities and everyday
activities (walking, running, stairs, standing, sitting, lying, sit-to-stand
transitions). The model was trained and evaluated using smartphone sensor data
from adult healthy controls (HC) and people with multiple sclerosis (PwMS) with
Expanded Disability Status Scale (EDSS) scores between 0.0-6.5. Datasets
included the GaitLab study (ISRCTN15993728), an internal Roche dataset, and
publicly available data sources (training only). Data from 34 HC and 68 PwMS
(mean [SD] EDSS: 4.7 [1.5]) were included in the evaluation. The HAR model
showed 98.4% and 99.6% accuracy in detecting gait versus non-gait activities in
the GaitLab and Roche datasets, respectively, similar to a comparative
state-of-the-art ResNet model (99.3% and 99.4%). For everyday activities, the
proposed model not only demonstrated higher accuracy than the state-of-the-art
model (96.2% vs 91.9%; internal Roche dataset) but also maintained high
performance across 9 smartphone wear locations (handbag, shopping bag,
crossbody bag, backpack, hoodie pocket, coat/jacket pocket, hand, neck, belt),
outperforming the state-of-the-art model by 2.8% - 9.0%. In conclusion, the
proposed HAR model accurately detects everyday activities and shows high
robustness to various smartphone wear locations, demonstrating its practical
applicability.

### Neural and Evolutionary Computing

### 1. [Discovering Interpretable Programmatic Policies via Multimodal LLM-assisted Evolutionary Search](http://arxiv.org/pdf/2508.05433v1)

Authors: Qinglong Hu, Xialiang Tong, Mingxuan Yuan, Fei Liu, Zhichao Lu, Qingfu Zhang

Interpretability and high performance are essential goals in designing
control policies, particularly for safety-critical tasks. Deep reinforcement
learning has greatly enhanced performance, yet its inherent lack of
interpretability often undermines trust and hinders real-world deployment. This
work addresses these dual challenges by introducing a novel approach for
programmatic policy discovery, called Multimodal Large Language Model-assisted
Evolutionary Search (MLES). MLES utilizes multimodal large language models as
policy generators, combining them with evolutionary mechanisms for automatic
policy optimization. It integrates visual feedback-driven behavior analysis
within the policy generation process to identify failure patterns and
facilitate targeted improvements, enhancing the efficiency of policy discovery
and producing adaptable, human-aligned policies. Experimental results show that
MLES achieves policy discovery capabilities and efficiency comparable to
Proximal Policy Optimization (PPO) across two control tasks, while offering
transparent control logic and traceable design processes. This paradigm
overcomes the limitations of predefined domain-specific languages, facilitates
knowledge transfer and reuse, and is scalable across various control tasks.
MLES shows promise as a leading approach for the next generation of
interpretable control policy discovery.

### 2. [Harmonic fractal transformation for modeling complex neuronal effects: from bursting and noise shaping to waveform sensitivity and noise-induced subthreshold spiking](http://arxiv.org/pdf/2508.05341v1)

Authors: Mariia Sorokina

We propose the first fractal frequency mapping, which in a simple form
enables to replicate complex neuronal effects. Unlike the conventional filters,
which suppress or amplify the input spectral components according to the filter
weights, the transformation excites novel components by a fractal recomposition
of the input spectra resulting in a formation of spikes at resonant frequencies
that are optimal for sampling. This enables high sensitivity detection,
robustness to noise and noise-induced signal amplification. The proposed model
illustrates that a neuronal functionality can be viewed as a linear summation
of spectrum over nonlinearly transformed frequency domain.

### 3. [Echo State Networks for Bitcoin Time Series Prediction](http://arxiv.org/pdf/2508.05416v1)

Authors: Mansi Sharma, Enrico Sartor, Marc Cavazza, Helmut Prendinger

Forecasting stock and cryptocurrency prices is challenging due to high
volatility and non-stationarity, influenced by factors like economic changes
and market sentiment. Previous research shows that Echo State Networks (ESNs)
can effectively model short-term stock market movements, capturing nonlinear
patterns in dynamic data. To the best of our knowledge, this work is among the
first to explore ESNs for cryptocurrency forecasting, especially during extreme
volatility. We also conduct chaos analysis through the Lyapunov exponent in
chaotic periods and show that our approach outperforms existing machine
learning methods by a significant margin. Our findings are consistent with the
Lyapunov exponent analysis, showing that ESNs are robust during chaotic periods
and excel under high chaos compared to Boosting and Na\"ive methods.

### 4. [TrajEvo: Trajectory Prediction Heuristics Design via LLM-driven Evolution](http://arxiv.org/pdf/2508.05616v1)

Authors: Zhikai Zhao, Chuanbo Hua, Federico Berto, Kanghoon Lee, Zihan Ma, Jiachen Li, Jinkyoo Park

Trajectory prediction is a critical task in modeling human behavior,
especially in safety-critical domains such as social robotics and autonomous
vehicle navigation. Traditional heuristics based on handcrafted rules often
lack accuracy and generalizability. Although deep learning approaches offer
improved performance, they typically suffer from high computational cost,
limited explainability, and, importantly, poor generalization to
out-of-distribution (OOD) scenarios. In this paper, we introduce TrajEvo, a
framework that leverages Large Language Models (LLMs) to automatically design
trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to
generate and refine prediction heuristics from past trajectory data. We propose
two key innovations: Cross-Generation Elite Sampling to encourage population
diversity, and a Statistics Feedback Loop that enables the LLM to analyze and
improve alternative predictions. Our evaluations demonstrate that TrajEvo
outperforms existing heuristic methods across multiple real-world datasets, and
notably surpasses both heuristic and deep learning methods in generalizing to
an unseen OOD real-world dataset. TrajEvo marks a promising step toward the
automated design of fast, explainable, and generalizable trajectory prediction
heuristics. We release our source code to facilitate future research at
https://github.com/ai4co/trajevo.

### Networking and Internet Architecture

### 1. [Modular Design and Experimental Evaluation of 5G Mobile Cell Architectures Based on Overlay and Integrated Models](http://arxiv.org/pdf/2508.05249v1)

Authors: José Ruela, Ivan Cojocaru, André Coelho, Rui Campos, Manuel Ricardo

This paper presents the concept, architectural design, and performance
evaluation of a 5G Mobile Cell (MC) used to provide 5G wireless connectivity to
User Equipment (UE) in areas with limited fixed 5G infrastructures or subject
to adverse radio conditions. We consider two main approaches to MC design: an
overlay model, where the MC obtains backhaul connectivity from a 5G overlay
network, and an Integrated Access and Backhaul (IAB)-based model, discussing
their protocol stacks and architectural implications. In order to validate the
MC's performance, we employ an emulation-based testbed using the
OpenAirInterface (OAI) implementation, considering different MC positions. The
results validate the MC concept and demonstrate that MC positioning
significantly influences network performance. This paper has the potential to
aid network operators and service providers in selecting and deploying MC
architectures for temporary coverage extension and capacity reinforcement in
different environments, including seaports, industrial scenarios, and public
safety.

### 2. [A Design for an Early Quantum Network](http://arxiv.org/pdf/2508.04967v1)

Authors: Yuan Li, Chen Zhang, Hao Zhang, Tao Huang, Yunjie Liu

With the rapid advancement of quantum information technology, quantum
networks have become essential for supporting diverse applications, which often
have stringent demands for key metrics such as fidelity and request completion
time. In this work, we propose a design for early-stage quantum networks that
is compatible with the three existing quantum repeater technologies. The design
aims to maximize the ability of the network to accommodate the diverse needs of
quantum applications, even under conditions of limited quantum resources and
suboptimal network performance. We have also described the required identifiers
in the quantum network and the specific process for implementing quantum
requests. To assess the feasibility of our design, we conduct simulations based
on discrete-event modeling of quantum networks. The simulations consider
various types of noise and imperfect parameters that might exist in early-stage
networks. We analyze the impact of these parameters on the fidelity of the
generated entangled states and the request completion time. Furthermore, we
investigated additional decisions that the central controller can make beyond
path selection, such as the choice of cutoff time and the allocation of network
resources to requests.

### 3. [TeraRIS NOMA-MIMO Communications for 6G and Beyond Industrial Networks](http://arxiv.org/pdf/2508.05130v1)

Authors: Ali Raza, Muhammad Farhan Khan, Zeeshan Alam, Muhammad Saad, Ilyas Saleem, Muhammad Ahmed Mohsin, Muhammad Ali Jamshed

This paper presents a joint framework that integrates reconfigurable
intelligent surfaces (RISs) with Terahertz (THz) communications and
non-orthogonal multiple access (NOMA) to enhance smart industrial
communications. The proposed system leverages the advantages of RIS and THz
bands to improve spectral efficiency, coverage, and reliability key
requirements for industrial automation and real-time communications in future
6G networks and beyond. Within this framework, two power allocation strategies
are investigated: the first optimally distributes power between near and far
industrial nodes, and the second prioritizes network demands to enhance system
performance further. A performance evaluation is conducted to compare the sum
rate and outage probability against a fixed power allocation scheme. Our scheme
achieves up to a 23% sum rate gain over fixed PA at 30 dBm. Simulation results
validate the theoretical analysis, demonstrating the effectiveness and
robustness of the RIS-assisted NOMA MIMO framework for THz enabled industrial
communications.

### Robotics

### 1. [Optimal Planning for Multi-Robot Simultaneous Area and Line Coverage Using Hierarchical Cyclic Merging Regulation](http://arxiv.org/pdf/2508.04981v1)

Authors: Tianyuan Zheng, Jingang Yi, Kaiyan Yu

The double coverage problem focuses on determining efficient, collision-free
routes for multiple robots to simultaneously cover linear features (e.g.,
surface cracks or road routes) and survey areas (e.g., parking lots or local
regions) in known environments. In these problems, each robot carries two
functional roles: service (linear feature footprint coverage) and exploration
(complete area coverage). Service has a smaller operational footprint but
incurs higher costs (e.g., time) compared to exploration. We present optimal
planning algorithms for the double coverage problems using hierarchical cyclic
merging regulation (HCMR). To reduce the complexity for optimal planning
solutions, we analyze the manifold attachment process during graph traversal
from a Morse theory perspective. We show that solutions satisfying minimum path
length and collision-free constraints must belong to a Morse-bounded
collection. To identify this collection, we introduce the HCMR algorithm. In
HCMR, cyclic merging search regulates traversal behavior, while edge sequence
back propagation converts these regulations into graph edge traversal
sequences. Incorporating balanced partitioning, the optimal sequence is
selected to generate routes for each robot. We prove the optimality of the HCMR
algorithm under a fixed sweep direction. The multi-robot simulation results
demonstrate that the HCMR algorithm significantly improves planned path length
by at least 10.0%, reduces task time by at least 16.9% in average, and ensures
conflict-free operation compared to other state-of-the-art planning methods.

### 2. [MAG-Nav: Language-Driven Object Navigation Leveraging Memory-Reserved Active Grounding](http://arxiv.org/pdf/2508.05021v1)

Authors: Weifan Zhang, Tingguang Li, Yuzhen Liu

Visual navigation in unknown environments based solely on natural language
descriptions is a key capability for intelligent robots. In this work, we
propose a navigation framework built upon off-the-shelf Visual Language Models
(VLMs), enhanced with two human-inspired mechanisms: perspective-based active
grounding, which dynamically adjusts the robot's viewpoint for improved visual
inspection, and historical memory backtracking, which enables the system to
retain and re-evaluate uncertain observations over time. Unlike existing
approaches that passively rely on incidental visual inputs, our method actively
optimizes perception and leverages memory to resolve ambiguity, significantly
improving vision-language grounding in complex, unseen environments. Our
framework operates in a zero-shot manner, achieving strong generalization to
diverse and open-ended language descriptions without requiring labeled data or
model fine-tuning. Experimental results on Habitat-Matterport 3D (HM3D) show
that our method outperforms state-of-the-art approaches in language-driven
object navigation. We further demonstrate its practicality through real-world
deployment on a quadruped robot, achieving robust and effective navigation
performance.

### 3. [Benchmarking Shortcutting Techniques for Multi-Robot-Arm Motion Planning](http://arxiv.org/pdf/2508.05027v1)

Authors: Philip Huang, Yorai Shaoul, Jiaoyang Li

Generating high-quality motion plans for multiple robot arms is challenging
due to the high dimensionality of the system and the potential for inter-arm
collisions. Traditional motion planning methods often produce motions that are
suboptimal in terms of smoothness and execution time for multi-arm systems.
Post-processing via shortcutting is a common approach to improve motion quality
for efficient and smooth execution. However, in multi-arm scenarios, optimizing
one arm's motion must not introduce collisions with other arms. Although
existing multi-arm planning works often use some form of shortcutting
techniques, their exact methodology and impact on performance are often vaguely
described. In this work, we present a comprehensive study quantitatively
comparing existing shortcutting methods for multi-arm trajectories across
diverse simulated scenarios. We carefully analyze the pros and cons of each
shortcutting method and propose two simple strategies for combining these
methods to achieve the best performance-runtime tradeoff. Video, code, and
dataset are available at https://philip-huang.github.io/mr-shortcut/.

### 4. [A Vision-Based Collision Sensing Method for Stable Circular Object Grasping with A Soft Gripper System](http://arxiv.org/pdf/2508.05040v1)

Authors: Boyang Zhang, Jiahui Zuo, Zeyu Duan, Fumin Zhang

External collisions to robot actuators typically pose risks to grasping
circular objects. This work presents a vision-based sensing module capable of
detecting collisions to maintain stable grasping with a soft gripper system.
The system employs an eye-in-palm camera with a broad field of view to
simultaneously monitor the motion of fingers and the grasped object.
Furthermore, we have developed a collision-rich grasping strategy to ensure the
stability and security of the entire dynamic grasping process. A physical soft
gripper was manufactured and affixed to a collaborative robotic arm to evaluate
the performance of the collision detection mechanism. An experiment regarding
testing the response time of the mechanism confirmed the system has the
capability to react to the collision instantaneously. A dodging test was
conducted to demonstrate the gripper can detect the direction and scale of
external collisions precisely.

### 5. [Examining the legibility of humanoid robot arm movements in a pointing task](http://arxiv.org/pdf/2508.05104v1)

Authors: Andrej Lúčny, Matilde Antonj, Carlo Mazzola, Hana Hornáčková, Ana Farić, Kristína Malinovská, Michal Vavrecka, Igor Farkaš

Human--robot interaction requires robots whose actions are legible, allowing
humans to interpret, predict, and feel safe around them. This study
investigates the legibility of humanoid robot arm movements in a pointing task,
aiming to understand how humans predict robot intentions from truncated
movements and bodily cues. We designed an experiment using the NICO humanoid
robot, where participants observed its arm movements towards targets on a
touchscreen. Robot cues varied across conditions: gaze, pointing, and pointing
with congruent or incongruent gaze. Arm trajectories were stopped at 60\% or
80\% of their full length, and participants predicted the final target. We
tested the multimodal superiority and ocular primacy hypotheses, both of which
were supported by the experiment.

### 6. [From Canada to Japan: How 10,000 km Affect User Perception in Robot Teleoperation](http://arxiv.org/pdf/2508.05143v1)

Authors: Siméon Capy, Thomas M. Kwok, Kevin Joseph, Yuichiro Kawasumi, Koichi Nagashima, Tomoya Sasaki, Yue Hu, Eiichi Yoshida

Robot teleoperation (RTo) has emerged as a viable alternative to local
control, particularly when human intervention is still necessary. This research
aims to study the distance effect on user perception in RTo, exploring the
potential of teleoperated robots for older adult care. We propose an evaluation
of non-expert users' perception of long-distance RTo, examining how their
perception changes before and after interaction, as well as comparing it to
that of locally operated robots. We have designed a specific protocol
consisting of multiple questionnaires, along with a dedicated software
architecture using the Robotics Operating System (ROS) and Unity. The results
revealed no statistically significant differences between the local and remote
robot conditions, suggesting that robots may be a viable alternative to
traditional local control.

### 7. [GhostShell: Streaming LLM Function Calls for Concurrent Embodied Programming](http://arxiv.org/pdf/2508.05298v1)

Authors: Jian Gong, Youwei Huang, Bo Yuan, Ming Zhu, Juncheng Zhan, Jinke Wang, Hang Shu, Mingyue Xiong, Yanjun Ye, Yufan Zu, Yang Zhou, Yihan Ding, Xuannian Chen, Xingyu Lu, Runjie Ban, Bingchao Huang, Fusen Liu

We present GhostShell, a novel approach that leverages Large Language Models
(LLMs) to enable streaming and concurrent behavioral programming for embodied
systems. In contrast to conventional methods that rely on pre-scheduled action
sequences or behavior trees, GhostShell drives embodied systems to act
on-the-fly by issuing function calls incrementally as tokens are streamed from
the LLM. GhostShell features a streaming XML function token parser, a dynamic
function interface mapper, and a multi-channel scheduler that orchestrates
intra-channel synchronous and inter-channel asynchronous function calls,
thereby coordinating serial-parallel embodied actions across multiple robotic
components as directed by the LLM. We evaluate GhostShell on our robot
prototype COCO through comprehensive grounded experiments across 34 real-world
interaction tasks and multiple LLMs. The results demonstrate that our approach
achieves state-of-the-art Behavioral Correctness Metric of 0.85 with Claude-4
Sonnet and up to 66X faster response times compared to LLM native function
calling APIs. GhostShell also proves effective in long-horizon multimodal
tasks, demonstrating strong robustness and generalization.

### 8. [Affecta-Context: The Context-Guided Behavior Adaptation Framework](http://arxiv.org/pdf/2508.05359v1)

Authors: Morten Roed Frederiksen, Kasper Støy

This paper presents Affecta-context, a general framework to facilitate
behavior adaptation for social robots. The framework uses information about the
physical context to guide its behaviors in human-robot interactions. It
consists of two parts: one that represents encountered contexts and one that
learns to prioritize between behaviors through human-robot interactions. As
physical contexts are encountered the framework clusters them by their measured
physical properties. In each context, the framework learns to prioritize
between behaviors to optimize the physical attributes of the robot's behavior
in line with its current environment and the preferences of the users it
interacts with. This paper illlustrates the abilities of the Affecta-context
framework by enabling a robot to autonomously learn the prioritization of
discrete behaviors. This was achieved by training across 72 interactions in two
different physical contexts with 6 different human test participants. The paper
demonstrates the trained Affecta-context framework by verifying the robot's
ability to generalize over the input and to match its behaviors to a previously
unvisited physical context.

### 9. [Robots can defuse high-intensity conflict situations](http://arxiv.org/pdf/2508.05373v1)

Authors: Morten Roed Frederiksen, Kasper Støy

This paper investigates the specific scenario of high-intensity
confrontations between humans and robots, to understand how robots can defuse
the conflict. It focuses on the effectiveness of using five different affective
expression modalities as main drivers for defusing the conflict. The aim is to
discover any strengths or weaknesses in using each modality to mitigate the
hostility that people feel towards a poorly performing robot. The defusing of
the situation is accomplished by making the robot better at acknowledging the
conflict and by letting it express remorse. To facilitate the tests, we used a
custom affective robot in a simulated conflict situation with 105 test
participants. The results show that all tested expression modalities can
successfully be used to defuse the situation and convey an acknowledgment of
the confrontation. The ratings were remarkably similar, but the movement
modality was different (ANON p$<$.05) than the other modalities. The test
participants also had similar affective interpretations on how impacted the
robot was of the confrontation across all expression modalities. This indicates
that defusing a high-intensity interaction may not demand special attention to
the expression abilities of the robot, but rather require attention to the
abilities of being socially aware of the situation and reacting in accordance
with it.

### 10. [Computational Design and Fabrication of Modular Robots with Untethered Control](http://arxiv.org/pdf/2508.05410v1)

Authors: Manas Bhargava, Takefumi Hiraki, Malina Strugaru, Michal Piovarci, Chiara Daraio, Daisuke Iwai, Bernd Bickel

Natural organisms use distributed actuation via their musculoskeletal systems
to adapt their gait for traversing diverse terrains or to morph their bodies to
perform varied tasks. A longstanding challenge in the field of robotics is to
mimic this extensive adaptability and range of motion. This has led humans to
develop various soft robotic systems that emulate natural organisms. However,
such systems are generally optimized for a single functionality, lack the
ability to change form or function on demand, or are often tethered to bulky
control systems. To address these challenges, we present our framework for
designing and controlling robots that mimic nature's blueprint by utilizing
distributed actuation. We propose a novel building block that combines
3D-printed bones with liquid crystal elastomer (LCE) muscles as lightweight
actuators and enables the modular assembly of musculoskeletal robots. We
developed LCE rods that contract in response to infrared radiation, thereby
achieving local and untethered control over the distributed network of bones,
which in turn results in global deformation of the robot. Furthermore, to
capitalize on the extensive design space, we develop two computational tools:
one to optimize the robot's skeletal graph, enabling multiple target
deformations, and another to co-optimize the skeletal designs and control gaits
to achieve target locomotion. We validate our system by building several robots
that show complex shape morphing, varying control schemes, and adaptability to
their environment. Our system integrates advances in modular material building,
untethered and distributed control, and computational design to introduce a new
generation of robots that brings us closer to the capabilities of living
organisms.

### Software Engineering

### 1. [Generative AI for Object-Oriented Programming: Writing the Right Code and Reasoning the Right Logic](http://arxiv.org/pdf/2508.05005v1)

Authors: Gang Xu, Airong Wang, Yushan Pan

We find ourselves in the midst of an explosion in artificial intelligence
research, particularly with large language models (LLMs). These models have
diverse applications spanning finance, commonsense knowledge graphs, medicine,
and visual analysis. In the world of Object-Oriented Programming(OOP), a robust
body of knowledge and methods has been developed for managing complex tasks
through object-oriented thinking. However, the intersection of LLMs with OOP
remains an underexplored territory. Empirically, we currently possess limited
understanding of how LLMs can enhance the effectiveness of OOP learning and
code writing, as well as how we can evaluate such AI-powered tools. Our work
aims to address this gap by presenting a vision from the perspectives of key
stakeholders involved in an OOP task: programmers, mariners, and experienced
programmers. We identify critical junctures within typical coding workflows
where the integration of LLMs can offer significant benefits. Furthermore, we
propose ways to augment existing logical reasoning and code writing, ultimately
enhancing the programming experience.

### 2. [LadyBug: A GitHub Bot for UI-Enhanced Bug Localization in Mobile Apps](http://arxiv.org/pdf/2508.05085v1)

Authors: Junayed Mahmud, James Chen, Terry Achille, Camilo Alvarez-Velez, Darren Dean Bansil, Patrick Ijieh, Samar Karanch, Nadeeshan De Silva, Oscar Chaparro, Andrian Marcus, Kevin Moran

This paper introduces LadyBug, a GitHub bot that automatically localizes bugs
for Android apps by combining UI interaction information with text retrieval.
LadyBug connects to an Android app's GitHub repository, and is triggered when a
bug is reported in the corresponding issue tracker. Developers can then record
a reproduction trace for the bug on a device or emulator and upload the trace
to LadyBug via the GitHub issue tracker. This enables LadyBug to utilize both
the text from the original bug description, and UI information from the
reproduction trace to accurately retrieve a ranked list of files from the
project that most likely contain the reported bug.
  We empirically evaluated LadyBug using an automated testing pipeline and
benchmark called RedWing that contains 80 fully-localized and reproducible bug
reports from 39 Android apps. Our results illustrate that LadyBug outperforms
text-retrieval-based baselines and that the utilization of UI information leads
to a substantial increase in localization accuracy. LadyBug is an open-source
tool, available at https://github.com/LadyBugML/ladybug.
  A video showing the capabilities of Ladybug can be viewed here:
https://youtu.be/hI3tzbRK0Cw

### 3. [AI-assisted JSON Schema Creation and Mapping](http://arxiv.org/pdf/2508.05192v1)

Authors: Felix Neubauer, Jürgen Pleiss, Benjamin Uekermann

Model-Driven Engineering (MDE) places models at the core of system and data
engineering processes. In the context of research data, these models are
typically expressed as schemas that define the structure and semantics of
datasets. However, many domains still lack standardized models, and creating
them remains a significant barrier, especially for non-experts. We present a
hybrid approach that combines large language models (LLMs) with deterministic
techniques to enable JSON Schema creation, modification, and schema mapping
based on natural language inputs by the user. These capabilities are integrated
into the open-source tool MetaConfigurator, which already provides visual model
editing, validation, code generation, and form generation from models. For data
integration, we generate schema mappings from heterogeneous JSON, CSV, XML, and
YAML data using LLMs, while ensuring scalability and reliability through
deterministic execution of generated mapping rules. The applicability of our
work is demonstrated in an application example in the field of chemistry. By
combining natural language interaction with deterministic safeguards, this work
significantly lowers the barrier to structured data modeling and data
integration for non-experts.

### 4. [STEPWISE-CODEX-Bench: Evaluating Complex Multi-Function Comprehension and Fine-Grained Execution Reasoning](http://arxiv.org/pdf/2508.05193v1)

Authors: Kaiwen Yan, Yuhang Chang, Zirui Guo, Yaling Mou, Jiang Ming, Jingwei Sun

In recent years, large language models (LLMs) have made significant progress
in code intelligence, yet systematically evaluating their code understanding
and reasoning abilities remains challenging. Mainstream benchmarks such as
HumanEval and MBPP primarily assess functional correctness, while reasoning
benchmarks like CRUXEVAL are limited to single-function, low-complexity
scenarios. As a result, advanced models achieve nearly saturated scores,
limiting their discriminative power. To address this, we present
STEPWISE-CODEX-Bench (SX-Bench), a novel benchmark designed for complex
multi-function understanding and fine-grained execution reasoning. SX-Bench
features tasks involving collaboration among multiple sub-functions (e.g.,
chained calls, nested loops), shifting evaluation towards overall control and
data flow modeling. It defines "computation steps" as the minimal execution
unit and requires models to predict the total number of steps in reasoning
tasks, thereby assessing a model's in-depth understanding of dynamic execution
beyond simple I/O matching. Evaluation on over 20 mainstream models (including
14 reasoning-enhanced models) demonstrates that SX-Bench is highly
discriminative: even the state-of-the-art OpenAI-O3 achieves only 78.37 percent
accuracy on Hard-Reasoning tasks, much lower than its saturated scores on
previous benchmarks, thereby revealing bottlenecks in complex and fine-grained
reasoning. We also release an automated pipeline combining program synthesis,
symbolic execution, and LLM-aided validation for efficient benchmark generation
and quality assurance. SX-Bench advances code evaluation from "single-function
verification" to "multi-function dynamic reasoning," providing a key tool for
the in-depth assessment of advanced code intelligence models.

### 5. [An ML-based Approach to Predicting Software Change Dependencies: Insights from an Empirical Study on OpenStack](http://arxiv.org/pdf/2508.05034v1)

Authors: Arabat, Ali, Sayagh, Mohammed, Hassine, Jameleddine

As software systems grow in complexity, accurately identifying and managing
dependencies among changes becomes increasingly critical. For instance, a
change that leverages a function must depend on the change that introduces it.
Establishing such dependencies allows CI/CD pipelines to build and orchestrate
changes effectively, preventing build failures and incomplete feature
deployments. In modern software systems, dependencies often span multiple
components across teams, creating challenges for development and deployment.
They serve various purposes, from enabling new features to managing
configurations, and can even involve traditionally independent changes like
documentation updates. To address these challenges, we conducted a preliminary
study on dependency management in OpenStack, a large-scale software system. Our
study revealed that a substantial portion of software changes in OpenStack over
the past 10 years are interdependent. Surprisingly, 51.08% of these
dependencies are identified during the code review phase-after a median delay
of 5.06 hours-rather than at the time of change creation. Developers often
spend a median of 57.12 hours identifying dependencies, searching among a
median of 463 other changes. To help developers proactively identify
dependencies, we propose a semi-automated approach that leverages two ML
models. The first model predicts the likelihood of dependencies among changes,
while the second identifies the exact pairs of dependent changes. Our proposed
models demonstrate strong performance, achieving average AUC scores of 79.33%
and 91.89%, and Brier scores of 0.11 and 0.014, respectively. Indeed, the
second model has a good top-k recall across all types of pairs, while the top-k
precision has room for improvement.

### 6. [EvoGraph: Hybrid Directed Graph Evolution toward Software 3.0](http://arxiv.org/pdf/2508.05199v1)

Authors: Igor Costa, Christopher Baran

We introduce **EvoGraph**, a framework that enables software systems to
evolve their own source code, build pipelines, documentation, and tickets.
EvoGraph represents every artefact in a typed directed graph, applies learned
mutation operators driven by specialized small language models (SLMs), and
selects survivors with a multi-objective fitness. On three benchmarks, EvoGraph
fixes 83% of known security vulnerabilities, translates COBOL to Java with 93%
functional equivalence (test verified), and maintains documentation freshness
within two minutes. Experiments show a 40% latency reduction and a sevenfold
drop in feature lead time compared with strong baselines. We extend our
approach to **evoGraph**, leveraging language-specific SLMs for modernizing
.NET, Lisp, CGI, ColdFusion, legacy Python, and C codebases, achieving 82-96%
semantic equivalence across languages while reducing computational costs by 90%
compared to large language models. EvoGraph's design responds to empirical
failure modes in legacy modernization, such as implicit contracts, performance
preservation, and integration evolution. Our results suggest a practical path
toward Software 3.0, where systems adapt continuously yet remain under
measurable control.

### 7. [A Conceptual Model and Methodology for Sustainability-aware, IoT-enhanced Business Processes](http://arxiv.org/pdf/2508.05301v1)

Authors: Victoria Torres Bosch, Ronny Seiger, Manuela Albert Albiol, Antoni Mestre Gascon, Pedro Jose Valderas Aranda

The real-time data collection and automation capabilities offered by the
Internet of Things (IoT) are revolutionizing and transforming Business
Processes (BPs) into IoT-enhanced BPs, showing high potential for improving
sustainability. Although already studied in Business Process Management (BPM),
sustainability research has primarily focused on environmental concerns.
However, achieving a holistic and lasting impact requires a systematic approach
to address sustainability beyond the environmental dimension. This work
proposes a conceptual model and a structured methodology with the goal of
analyzing the potential of IoT to measure and improve the sustainability of
BPs. The conceptual model formally represents key sustainability concepts,
linking BPM and IoT by highlighting how IoT devices support and contribute to
sustainability. The methodology guides the systematic analysis of existing BPs,
identifies opportunities, and implements sustainability-aware, IoT-enhanced
BPs. The approach is illustrated through a running example from the tourism
domain and a case study in healthcare.

### 8. [Posterior-GRPO: Rewarding Reasoning Processes in Code Generation](http://arxiv.org/pdf/2508.05170v1)

Authors: Lishui Fan, Yu Zhang, Mouxiang Chen, Zhongxin Liu

Reinforcement learning (RL) has significantly advanced code generation for
large language models (LLMs). However, current paradigms rely on outcome-based
rewards from test cases, neglecting the quality of the intermediate reasoning
process. While supervising the reasoning process directly is a promising
direction, it is highly susceptible to reward hacking, where the policy model
learns to exploit the reasoning reward signal without improving final outcomes.
To address this, we introduce a unified framework that can effectively
incorporate the quality of the reasoning process during RL. First, to enable
reasoning evaluation, we develop LCB-RB, a benchmark comprising preference
pairs of superior and inferior reasoning processes. Second, to accurately score
reasoning quality, we introduce an Optimized-Degraded based (OD-based) method
for reward model training. This method generates high-quality preference pairs
by systematically optimizing and degrading initial reasoning paths along
curated dimensions of reasoning quality, such as factual accuracy, logical
rigor, and coherence. A 7B parameter reward model with this method achieves
state-of-the-art (SOTA) performance on LCB-RB and generalizes well to other
benchmarks. Finally, we introduce Posterior-GRPO (P-GRPO), a novel RL method
that conditions process-based rewards on task success. By selectively applying
rewards to the reasoning processes of only successful outcomes, P-GRPO
effectively mitigates reward hacking and aligns the model's internal reasoning
with final code correctness. A 7B parameter model with P-GRPO achieves superior
performance across diverse code generation tasks, outperforming outcome-only
baselines by 4.5%, achieving comparable performance to GPT-4-Turbo. We further
demonstrate the generalizability of our approach by extending it to
mathematical tasks. Our models, dataset, and code are publicly available.

### 9. [Everything You Need to Know About CS Education: Open Results from a Survey of More Than 18,000 Participants](http://arxiv.org/pdf/2508.05286v1)

Authors: Katsiaryna Dzialets, Aleksandra Makeeva, Ilya Vlasov, Anna Potriasaeva, Aleksei Rostovskii, Yaroslav Golubev, Anastasiia Birillo

Computer science education is a dynamic field with many aspects that
influence the learner's path. While these aspects are usually studied in depth
separately, it is also important to carry out broader large-scale studies that
touch on many topics, because they allow us to put different results into each
other's perspective. Past large-scale surveys have provided valuable insights,
however, the emergence of new trends (e.g., AI), new learning formats (e.g.,
in-IDE learning), and the increasing learner diversity highlight the need for
an updated comprehensive study. To address this, we conducted a survey with
18,032 learners from 173 countries, ensuring diverse representation and
exploring a wide range of topics - formal education, learning formats, AI
usage, challenges, motivation, and more. This paper introduces the results of
this survey as an open dataset, describes our methodology and the survey
questions, and highlights, as a motivating example, three possible research
directions within this data: challenges in learning, emerging formats, and
insights into the in-IDE format. The dataset aims to support further research
and foster advancements in computer education.

### Social and Information Networks

### 1. [Modeling roles and trade-offs in multiplex networks](http://arxiv.org/pdf/2508.05488v1)

Authors: Nikolaos Nakis, Sune Lehmann, Nicholas A. Christakis, Morten Mørup

A multiplex social network captures multiple types of social relations among
the same set of people, with each layer representing a distinct type of
relationship. Understanding the structure of such systems allows us to identify
how social exchanges may be driven by a person's own attributes and actions
(independence), the status or resources of others (dependence), and mutual
influence between entities (interdependence). Characterizing structure in
multiplex networks is challenging, as the distinct layers can reflect different
yet complementary roles, with interdependence emerging across multiple scales.
Here, we introduce the Multiplex Latent Trade-off Model (MLT), a framework for
extracting roles in multiplex social networks that accounts for independence,
dependence, and interdependence. MLT defines roles as trade-offs, requiring
each node to distribute its source and target roles across layers while
simultaneously distributing community memberships within hierarchical,
multi-scale structures. Applying the MLT approach to 176 real-world multiplex
networks, composed of social, health, and economic layers, from villages in
western Honduras, we see core social exchange principles emerging, while also
revealing local, layer-specific, and multi-scale communities. Link prediction
analyses reveal that modeling interdependence yields the greatest performance
gains in the social layer, with subtler effects in health and economic layers.
This suggests that social ties are structurally embedded, whereas health and
economic ties are primarily shaped by individual status and behavioral
engagement. Our findings offer new insights into the structure of human social
systems.

### 2. [Community-Aware Social Community Recommendation](http://arxiv.org/pdf/2508.05107v1)

Authors: Runhao Jiang, Renchi Yang, Wenqing Lin

Social recommendation, which seeks to leverage social ties among users to
alleviate the sparsity issue of user-item interactions, has emerged as a
popular technique for elevating personalized services in recommender systems.
Despite being effective, existing social recommendation models are mainly
devised for recommending regular items such as blogs, images, and products, and
largely fail for community recommendations due to overlooking the unique
characteristics of communities. Distinctly, communities are constituted by
individuals, who present high dynamicity and relate to rich structural patterns
in social networks. To our knowledge, limited research has been devoted to
comprehensively exploiting this information for recommending communities.
  To bridge this gap, this paper presents CASO, a novel and effective model
specially designed for social community recommendation. Under the hood, CASO
harnesses three carefully-crafted encoders for user embedding, wherein two of
them extract community-related global and local structures from the social
network via social modularity maximization and social closeness aggregation,
while the third one captures user preferences using collaborative filtering
with observed user-community affiliations. To further eliminate feature
redundancy therein, we introduce a mutual exclusion between social and
collaborative signals. Finally, CASO includes a community detection loss in the
model optimization, thereby producing community-aware embeddings for
communities. Our extensive experiments evaluating CASO against nine strong
baselines on six real-world social networks demonstrate its consistent and
remarkable superiority over the state of the art in terms of community
recommendation performance.

### Systems and Control

### 1. [Uncovering the Influence Flow Model of Transistor Amplifiers, Its Reconstruction and Application](http://arxiv.org/pdf/2508.04977v1)

Authors: Mohammed Tuhin Rana, Mishfad Shaikh Veedu, Murti V. Salapaka

Multistage transistor amplifiers can be effectively modeled as network of
dynamic systems where individual amplifier stages interact through couplings
that are dynamic in nature. Using circuit analysis techniques, we show that a
large class of transistor amplifiers can be modeled as Linear Dynamic Influence
Model (LDIM), where the interactions between different amplifier stages are
modeled as linear dynamic equations. LDIM modeling of transistor circuits leads
to application of data-driven network reconstruction techniques to characterize
stage interactions and identify faults and critical circuit parameters
efficiently. Employing graphical modeling techniques and Wiener filtering, we
demonstrate that the network structure can be reconstructed solely from voltage
time-series measurements sampled at specified points in the circuit. The
efficacy of these network reconstruction methods in multistage amplifiers is
demonstrated through extensive simulations involving multiple amplifier
circuits in Cadence, as well as experimental results on physical hardware. The
ability to infer network structure directly from measurement data offers
designers and users efficient tools to design, analyze, and debug amplifier
circuits. To demonstrate the utility of network reconstruction in multistage
amplifier circuits, a fault diagnosis method leveraging these techniques is
presented.

### 2. [Passive nonlinear FIR filters for data-driven control](http://arxiv.org/pdf/2508.05279v1)

Authors: Zixing Wang, Fulvio Forni

We propose a new class of passive nonlinear finite impulse response
operators. This class is constructed by the action of finite impulse response
filters in a lifted space. This allows for efficient control synthesis through
constrained optimization. Closed-loop performance is taken into account through
least-squares fitting, based on the theory of virtual reference feedback
tuning. Passivity is established through efficient linear constraints, based on
sampling in the frequency domain. Because of passivity, this class of operators
is particularly suited for the control of physical systems, such as
electromechanical systems.

### 3. [A 20-Year Retrospective on Power and Thermal Modeling and Management](http://arxiv.org/pdf/2508.05495v1)

Authors: David Atienza, Kai Zhu, Darong Huang, Luis Costero

As processor performance advances, increasing power densities and complex
thermal behaviors threaten both energy efficiency and system reliability. This
survey covers more than two decades of research on power and thermal modeling
and management in modern processors. We start by comparing analytical,
regression-based, and neural network-based techniques for power estimation,
then review thermal modeling methods, including finite element, finite
difference, and data-driven approaches. Next, we categorize dynamic runtime
management strategies that balance performance, power consumption, and
reliability. Finally, we conclude with a discussion of emerging challenges and
promising research directions.

### 4. [Research on integrated intelligent energy management system based on big data analysis and machine learning](http://arxiv.org/pdf/2508.05583v1)

Authors: Jinzhou Xu, Yadan Zhang, Paola Tapia

The application of big data is one of the significant features of integrated
smart energy. Applying it to the file management of integrated smart energy
projects is of great significance for improving the efficiency of project
management and control. This article first discussed the benefits and
challenges of implementing big data analysis in document management and control
of integrated smart energy projects. In addition, an implementation framework
for big data analysis in integrated smart energy project document management
was developed, and a method for optimizing the efficiency of integrated smart
energy project document management through machine learning was proposed. Using
various types of data and information generated during the project document
management process, the efficiency of the entire process project document
control through three different machine learning methods was optimized. The
result of fitting a penalty linear regression model shows that when there is
enough data as a training set, the accuracy of the model achieved can reach
over 95\%. By using big data analysis and machine learning to analyze the
efficiency of comprehensive smart energy project document management, it is
possible to track the entire process of comprehensive smart energy project
documents and optimize business processes, thereby strengthening project
construction control and improving project construction efficiency.

### 5. [Error Bounds for Radial Network Topology Learning from Quantized Measurements](http://arxiv.org/pdf/2508.05620v1)

Authors: Samuel Talkington, Aditya Rangarajan, Pedro A. de Alcântara, Line Roald, Daniel K. Molzahn, Daniel R. Fuhrmann

We probabilistically bound the error of a solution to a radial network
topology learning problem where both connectivity and line parameters are
estimated. In our model, data errors are introduced by the precision of the
sensors, i.e., quantization. This produces a nonlinear measurement model that
embeds the operation of the sensor communication network into the learning
problem, expanding beyond the additive noise models typically seen in power
system estimation algorithms. We show that the error of a learned radial
network topology is proportional to the quantization bin width and grows
sublinearly in the number of nodes, provided that the number of samples per
node is logarithmic in the number of nodes.

### 6. [RCUKF: Data-Driven Modeling Meets Bayesian Estimation](http://arxiv.org/pdf/2508.04985v1)

Authors: Kumar Anurag, Kasra Azizi, Francesco Sorrentino, Wenbin Wan

Accurate modeling is crucial in many engineering and scientific applications,
yet obtaining a reliable process model for complex systems is often
challenging. To address this challenge, we propose a novel framework, reservoir
computing with unscented Kalman filtering (RCUKF), which integrates data-driven
modeling via reservoir computing (RC) with Bayesian estimation through the
unscented Kalman filter (UKF). The RC component learns the nonlinear system
dynamics directly from data, serving as a surrogate process model in the UKF
prediction step to generate state estimates in high-dimensional or chaotic
regimes where nominal mathematical models may fail. Meanwhile, the UKF
measurement update integrates real-time sensor data to correct potential drift
in the data-driven model. We demonstrate RCUKF effectiveness on well-known
benchmark problems and a real-time vehicle trajectory estimation task in a
high-fidelity simulation environment.

### 7. [Probabilistic Alternating Simulations for Policy Synthesis in Uncertain Stochastic Dynamical Systems](http://arxiv.org/pdf/2508.05062v1)

Authors: Thom Badings, Alessandro Abate

A classical approach to formal policy synthesis in stochastic dynamical
systems is to construct a finite-state abstraction, often represented as a
Markov decision process (MDP). The correctness of these approaches hinges on a
behavioural relation between the dynamical system and its abstraction, such as
a probabilistic simulation relation. However, probabilistic simulation
relations do not suffice when the system dynamics are, next to being
stochastic, also subject to nondeterministic (i.e., set-valued) disturbances.
In this work, we extend probabilistic simulation relations to systems with both
stochastic and nondeterministic disturbances. Our relation, which is inspired
by a notion of alternating simulation, generalises existing relations used for
verification and policy synthesis used in several works. Intuitively, our
relation allows reasoning probabilistically over stochastic uncertainty, while
reasoning robustly (i.e., adversarially) over nondeterministic disturbances. We
experimentally demonstrate the applicability of our relations for policy
synthesis in a 4D-state Dubins vehicle.

### 8. [Preparing for the worst: Long-term and short-term weather extremes in resource adequacy assessment](http://arxiv.org/pdf/2508.05163v1)

Authors: Aleksander Grochowicz, Hannah C. Bloomfield, Marta Victoria

Security of supply is a common and important concern when integrating
renewables in net-zero power systems. Extreme weather affects both demand and
supply leading to power system stress; in Europe this stress spreads
continentally beyond the meteorological root cause. We use an approach based on
shadow prices to identify periods of elevated stress called system-defining
events and analyse their impact on the power system. By classifying different
types of system-defining events, we identify challenges to power system
operation and planning. Crucially, we find the need for sufficient resilience
back-up (power) capacities whose financial viability is precarious due to
weather variability. Furthermore, we disentangle short- and long-term
resilience challenges with distinct metrics and stress tests to incorporate
both into future energy modelling assessments. Our methodology and
implementation in the open model PyPSA-Eur can be re-applied to other systems
and help researchers and policymakers in building more resilient and adequate
energy systems.

### 9. [Overview of Controllability Definitions in Supervisory Control Theory](http://arxiv.org/pdf/2508.05177v1)

Authors: Jeroen J. A. Keiren, Michel A. Reniers

In the field of supervisory control theory, the literature often proposes
different definitions for the same concept, making it difficult to understand
how these definitions are related. This is definitely so for the fundamental
notion of controllability of a supervisor w.r.t. a plant. This paper lists
definitions of controllability found in the literature and studies their
relationships in settings of both deterministic and nondeterministic automata.
In the general context, where both the supervisor and the plant are allowed to
be nondeterministic, the notions of controllability as described by Flordal and
Malik, and uncontrollable event admissibility by Kushi and Takai are
equivalent. These are also the only notions that imply the traditional notion
of (language) controllability. From a practical perspective, one is often more
interested in controllability of a supervised plant w.r.t. a plant. In this
context, in addition to the previous two controllability notions, state
controllability by Zhou et al. implies language controllability.

### 10. [Advanced Hybrid Transformer LSTM Technique with Attention and TS Mixer for Drilling Rate of Penetration Prediction](http://arxiv.org/pdf/2508.05210v1)

Authors: Saddam Hussain Khan

The Rate of Penetration (ROP) is crucial for optimizing drilling operations;
however, accurately predicting it is hindered by the complex, dynamic, and
high-dimensional nature of drilling data. Traditional empirical, physics-based,
and basic machine learning models often fail to capture intricate temporal and
contextual relationships, resulting in suboptimal predictions and limited
real-time utility. To address this gap, we propose a novel hybrid deep learning
architecture integrating Long Short-Term Memory (LSTM) networks, Transformer
encoders, Time-Series Mixer (TS-Mixer) blocks, and attention mechanisms to
synergistically model temporal dependencies, static feature interactions,
global context, and dynamic feature importance. Evaluated on a real-world
drilling dataset, our model outperformed benchmarks (standalone LSTM, TS-Mixer,
and simpler hybrids) with an R-squared score of 0.9988 and a Mean Absolute
Percentage Error of 1.447%, as measured by standard regression metrics
(R-squared, MAE, RMSE, MAPE). Model interpretability was ensured using SHAP and
LIME, while actual vs. predicted curves and bias checks confirmed accuracy and
fairness across scenarios. This advanced hybrid approach enables reliable
real-time ROP prediction, paving the way for intelligent, cost-effective
drilling optimization systems with significant operational impact.

### Machine Learning (Statistics Category)

### 1. [Near Optimal Inference for the Best-Performing Algorithm](http://arxiv.org/pdf/2508.05173v1)

Authors: Amichai Painsky

Consider a collection of competing machine learning algorithms. Given their
performance on a benchmark of datasets, we would like to identify the best
performing algorithm. Specifically, which algorithm is most likely to rank
highest on a future, unseen dataset. A natural approach is to select the
algorithm that demonstrates the best performance on the benchmark. However, in
many cases the performance differences are marginal and additional candidates
may also be considered. This problem is formulated as subset selection for
multinomial distributions. Formally, given a sample from a countable alphabet,
our goal is to identify a minimal subset of symbols that includes the most
frequent symbol in the population with high confidence. In this work, we
introduce a novel framework for the subset selection problem. We provide both
asymptotic and finite-sample schemes that significantly improve upon currently
known methods. In addition, we provide matching lower bounds, demonstrating the
favorable performance of our proposed schemes.

### 2. [High-Dimensional Differentially Private Quantile Regression: Distributed Estimation and Statistical Inference](http://arxiv.org/pdf/2508.05212v1)

Authors: Ziliang Shen, Caixing Wang, Shaoli Wang, Yibo Yan

With the development of big data and machine learning, privacy concerns have
become increasingly critical, especially when handling heterogeneous datasets
containing sensitive personal information. Differential privacy provides a
rigorous framework for safeguarding individual privacy while enabling
meaningful statistical analysis. In this paper, we propose a differentially
private quantile regression method for high-dimensional data in a distributed
setting. Quantile regression is a powerful and robust tool for modeling the
relationships between the covariates and responses in the presence of outliers
or heavy-tailed distributions. To address the computational challenges due to
the non-smoothness of the quantile loss function, we introduce a Newton-type
transformation that reformulates the quantile regression task into an ordinary
least squares problem. Building on this, we develop a differentially private
estimation algorithm with iterative updates, ensuring both near-optimal
statistical accuracy and formal privacy guarantees. For inference, we further
propose a differentially private debiased estimator, which enables valid
confidence interval construction and hypothesis testing. Additionally, we
propose a communication-efficient and differentially private bootstrap for
simultaneous hypothesis testing in high-dimensional quantile regression,
suitable for distributed settings with both small and abundant local data.
Extensive simulations demonstrate the robustness and effectiveness of our
methods in practical scenarios.

### 3. [Periodic evaluation of defined-contribution pension fund: A dynamic risk measure approach](http://arxiv.org/pdf/2508.05241v1)

Authors: Wanting He, Wenyuan Li, Yunran Wei

This paper introduces an innovative framework for the periodic evaluation of
defined-contribution pension funds. The performance of the pension fund is
evaluated not only at retirement, but also within the interim periods. In
contrast to the traditional literature, we set the dynamic risk measure as the
criterion and manage the tail risk of the pension fund dynamically. To
effectively interact with the stochastic environment, a model-free
reinforcement learning algorithm is proposed to search for optimal investment
and insurance strategies. Using U.S. data, we calibrate pension members'
mortality rates and enhance mortality projections through a Lee-Carter model.
Our numerical results indicate that periodic evaluations lead to more
risk-averse strategies, while mortality improvements encourage more
risk-seeking behaviors.

### 4. [Negative Binomial Variational Autoencoders for Overdispersed Latent Modeling](http://arxiv.org/pdf/2508.05423v1)

Authors: Yixuan Zhang, Wenxin Zhang, Hua Jiang, Quyu Kong, Feng Zhou

Biological neurons communicate through spike trains, discrete, irregular
bursts of activity that exhibit variability far beyond the modeling capacity of
conventional variational autoencoders (VAEs). Recent work, such as the
Poisson-VAE, makes a biologically inspired move by modeling spike counts using
the Poisson distribution. However, they impose a rigid constraint: equal mean
and variance, which fails to reflect the true stochastic nature of neural
activity. In this work, we challenge this constraint and introduce NegBio-VAE,
a principled extension of the VAE framework that models spike counts using the
negative binomial distribution. This shift grants explicit control over
dispersion, unlocking a broader and more accurate family of neural
representations. We further develop two ELBO optimization schemes and two
differentiable reparameterization strategies tailored to the negative binomial
setting. By introducing one additional dispersion parameter, NegBio-VAE
generalizes the Poisson latent model to a negative binomial formulation.
Empirical results demonstrate this minor yet impactful change leads to
significant gains in reconstruction fidelity, highlighting the importance of
explicitly modeling overdispersion in spike-like activations.

### 5. [RCUKF: Data-Driven Modeling Meets Bayesian Estimation](http://arxiv.org/pdf/2508.04985v1)

Authors: Kumar Anurag, Kasra Azizi, Francesco Sorrentino, Wenbin Wan

Accurate modeling is crucial in many engineering and scientific applications,
yet obtaining a reliable process model for complex systems is often
challenging. To address this challenge, we propose a novel framework, reservoir
computing with unscented Kalman filtering (RCUKF), which integrates data-driven
modeling via reservoir computing (RC) with Bayesian estimation through the
unscented Kalman filter (UKF). The RC component learns the nonlinear system
dynamics directly from data, serving as a surrogate process model in the UKF
prediction step to generate state estimates in high-dimensional or chaotic
regimes where nominal mathematical models may fail. Meanwhile, the UKF
measurement update integrates real-time sensor data to correct potential drift
in the data-driven model. We demonstrate RCUKF effectiveness on well-known
benchmark problems and a real-time vehicle trajectory estimation task in a
high-fidelity simulation environment.

### 6. [L1-Regularized Functional Support Vector Machine](http://arxiv.org/pdf/2508.05567v1)

Authors: Bingfan Liu, Peijun Sang

In functional data analysis, binary classification with one functional
covariate has been extensively studied. We aim to fill in the gap of
considering multivariate functional covariates in classification. In
particular, we propose an $L_1$-regularized functional support vector machine
for binary classification. An accompanying algorithm is developed to fit the
classifier. By imposing an $L_1$ penalty, the algorithm enables us to identify
relevant functional covariates of the binary response. Numerical results from
simulations and one real-world application demonstrate that the proposed
classifier enjoys good performance in both prediction and feature selection.

### 7. [High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation](http://arxiv.org/pdf/2508.05570v1)

Authors: Ilya Levin, Alexey Naumov, Sergey Samsonov

In this paper, we study the bias and high-order error bounds of the Linear
Stochastic Approximation (LSA) algorithm with Polyak-Ruppert (PR) averaging
under Markovian noise. We focus on the version of the algorithm with constant
step size $\alpha$ and propose a novel decomposition of the bias via a
linearization technique. We analyze the structure of the bias and show that the
leading-order term is linear in $\alpha$ and cannot be eliminated by PR
averaging. To address this, we apply the Richardson-Romberg (RR) extrapolation
procedure, which effectively cancels the leading bias term. We derive
high-order moment bounds for the RR iterates and show that the leading error
term aligns with the asymptotically optimal covariance matrix of the vanilla
averaged LSA iterates.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-08 PST.

### 1. [MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in Large Language Models Using Odyssey Math Data](https://www.nature.com/articles/s41597-025-05283-3)

Authors: Meng Fang et al.

### 2. [ERNIE-TextCNN: research on classification methods of Chinese news headlines in different situations](https://www.nature.com/articles/s41598-025-14955-4)

Authors: Yumin Yan

### 3. [Research on GNNs with stable learning](https://www.nature.com/articles/s41598-025-12840-8)

Authors: Wenbin Li et al.

### 4. [Accelerating clinical evidence synthesis with large language models](https://www.nature.com/articles/s41746-025-01840-7)

Authors: Zifeng Wang et al.

### 5. [Scientific planning of dynamic crops in complex agricultural landscapes based on adaptive optimization hybrid SA-GA method](https://www.nature.com/articles/s41598-025-14188-5)

Authors: Changlong Li et al.

### 6. [A condition diagnosis method for subway track structures employing distributed optical fiber sensing](https://www.nature.com/articles/s41598-025-14806-2)

Authors: Hong Han et al.

