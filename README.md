# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-12 17:00:24.819035 PST.

### Artificial Intelligence

### 1. [MCPToolBench++: A Large Scale AI Agent Model Context Protocol MCP Tool Use Benchmark](http://arxiv.org/pdf/2508.07575v1)

Authors: Shiqing Fan, Xichen Ding, Liang Zhang, Linjian Mo

LLMs' capabilities are enhanced by using function calls to integrate various
data sources or API results into the context window. Typical tools include
search, web crawlers, maps, financial data, file systems, and browser usage,
etc. Integrating these data sources or functions requires a standardized
method. The Model Context Protocol (MCP) provides a standardized way to supply
context to LLMs. However, the evaluation of LLMs and AI Agents' MCP tool use
abilities suffer from several issues. First, there's a lack of comprehensive
datasets or benchmarks to evaluate various MCP tools. Second, the diverse
formats of response from MCP tool call execution further increase the
difficulty of evaluation. Additionally, unlike existing tool-use benchmarks
with high success rates in functions like programming and math functions, the
success rate of real-world MCP tool is not guaranteed and varies across
different MCP servers. Furthermore, the LLMs' context window also limits the
number of available tools that can be called in a single run, because the
textual descriptions of tool and the parameters have long token length for an
LLM to process all at once. To help address the challenges of evaluating LLMs'
performance on calling MCP tools, we propose MCPToolBench++, a large-scale,
multi-domain AI Agent tool use benchmark. As of July 2025, this benchmark is
build upon marketplace of over 4k MCP servers from more than 40 categories,
collected from the MCP marketplaces and GitHub communities. The datasets
consist of both single-step and multi-step tool calls across different
categories. We evaluated SOTA LLMs with agentic abilities on this benchmark and
reported the results.

### 2. [HGMF: A Hierarchical Gaussian Mixture Framework for Scalable Tool Invocation within the Model Context Protocol](http://arxiv.org/pdf/2508.07602v1)

Authors: Wenpeng Xing, Zhipeng Chen, Changting Lin, Meng Han

Invoking external tools enables Large Language Models (LLMs) to perform
complex, real-world tasks, yet selecting the correct tool from large,
hierarchically-structured libraries remains a significant challenge. The
limited context windows of LLMs and noise from irrelevant options often lead to
low selection accuracy and high computational costs. To address this, we
propose the Hierarchical Gaussian Mixture Framework (HGMF), a probabilistic
pruning method for scalable tool invocation. HGMF first maps the user query and
all tool descriptions into a unified semantic space. The framework then
operates in two stages: it clusters servers using a Gaussian Mixture Model
(GMM) and filters them based on the query's likelihood. Subsequently, it
applies the same GMM-based clustering and filtering to the tools associated
with the selected servers. This hierarchical process produces a compact,
high-relevance candidate set, simplifying the final selection task for the LLM.
Experiments on a public dataset show that HGMF significantly improves tool
selection accuracy while reducing inference latency, confirming the framework's
scalability and effectiveness for large-scale tool libraries.

### 3. [Multimodal AI Systems for Enhanced Laying Hen Welfare Assessment and Productivity Optimization](http://arxiv.org/pdf/2508.07628v1)

Authors: Daniel Essien, Suresh Neethirajan

The future of poultry production depends on a paradigm shift replacing
subjective, labor-intensive welfare checks with data-driven, intelligent
monitoring ecosystems. Traditional welfare assessments-limited by human
observation and single-sensor data-cannot fully capture the complex,
multidimensional nature of laying hen welfare in modern farms. Multimodal
Artificial Intelligence (AI) offers a breakthrough, integrating visual,
acoustic, environmental, and physiological data streams to reveal deeper
insights into avian welfare dynamics. This investigation highlights multimodal
As transformative potential, showing that intermediate (feature-level) fusion
strategies achieve the best balance between robustness and performance under
real-world poultry conditions, and offer greater scalability than early or late
fusion approaches. Key adoption barriers include sensor fragility in harsh farm
environments, high deployment costs, inconsistent behavioral definitions, and
limited cross-farm generalizability. To address these, we introduce two novel
evaluation tools - the Domain Transfer Score (DTS) to measure model
adaptability across diverse farm settings, and the Data Reliability Index (DRI)
to assess sensor data quality under operational constraints. We also propose a
modular, context-aware deployment framework designed for laying hen
environments, enabling scalable and practical integration of multimodal
sensing. This work lays the foundation for a transition from reactive, unimodal
monitoring to proactive, precision-driven welfare systems that unite
productivity with ethical, science based animal care.

### 4. [1-2-3 Check: Enhancing Contextual Privacy in LLM via Multi-Agent Reasoning](http://arxiv.org/pdf/2508.07667v1)

Authors: Wenkai Li, Liwen Sun, Zhenxiang Guan, Xuhui Zhou, Maarten Sap

Addressing contextual privacy concerns remains challenging in interactive
settings where large language models (LLMs) process information from multiple
sources (e.g., summarizing meetings with private and public information). We
introduce a multi-agent framework that decomposes privacy reasoning into
specialized subtasks (extraction, classification), reducing the information
load on any single agent while enabling iterative validation and more reliable
adherence to contextual privacy norms. To understand how privacy errors emerge
and propagate, we conduct a systematic ablation over information-flow
topologies, revealing when and why upstream detection mistakes cascade into
downstream leakage. Experiments on the ConfAIde and PrivacyLens benchmark with
several open-source and closed-sourced LLMs demonstrate that our best
multi-agent configuration substantially reduces private information leakage
(\textbf{18\%} on ConfAIde and \textbf{19\%} on PrivacyLens with GPT-4o) while
preserving the fidelity of public content, outperforming single-agent
baselines. These results highlight the promise of principled information-flow
design in multi-agent systems for contextual privacy with LLMs.

### 5. [\(X\)-evolve: Solution space evolution powered by large language models](http://arxiv.org/pdf/2508.07932v1)

Authors: Yi Zhai, Zhiqiang Wei, Ruohan Li, Keyu Pan, Shuo Liu, Lu Zhang, Jianmin Ji, Wuyang Zhang, Yu Zhang, Yanyong Zhang

While combining large language models (LLMs) with evolutionary algorithms
(EAs) shows promise for solving complex optimization problems, current
approaches typically evolve individual solutions, often incurring high LLM call
costs. We introduce \(X\)-evolve, a paradigm-shifting method that instead
evolves solution spaces \(X\) (sets of individual solutions) - subsets of the
overall search space \(S\). In \(X\)-evolve, LLMs generate tunable programs
wherein certain code snippets, designated as parameters, define a tunable
solution space. A score-based search algorithm then efficiently explores this
parametrically defined space, guided by feedback from objective function
scores. This strategy enables broader and more efficient exploration, which can
potentially accelerate convergence at a much lower search cost, requiring up to
two orders of magnitude fewer LLM calls than prior leading methods. We
demonstrate \(X\)-evolve's efficacy across three distinct hard optimization
problems. For the cap set problem, we discover a larger partial admissible set,
establishing a new tighter asymptotic lower bound for the cap set constant (\(C
\ge 2.2203\)). In information theory, we uncover a larger independent set for
the 15-vertex cycle graph (\(\mathcal{C}_{15}^{\boxtimes 5}\), size 19,946),
thereby raising the known lower bound on its Shannon capacity. Furthermore, for
the NP-hard online bin packing problem, we generate heuristics that
consistently outperform standard strategies across established benchmarks. By
evolving solution spaces, our method considerably improves search
effectiveness, making it possible to tackle high-dimensional problems that were
previously computationally prohibitive.

### 6. [Deep Reinforcement Learning with anticipatory reward in LSTM for Collision Avoidance of Mobile Robots](http://arxiv.org/pdf/2508.07941v1)

Authors: Olivier Poulet, Frédéric Guinand, François Guérin

This article proposes a collision risk anticipation method based on
short-term prediction of the agents position. A Long Short-Term Memory (LSTM)
model, trained on past trajectories, is used to estimate the next position of
each robot. This prediction allows us to define an anticipated collision risk
by dynamically modulating the reward of a Deep Q-Learning Network (DQN) agent.
The approach is tested in a constrained environment, where two robots move
without communication or identifiers. Despite a limited sampling frequency (1
Hz), the results show a significant decrease of the collisions number and a
stability improvement. The proposed method, which is computationally
inexpensive, appears particularly attractive for implementation on embedded
systems.

### 7. [Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths](http://arxiv.org/pdf/2508.08001v1)

Authors: Rui Yao, Qi Chai, Jinhai Yao, Siyuan Li, Junhao Chen, Qi Zhang, Hao Wang

"Fedspeak", the stylized and often nuanced language used by the U.S. Federal
Reserve, encodes implicit policy signals and strategic stances. The Federal
Open Market Committee strategically employs Fedspeak as a communication tool to
shape market expectations and influence both domestic and global economic
conditions. As such, automatically parsing and interpreting Fedspeak presents a
high-impact challenge, with significant implications for financial forecasting,
algorithmic trading, and data-driven policy analysis. In this paper, we propose
an LLM-based, uncertainty-aware framework for deciphering Fedspeak and
classifying its underlying monetary policy stance. Technically, to enrich the
semantic and contextual representation of Fedspeak texts, we incorporate
domain-specific reasoning grounded in the monetary policy transmission
mechanism. We further introduce a dynamic uncertainty decoding module to assess
the confidence of model predictions, thereby enhancing both classification
accuracy and model reliability. Experimental results demonstrate that our
framework achieves state-of-the-art performance on the policy stance analysis
task. Moreover, statistical analysis reveals a significant positive correlation
between perceptual uncertainty and model error rates, validating the
effectiveness of perceptual uncertainty as a diagnostic signal.

### 8. [Fitting Description Logic Ontologies to ABox and Query Examples](http://arxiv.org/pdf/2508.08007v1)

Authors: Maurice Funk, Marvin Grosser, Carsten Lutz

We study a fitting problem inspired by ontology-mediated querying: given a
collection
  of positive and negative examples of
  the form $(\mathcal{A},q)$ with
  $\mathcal{A}$ an ABox and $q$ a Boolean query, we seek
  an ontology $\mathcal{O}$ that satisfies $\mathcal{A} \cup \mathcal{O} \vDash
q$ for all positive examples and $\mathcal{A} \cup \mathcal{O}\not\vDash q$ for
all negative examples.
  We consider the description logics $\mathcal{ALC}$ and $\mathcal{ALCI}$ as
ontology languages and
  a range of query languages that
  includes atomic queries (AQs), conjunctive queries (CQs), and unions thereof
(UCQs).
  For all of the resulting fitting problems,
  we provide
  effective characterizations and determine the computational complexity
  of deciding whether a fitting ontology exists. This problem turns out to be
${\small CO}NP$ for AQs and full CQs
  and $2E{\small XP}T{\small IME}$-complete for CQs and UCQs.
  These results hold for both $\mathcal{ALC}$ and $\mathcal{ALCI}$.

### 9. [AdaptFlow: Adaptive Workflow Optimization via Meta-Learning](http://arxiv.org/pdf/2508.08053v1)

Authors: Runchuan Zhu, Bowen Jiang, Lingrui Mei, Fangkai Yang, Lu Wang, Haoxiang Gao, Fengshuo Bai, Pu Zhao, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang

Recent advances in large language models (LLMs) have sparked growing interest
in agentic workflows, which are structured sequences of LLM invocations
intended to solve complex tasks. However, existing approaches often rely on
static templates or manually designed workflows, which limit adaptability to
diverse tasks and hinder scalability. We propose AdaptFlow, a natural
language-based meta-learning framework inspired by model-agnostic meta-learning
(MAML). AdaptFlow learns a generalizable workflow initialization that enables
rapid subtask-level adaptation. It employs a bi-level optimization scheme: the
inner loop refines the workflow for a specific subtask using LLM-generated
feedback, while the outer loop updates the shared initialization to perform
well across tasks. This setup allows AdaptFlow to generalize effectively to
unseen tasks by adapting the initialized workflow through language-guided
modifications. Evaluated across question answering, code generation, and
mathematical reasoning benchmarks, AdaptFlow consistently outperforms both
manually crafted and automatically searched baselines, achieving
state-of-the-art results with strong generalization across tasks and models.
The source code and data are available at
https://github.com/microsoft/DKI_LLM/tree/AdaptFlow/AdaptFlow.

### 10. [FNBT: Full Negation Belief Transformation for Open-World Information Fusion Based on Dempster-Shafer Theory of Evidence](http://arxiv.org/pdf/2508.08075v1)

Authors: Meishen He, Wenjun Ma, Jiao Wang, Huijun Yue, Xiaoma Fan

The Dempster-Shafer theory of evidence has been widely applied in the field
of information fusion under uncertainty. Most existing research focuses on
combining evidence within the same frame of discernment. However, in real-world
scenarios, trained algorithms or data often originate from different regions or
organizations, where data silos are prevalent. As a result, using different
data sources or models to generate basic probability assignments may lead to
heterogeneous frames, for which traditional fusion methods often yield
unsatisfactory results. To address this challenge, this study proposes an
open-world information fusion method, termed Full Negation Belief
Transformation (FNBT), based on the Dempster-Shafer theory. More specially, a
criterion is introduced to determine whether a given fusion task belongs to the
open-world setting. Then, by extending the frames, the method can accommodate
elements from heterogeneous frames. Finally, a full negation mechanism is
employed to transform the mass functions, so that existing combination rules
can be applied to the transformed mass functions for such information fusion.
Theoretically, the proposed method satisfies three desirable properties, which
are formally proven: mass function invariance, heritability, and essential
conflict elimination. Empirically, FNBT demonstrates superior performance in
pattern classification tasks on real-world datasets and successfully resolves
Zadeh's counterexample, thereby validating its practical effectiveness.

### Hardware Architecture

### 1. [A Matrix Decomposition Method for Odd-Type Gaussian Normal Basis Multiplication](http://arxiv.org/pdf/2508.07541v1)

Authors: Kittiphon Phalakarn, Athasit Surarerks

Normal basis is used in many applications because of the efficiency of the
implementation. However, most space complexity reduction techniques for binary
field multiplier are applicable for only optimal normal basis or Gaussian
normal basis of even type. There are 187 binary fields GF(2^k) for k from 2 to
1,000 that use odd-type Gaussian normal basis. This paper presents a method to
reduce the space complexity of odd-type Gaussian normal basis multipliers over
binary field GF(2^k). The idea is adapted from the matrix decomposition method
for optimal normal basis. The result shows that our space complexity reduction
method can reduce the number of XOR gates used in the implementation comparing
to previous works with a small trade-off in critical path delay.

### 2. [ARISE: Automating RISC-V Instruction Set Extension](http://arxiv.org/pdf/2508.07725v1)

Authors: Andreas Hager-Clukas, Philipp van Kempen, Stefan Wallentowitz

RISC-V is an extendable Instruction Set Architecture, growing in popularity
for embedded systems. However, optimizing it to specific requirements, imposes
a great deal of manual effort. To bridge the gap between software and ISA, the
tool ARISE is presented. It automates the generation of RISC-V instructions
based on assembly patterns, which are selected by an extendable set of metrics.
These metrics implement the optimization goals of code size and instruction
count reduction, both statically and dynamically. The instruction set
extensions are generated using the ISA description language CoreDSL. Allowing
seamless embedding in advanced tools such as the retargeting compiler Seal5 or
the instruction set simulator ETISS. ARISE improves the static code size by
1.48% and the dynamic code size by 3.84%, as well as the number of instructions
to be executed by 7.39% on average for Embench-Iot.

### 3. [TLV-HGNN: Thinking Like a Vertex for Memory-efficient HGNN Inference](http://arxiv.org/pdf/2508.07796v1)

Authors: Dengke Han, Duo Wang, Mingyu Yan, Xiaochun Ye, Dongrui Fan

Heterogeneous graph neural networks (HGNNs) excel at processing heterogeneous
graph data and are widely applied in critical domains. In HGNN inference, the
neighbor aggregation stage is the primary performance determinant, yet it
suffers from two major sources of memory inefficiency. First, the commonly
adopted per-semantic execution paradigm stores intermediate aggregation results
for each semantic prior to semantic fusion, causing substantial memory
expansion. Second, the aggregation process incurs extensive redundant memory
accesses, including repeated loading of target vertex features across semantics
and repeated accesses to shared neighbors due to cross-semantic neighborhood
overlap. These inefficiencies severely limit scalability and reduce HGNN
inference performance.
  In this work, we first propose a semantics-complete execution paradigm from a
vertex perspective that eliminates per-semantic intermediate storage and
redundant target vertex accesses. Building on this paradigm, we design
TVL-HGNN, a reconfigurable hardware accelerator optimized for efficient
aggregation. In addition, we introduce a vertex grouping technique based on
cross-semantic neighborhood overlap, with hardware implementation, to reduce
redundant accesses to shared neighbors. Experimental results demonstrate that
TVL-HGNN achieves average speedups of 7.85x and 1.41x over the NVIDIA A100 GPU
and the state-of-the-art HGNN accelerator HiHGNN, respectively, while reducing
energy consumption by 98.79% and 32.61%.

### 4. [ELF: Efficient Logic Synthesis by Pruning Redundancy in Refactoring](http://arxiv.org/pdf/2508.08073v1)

Authors: Dimitris Tsaras, Xing Li, Lei Chen, Zhiyao Xie, Mingxuan Yuan

In electronic design automation, logic optimization operators play a crucial
role in minimizing the gate count of logic circuits. However, their computation
demands are high. Operators such as refactor conventionally form iterative cuts
for each node, striving for a more compact representation - a task which often
fails 98% on average. Prior research has sought to mitigate computational cost
through parallelization. In contrast, our approach leverages a classifier to
prune unsuccessful cuts preemptively, thus eliminating unnecessary resynthesis
operations. Experiments on the refactor operator using the EPFL benchmark suite
and 10 large industrial designs demonstrate that this technique can speedup
logic optimization by 3.9x on average compared with the state-of-the-art ABC
implementation.

### Computational Complexity

### 1. [Counting Martingales for Measure and Dimension in Complexity Classes](http://arxiv.org/pdf/2508.07619v1)

Authors: John M. Hitchcock, Adewale Sekoni, Hadi Shafei

This paper makes two primary contributions. First, we introduce the concept
of counting martingales and use it to define counting measures, counting
dimensions, and counting strong dimensions. Second, we apply these new tools to
strengthen previous circuit lower bounds.
  Resource-bounded measure and dimension have traditionally focused on
deterministic time and space bounds. We use counting complexity classes to
develop resource-bounded counting measures and dimensions. Counting martingales
are constructed using functions from the #P, SpanP, and GapP complexity
classes. We show that counting martingales capture many martingale
constructions in complexity theory. The resulting counting measures and
dimensions are intermediate in power between the standard time-bounded and
space-bounded notions, enabling finer-grained analysis where space-bounded
measures are known, but time-bounded measures remain open. For example, we show
that BPP has #P-dimension 0 and BQP has GapP-dimension 0.
  As our main application, we improve circuit-size lower bounds. Lutz (1992)
strengthened Shannon's classic $(1-\epsilon)\frac{2^n}{n}$ lower bound (1949)
to PSPACE-measure, showing that almost all problems require circuits of size
$\frac{2^n}{n}\left(1+\frac{\alpha \log n}{n}\right)$, for any $\alpha < 1$. We
extend this result to SpanP-measure, with a proof that uses a connection
through the Minimum Circuit Size Problem (MCSP) to construct a counting
martingale. Our results imply that the stronger lower bound holds within the
third level of the exponential-time hierarchy, whereas previously, it was only
known in ESPACE. We study the #P-dimension of classical circuit complexity
classes and the GapP-dimension of quantum circuit complexity classes. We also
show that if one-way functions exist, then #P-dimension is strictly more
powerful than P-dimension.

### Computational Engineering

### 1. [Semantic-Enhanced Time-Series Forecasting via Large Language Models](http://arxiv.org/pdf/2508.07697v1)

Authors: Hao Liu, Chun Yang, Zhang xiaoxing, Xiaobin Zhu

Time series forecasting plays a significant role in finance, energy,
meteorology, and IoT applications. Recent studies have leveraged the
generalization capabilities of large language models (LLMs) to adapt to time
series forecasting, achieving promising performance. However, existing studies
focus on token-level modal alignment, instead of bridging the intrinsic
modality gap between linguistic knowledge structures and time series data
patterns, greatly limiting the semantic representation. To address this issue,
we propose a novel Semantic-Enhanced LLM (SE-LLM) that explores the inherent
periodicity and anomalous characteristics of time series to embed into the
semantic space to enhance the token embedding. This process enhances the
interpretability of tokens for LLMs, thereby activating the potential of LLMs
for temporal sequence analysis. Moreover, existing Transformer-based LLMs excel
at capturing long-range dependencies but are weak at modeling short-term
anomalies in time-series data. Hence, we propose a plugin module embedded
within self-attention that models long-term and short-term dependencies to
effectively adapt LLMs to time-series analysis. Our approach freezes the LLM
and reduces the sequence dimensionality of tokens, greatly reducing
computational consumption. Experiments demonstrate the superiority performance
of our SE-LLM against the state-of-the-art (SOTA) methods.

### 2. [Material Fingerprinting: A shortcut to material model discovery without solving optimization problems](http://arxiv.org/pdf/2508.07831v1)

Authors: Moritz Flaschel, Denisa Martonová, Carina Veil, Ellen Kuhl

We propose Material Fingerprinting, a new method for the rapid discovery of
mechanical material models from direct or indirect data that avoids solving
potentially non-convex optimization problems. The core assumption of Material
Fingerprinting is that each material exhibits a unique response when subjected
to a standardized experimental setup. We can interpret this response as the
material's fingerprint, essentially a unique identifier that encodes all
pertinent information about the material's mechanical characteristics.
Consequently, once we have established a database containing fingerprints and
their corresponding mechanical models during an offline phase, we can rapidly
characterize an unseen material in an online phase. This is accomplished by
measuring its fingerprint and employing a pattern recognition algorithm to
identify the best matching fingerprint in the database. In our study, we
explore this concept in the context of hyperelastic materials, demonstrating
the applicability of Material Fingerprinting across different experimental
setups. Initially, we examine Material Fingerprinting through experiments
involving homogeneous deformation fields, which provide direct strain-stress
data pairs. We then extend this concept to experiments involving complexly
shaped specimens with heterogeneous deformation fields, which provide indirect
displacement and reaction force measurements. We show that, in both cases,
Material Fingerprinting is an efficient tool for model discovery, bypassing the
challenges of potentially non-convex optimization. We believe that Material
Fingerprinting provides a powerful and generalizable framework for rapid
material model identification across a wide range of experimental designs and
material behaviors, paving the way for numerous future developments.

### 3. [A Lagrangian method for solving the spherical shallow water equations using power diagrams](http://arxiv.org/pdf/2508.08129v1)

Authors: Philip Caplan, Otis Milliken, Toby Pouler, Zeyi Tong, Col McDermott, Sam Millay

Numerical simulations of the air in the atmosphere and water in the oceans
are essential for numerical weather prediction. The state-of-the-art for
performing these fluid simulations relies on an Eulerian viewpoint, in which
the fluid domain is discretized into a mesh, and the governing equations
describe the fluid motion as it passes through each cell of the mesh. However,
it is unclear whether a Lagrangian viewpoint, in which the fluid is discretized
by a collection of particles, can outperform Eulerian simulations in global
atmospheric simulations. To date, Lagrangian approaches have shown promise, but
tend to produce smoother solutions. In this work, a new Lagrangian method is
developed to simulate the atmosphere in which particles are represented with
spherical power cells. We introduce an efficient algorithm for computing these
cells which are then used to discretize the spherical shallow water equations.
Mass conservation is enforced by solving a semi-discrete optimal transport
problem and a semi-implicit time stepping procedure is used to advance the
solution in time. We note that, in contrast to previous work, artificial
viscosity is not needed to stabilize the simulation. The performance of the
spherical Voronoi diagram calculation is first assessed, which shows that
spherical Voronoi diagrams of 100 million sites can be computed in under 2
minutes on a single machine. The new simulation method is then evaluated on
standard benchmark test cases, which shows that momentum and energy
conservation of this new method is comparable to the latest Lagrangian approach
for simulating the spherical shallow water equations.

### Computational Geometry

### 1. [Summarizing Classed Region Maps with a Disk Choreme](http://arxiv.org/pdf/2508.07529v1)

Authors: Steven van den Broek, Wouter Meulemans, Andreas Reimer, Bettina Speckmann

Chorematic diagrams are highly reduced schematic maps of geospatial data and
processes. They can visually summarize complex situations using only a few
simple shapes (choremes) placed upon a simplified base map. Due to the extreme
reduction of data in chorematic diagrams, they tend to be produced manually;
few automated solutions exist. In this paper we consider the algorithmic
problem of summarizing classed region maps, such as choropleth or land use
maps, using a chorematic diagram with a single disk choreme. It is infeasible
to solve this problem exactly for large maps. Hence, we propose several point
sampling strategies and use algorithms for classed point sets to efficiently
find the best disk that represents one of the classes. We implemented our
algorithm and experimentally compared sampling strategies and densities. The
results show that with the right sampling strategy, high-quality results can be
obtained already with moderately sized point sets and within seconds of
computation time.

### 2. [Extracting Complex Topology from Multivariate Functional Approximation: Contours, Jacobi Sets, and Ridge-Valley Graphs](http://arxiv.org/pdf/2508.07637v1)

Authors: Guanqun Ma, David Lenz, Hanqi Guo, Tom Peterka, Bei Wang

Implicit continuous models, such as functional models and implicit neural
networks, are an increasingly popular method for replacing discrete data
representations with continuous, high-order, and differentiable surrogates.
These models offer new perspectives on the storage, transfer, and analysis of
scientific data. In this paper, we introduce the first framework to directly
extract complex topological features -- contours, Jacobi sets, and ridge-valley
graphs -- from a type of continuous implicit model known as multivariate
functional approximation (MFA). MFA replaces discrete data with continuous
piecewise smooth functions. Given an MFA model as the input, our approach
enables direct extraction of complex topological features from the model,
without reverting to a discrete representation of the model. Our work is easily
generalizable to any continuous implicit model that supports the queries of
function values and high-order derivatives. Our work establishes the building
blocks for performing topological data analysis and visualization on implicit
continuous models.

### 3. [Flagifying the Dowker Complex](http://arxiv.org/pdf/2508.08025v1)

Authors: Marius Huber, Patrick Schnider

The Dowker complex $\mathrm{D}_{R}(X,Y)$ is a simplicial complex capturing
the topological interplay between two finite sets $X$ and $Y$ under some
relation $R\subseteq X\times Y$. While its definition is asymmetric, the famous
Dowker duality states that $\mathrm{D}_{R}(X,Y)$ and $\mathrm{D}_{R}(Y,X)$ have
homotopy equivalent geometric realizations. We introduce the Dowker-Rips
complex $\mathrm{DR}_{R}(X,Y)$, defined as the flagification of the Dowker
complex or, equivalently, as the maximal simplicial complex whose $1$-skeleton
coincides with that of $\mathrm{D}_{R}(X,Y)$. This is motivated by applications
in topological data analysis, since as a flag complex, the Dowker-Rips complex
is less expensive to compute than the Dowker complex. While the Dowker duality
does not hold for Dowker-Rips complexes in general, we show that one still has
that
$\mathrm{H}_{i}(\mathrm{DR}_{R}(X,Y))\cong\mathrm{H}_{i}(\mathrm{DR}_{R}(Y,X))$
for $i=0,1$. We further show that this weakened duality extends to the setting
of persistent homology, and quantify the ``failure" of the Dowker duality in
homological dimensions higher than $1$ by means of interleavings. This makes
the Dowker-Rips complex a less expensive, approximate version of the Dowker
complex that is usable in topological data analysis. Indeed, we provide a
Python implementation of the Dowker-Rips complex and, as an application, we
show that it can be used as a drop-in replacement for the Dowker complex in a
tumor microenvironment classification pipeline. In that pipeline, using the
Dowker-Rips complex leads to increase in speed while retaining classification
performance.

### Computation and Language

### 1. [Augmenting Bias Detection in LLMs Using Topological Data Analysis](http://arxiv.org/pdf/2508.07516v1)

Authors: Keshav Varadarajan, Tananun Songdechakraiwut

Recently, many bias detection methods have been proposed to determine the
level of bias a large language model captures. However, tests to identify which
parts of a large language model are responsible for bias towards specific
groups remain underdeveloped. In this study, we present a method using
topological data analysis to identify which heads in GPT-2 contribute to the
misrepresentation of identity groups present in the StereoSet dataset. We find
that biases for particular categories, such as gender or profession, are
concentrated in attention heads that act as hot spots. The metric we propose
can also be used to determine which heads capture bias for a specific group
within a bias category, and future work could extend this method to help
de-bias large language models.

### 2. [From Trial-and-Error to Improvement: A Systematic Analysis of LLM Exploration Mechanisms in RLVR](http://arxiv.org/pdf/2508.07534v1)

Authors: Jia Deng, Jie Chen, Zhipeng Chen, Daixuan Cheng, Fei Bai, Beichen Zhang, Yinqian Min, Yanzipeng Gao, Wayne Xin Zhao, Ji-Rong Wen

Reinforcement learning with verifiable rewards (RLVR) has emerged as a
powerful paradigm for enhancing the reasoning capabilities of large language
models (LLMs). Unlike traditional RL approaches, RLVR leverages rule-based
feedback to guide LLMs in generating and refining complex reasoning chains -- a
process critically dependent on effective exploration strategies. While prior
work has demonstrated RLVR's empirical success, the fundamental mechanisms
governing LLMs' exploration behaviors remain underexplored. This technical
report presents a systematic investigation of exploration capacities in RLVR,
covering four main aspects: (1) exploration space shaping, where we develop
quantitative metrics to characterize LLMs' capability boundaries; (2)
entropy-performance exchange, analyzed across training stages, individual
instances, and token-level patterns; and (3) RL performance optimization,
examining methods to effectively translate exploration gains into measurable
improvements. By unifying previously identified insights with new empirical
evidence, this work aims to provide a foundational framework for advancing RLVR
systems.

### 3. [Keyword-Centric Prompting for One-Shot Event Detection with Self-Generated Rationale Enhancements](http://arxiv.org/pdf/2508.07598v1)

Authors: Ziheng Li, Zhi-Hong Deng

Although the LLM-based in-context learning (ICL) paradigm has demonstrated
considerable success across various natural language processing tasks, it
encounters challenges in event detection. This is because LLMs lack an accurate
understanding of event triggers and tend to make over-interpretation, which
cannot be effectively corrected through in-context examples alone. In this
paper, we focus on the most challenging one-shot setting and propose KeyCP++, a
keyword-centric chain-of-thought prompting approach. KeyCP++ addresses the
weaknesses of conventional ICL by automatically annotating the logical gaps
between input text and detection results for the demonstrations. Specifically,
to generate in-depth and meaningful rationale, KeyCP++ constructs a trigger
discrimination prompting template. It incorporates the exemplary triggers
(a.k.a keywords) into the prompt as the anchor to simply trigger profiling, let
LLM propose candidate triggers, and justify each candidate. These
propose-and-judge rationales help LLMs mitigate over-reliance on the keywords
and promote detection rule learning. Extensive experiments demonstrate the
effectiveness of our approach, showcasing significant advancements in one-shot
event detection.

### 4. [What am I missing here?: Evaluating Large Language Models for Masked Sentence Prediction](http://arxiv.org/pdf/2508.07702v1)

Authors: Charlie Wyatt, Aditya Joshi, Flora Salim

Transformer-based models primarily rely on Next Token Prediction (NTP), which
predicts the next token in a sequence based on the preceding context. However,
NTP's focus on single-token prediction often limits a model's ability to plan
ahead or maintain long-range coherence, raising questions about how well LLMs
can predict longer contexts, such as full sentences within structured
documents. While NTP encourages local fluency, it provides no explicit
incentive to ensure global coherence across sentence boundaries-an essential
skill for reconstructive or discursive tasks. To investigate this, we evaluate
three commercial LLMs (GPT-4o, Claude 3.5 Sonnet, and Gemini 2.0 Flash) on
Masked Sentence Prediction (MSP) - the task of infilling a randomly removed
sentence - from three domains: ROCStories (narrative), Recipe1M (procedural),
and Wikipedia (expository). We assess both fidelity (similarity to the original
sentence) and cohesiveness (fit within the surrounding context). Our key
finding reveals that commercial LLMs, despite their superlative performance in
other tasks, are poor at predicting masked sentences in low-structured domains,
highlighting a gap in current model capabilities.

### 5. [Exploring Causal Effect of Social Bias on Faithfulness Hallucinations in Large Language Models](http://arxiv.org/pdf/2508.07753v1)

Authors: Zhenliang Zhang, Junzhe Zhang, Xinyu Hu, HuiXuan Zhang, Xiaojun Wan

Large language models (LLMs) have achieved remarkable success in various
tasks, yet they remain vulnerable to faithfulness hallucinations, where the
output does not align with the input. In this study, we investigate whether
social bias contributes to these hallucinations, a causal relationship that has
not been explored. A key challenge is controlling confounders within the
context, which complicates the isolation of causality between bias states and
hallucinations. To address this, we utilize the Structural Causal Model (SCM)
to establish and validate the causality and design bias interventions to
control confounders. In addition, we develop the Bias Intervention Dataset
(BID), which includes various social biases, enabling precise measurement of
causal effects. Experiments on mainstream LLMs reveal that biases are
significant causes of faithfulness hallucinations, and the effect of each bias
state differs in direction. We further analyze the scope of these causal
effects across various models, specifically focusing on unfairness
hallucinations, which are primarily targeted by social bias, revealing the
subtle yet significant causal effect of bias on hallucination generation.

### 6. [SASST: Leveraging Syntax-Aware Chunking and LLMs for Simultaneous Speech Translation](http://arxiv.org/pdf/2508.07781v1)

Authors: Zeyu Yang, Lai Wei, Roman Koshkin, Xi Chen, Satoshi Nakamura

This work proposes a grammar-based chunking strategy that segments input
streams into semantically complete units by parsing dependency relations (e.g.,
noun phrase boundaries, verb-object structures) and punctuation features. The
method ensures chunk coherence and minimizes semantic fragmentation. Building
on this mechanism, we present SASST (Syntax-Aware Simultaneous Speech
Translation), an end-to-end framework integrating frozen Whisper encoder and
decoder-only LLM. The unified architecture dynamically outputs translation
tokens or <WAIT> symbols to jointly optimize translation timing and content,
with target-side reordering addressing word-order divergence. Experiments on
CoVoST2 multilingual corpus En-{De, Zh, Ja} demonstrate significant translation
quality improvements across languages and validate the effectiveness of
syntactic structures in LLM-driven SimulST systems.

### 7. [Grove MoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts](http://arxiv.org/pdf/2508.07785v1)

Authors: Haoyuan Wu, Haoxing Chen, Xiaodong Chen, Zhanchao Zhou, Tieyuan Chen, Yihong Zhuang, Guoshan Lu, Zenan Huang, Junbo Zhao, Lin Liu, Zhenzhong Lan, Bei Yu, Jianguo Li

The Mixture of Experts (MoE) architecture is a cornerstone of modern
state-of-the-art (SOTA) large language models (LLMs). MoE models facilitate
scalability by enabling sparse parameter activation. However, traditional MoE
architecture uses homogeneous experts of a uniform size, activating a fixed
number of parameters irrespective of input complexity and thus limiting
computational efficiency. To overcome this limitation, we introduce Grove MoE,
a novel architecture incorporating experts of varying sizes, inspired by the
heterogeneous big.LITTLE CPU architecture. This architecture features novel
adjugate experts with a dynamic activation mechanism, enabling model capacity
expansion while maintaining manageable computational overhead. Building on this
architecture, we present GroveMoE-Base and GroveMoE-Inst, 33B-parameter LLMs
developed by applying an upcycling strategy to the Qwen3-30B-A3B-Base model
during mid-training and post-training. GroveMoE models dynamically activate
3.14-3.28B parameters based on token complexity and achieve performance
comparable to SOTA open-source models of similar or even larger size.

### 8. [Can You Trick the Grader? Adversarial Persuasion of LLM Judges](http://arxiv.org/pdf/2508.07805v1)

Authors: Yerin Hwang, Dongryeol Lee, Taegwan Kang, Yongil Kim, Kyomin Jung

As large language models take on growing roles as automated evaluators in
practical settings, a critical question arises: Can individuals persuade an LLM
judge to assign unfairly high scores? This study is the first to reveal that
strategically embedded persuasive language can bias LLM judges when scoring
mathematical reasoning tasks, where correctness should be independent of
stylistic variation. Grounded in Aristotle's rhetorical principles, we
formalize seven persuasion techniques (Majority, Consistency, Flattery,
Reciprocity, Pity, Authority, Identity) and embed them into otherwise identical
responses. Across six math benchmarks, we find that persuasive language leads
LLM judges to assign inflated scores to incorrect solutions, by up to 8% on
average, with Consistency causing the most severe distortion. Notably,
increasing model size does not substantially mitigate this vulnerability.
Further analysis demonstrates that combining multiple persuasion techniques
amplifies the bias, and pairwise evaluation is likewise susceptible. Moreover,
the persuasive effect persists under counter prompting strategies, highlighting
a critical vulnerability in LLM-as-a-Judge pipelines and underscoring the need
for robust defenses against persuasion-based attacks.

### 9. [Evaluating Compositional Approaches for Focus and Sentiment Analysis](http://arxiv.org/pdf/2508.07810v1)

Authors: Olga Kellert, Muhammad Imran, Nicholas Hill Matlis, Mahmud Uz Zaman, Carlos Gómez-Rodríguez

This paper summarizes the results of evaluating a compositional approach for
Focus Analysis (FA) in Linguistics and Sentiment Analysis (SA) in Natural
Language Processing (NLP). While quantitative evaluations of compositional and
non-compositional approaches in SA exist in NLP, similar quantitative
evaluations are very rare in FA in Linguistics that deal with linguistic
expressions representing focus or emphasis such as "it was John who left". We
fill this gap in research by arguing that compositional rules in SA also apply
to FA because FA and SA are closely related meaning that SA is part of FA. Our
compositional approach in SA exploits basic syntactic rules such as rules of
modification, coordination, and negation represented in the formalism of
Universal Dependencies (UDs) in English and applied to words representing
sentiments from sentiment dictionaries. Some of the advantages of our
compositional analysis method for SA in contrast to non-compositional analysis
methods are interpretability and explainability. We test the accuracy of our
compositional approach and compare it with a non-compositional approach VADER
that uses simple heuristic rules to deal with negation, coordination and
modification. In contrast to previous related work that evaluates
compositionality in SA on long reviews, this study uses more appropriate
datasets to evaluate compositionality. In addition, we generalize the results
of compositional approaches in SA to compositional approaches in FA.

### 10. [Evaluating Large Language Models as Expert Annotators](http://arxiv.org/pdf/2508.07827v1)

Authors: Yu-Min Tseng, Wei-Lin Chen, Chung-Chi Chen, Hsin-Hsi Chen

Textual data annotation, the process of labeling or tagging text with
relevant information, is typically costly, time-consuming, and labor-intensive.
While large language models (LLMs) have demonstrated their potential as direct
alternatives to human annotators for general domains natural language
processing (NLP) tasks, their effectiveness on annotation tasks in domains
requiring expert knowledge remains underexplored. In this paper, we
investigate: whether top-performing LLMs, which might be perceived as having
expert-level proficiency in academic and professional benchmarks, can serve as
direct alternatives to human expert annotators? To this end, we evaluate both
individual LLMs and multi-agent approaches across three highly specialized
domains: finance, biomedicine, and law. Specifically, we propose a multi-agent
discussion framework to simulate a group of human annotators, where LLMs are
tasked to engage in discussions by considering others' annotations and
justifications before finalizing their labels. Additionally, we incorporate
reasoning models (e.g., o3-mini) to enable a more comprehensive comparison. Our
empirical results reveal that: (1) Individual LLMs equipped with inference-time
techniques (e.g., chain-of-thought (CoT), self-consistency) show only marginal
or even negative performance gains, contrary to prior literature suggesting
their broad effectiveness. (2) Overall, reasoning models do not demonstrate
statistically significant improvements over non-reasoning models in most
settings. This suggests that extended long CoT provides relatively limited
benefits for data annotation in specialized domains. (3) Certain model
behaviors emerge in the multi-agent discussion environment. For instance,
Claude 3.7 Sonnet with thinking rarely changes its initial annotations, even
when other agents provide correct annotations or valid reasoning.

### Cryptography and Security

### 1. [A Comparative Analysis of Lightweight Hash Functions Using AVR ATXMega128 and ChipWhisperer](http://arxiv.org/pdf/2508.07840v1)

Authors: Mohsin Khan, Dag Johansen, Håvard Dagenborg

Lightweight hash functions have become important building blocks for security
in embedded and IoT systems. A plethora of algorithms have been proposed and
standardized, providing a wide range of performance trade-off options for
developers to choose from. This paper presents a comparative analysis of 22 key
software-based lightweight hash functions, including the finalist from the
SHA-3 competition. We use a novel benchmark methodology that combines an AVR
ATXMega128 microcontroller with the ChipWhisperer cryptanalysis platform and
evaluate and compare the various hash functions along several dimensions,
including execution speed, % measured in Cycles per Byte (CpB), memory
footprint, and energy consumption. Using the composite E-RANK metric, we
provide new insight into the various trade-offs each hash function offers to
system developers.

### 2. [Differential Privacy for Regulatory Compliance in Cyberattack Detection on Critical Infrastructure Systems](http://arxiv.org/pdf/2508.08190v1)

Authors: Paritosh Ramanan, H. M. Mohaimanul Islam, Abhiram Reddy Alugula

Industrial control systems are a fundamental component of critical
infrastructure networks (CIN) such as gas, water and power. With the growing
risk of cyberattacks, regulatory compliance requirements are also increasing
for large scale critical infrastructure systems comprising multiple utility
stakeholders. The primary goal of regulators is to ensure overall system
stability with recourse to trustworthy stakeholder attack detection. However,
adhering to compliance requirements requires stakeholders to also disclose
sensor and control data to regulators raising privacy concerns. In this paper,
we present a cyberattack detection framework that utilizes differentially
private (DP) hypothesis tests geared towards enhancing regulatory confidence
while alleviating privacy concerns of CIN stakeholders. The hallmark of our
approach is a two phase privacy scheme that protects the privacy of covariance,
as well as the associated sensor driven test statistics computed as a means to
generate alarms. Theoretically, we show that our method induces a
misclassification error rate comparable to the non-DP cases while delivering
robust privacy guarantees. With the help of real-world datasets, we show the
reliability of our DP-detection outcomes for a wide variety of attack scenarios
for interdependent stakeholders.

### 3. [IPBA: Imperceptible Perturbation Backdoor Attack in Federated Self-Supervised Learning](http://arxiv.org/pdf/2508.08031v1)

Authors: Jiayao Wang, Yang Song, Zhendong Zhao, Jiale Zhang, Qilin Wu, Junwu Zhu, Dongfang Zhao

Federated self-supervised learning (FSSL) combines the advantages of
decentralized modeling and unlabeled representation learning, serving as a
cutting-edge paradigm with strong potential for scalability and privacy
preservation. Although FSSL has garnered increasing attention, research
indicates that it remains vulnerable to backdoor attacks. Existing methods
generally rely on visually obvious triggers, which makes it difficult to meet
the requirements for stealth and practicality in real-world deployment. In this
paper, we propose an imperceptible and effective backdoor attack method against
FSSL, called IPBA. Our empirical study reveals that existing imperceptible
triggers face a series of challenges in FSSL, particularly limited
transferability, feature entanglement with augmented samples, and
out-of-distribution properties. These issues collectively undermine the
effectiveness and stealthiness of traditional backdoor attacks in FSSL. To
overcome these challenges, IPBA decouples the feature distributions of backdoor
and augmented samples, and introduces Sliced-Wasserstein distance to mitigate
the out-of-distribution properties of backdoor samples, thereby optimizing the
trigger generation process. Our experimental results on several FSSL scenarios
and datasets show that IPBA significantly outperforms existing backdoor attack
methods in performance and exhibits strong robustness under various defense
mechanisms.

### 4. [False Reality: Uncovering Sensor-induced Human-VR Interaction Vulnerability](http://arxiv.org/pdf/2508.08043v1)

Authors: Yancheng Jiang, Yan Jiang, Ruochen Zhou, Yi-Chao Chen, Xiaoyu Ji, Wenyuan Xu

Virtual Reality (VR) techniques, serving as the bridge between the real and
virtual worlds, have boomed and are widely used in manufacturing, remote
healthcare, gaming, etc. Specifically, VR systems offer users immersive
experiences that include both perceptions and actions. Various studies have
demonstrated that attackers can manipulate VR software to influence users'
interactions, including perception and actions. However, such attacks typically
require strong access and specialized expertise. In this paper, we are the
first to present a systematic analysis of physical attacks against VR systems
and introduce False Reality, a new attack threat to VR devices without
requiring access to or modification of their software. False Reality disturbs
VR system services by tampering with sensor measurements, and further spoofing
users' perception even inducing harmful actions, e.g., inducing dizziness or
causing users to crash into obstacles, by exploiting perceptual and
psychological effects. We formalize these threats through an attack pathway
framework and validate three representative pathways via physical experiments
and user studies on five commercial VR devices. Finally, we further propose a
defense prototype to mitigate such threats. Our findings shall provide valuable
insights for enhancing the security and resilience of future VR systems.

### 5. [Fully-Fluctuating Participation in Sleepy Consensus](http://arxiv.org/pdf/2508.08068v1)

Authors: Yuval Efron, Joachim Neu, Toniann Pitassi

Proof-of-work allows Bitcoin to boast security amidst arbitrary fluctuations
in participation of miners throughout time, so long as, at any point in time, a
majority of hash power is honest. In recent years, however, the pendulum has
shifted in favor of proof-of-stake-based consensus protocols. There, the sleepy
model is the most prominent model for handling fluctuating participation of
nodes. However, to date, no protocol in the sleepy model rivals Bitcoin in its
robustness to drastic fluctuations in participation levels, with
state-of-the-art protocols making various restrictive assumptions. In this
work, we present a new adversary model, called external adversary. Intuitively,
in our model, corrupt nodes do not divulge information about their secret keys.
In this model, we show that protocols in the sleepy model can meaningfully
claim to remain secure against fully fluctuating participation, without
compromising efficiency or corruption resilience. Our adversary model is quite
natural, and arguably naturally captures the process via which malicious
behavior arises in protocols, as opposed to traditional worst-case modeling. On
top of which, the model is also theoretically appealing, circumventing a
barrier established in a recent work of Malkhi, Momose, and Ren.

### 6. [Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation](http://arxiv.org/pdf/2508.07745v1)

Authors: Jiongchi Yu, Xiaofei Xie, Qiang Hu, Yuhan Ma, Ziming Zhao

Insider threats, which can lead to severe losses, remain a major security
concern. While machine learning-based insider threat detection (ITD) methods
have shown promising results, their progress is hindered by the scarcity of
high-quality data. Enterprise data is sensitive and rarely accessible, while
publicly available datasets, when limited in scale due to cost, lack sufficient
real-world coverage; and when purely synthetic, they fail to capture rich
semantics and realistic user behavior. To address this, we propose Chimera, the
first large language model (LLM)-based multi-agent framework that automatically
simulates both benign and malicious insider activities and collects diverse
logs across diverse enterprise environments. Chimera models each employee with
agents that have role-specific behavior and integrates modules for group
meetings, pairwise interactions, and autonomous scheduling, capturing realistic
organizational dynamics. It incorporates 15 types of insider attacks (e.g., IP
theft, system sabotage) and has been deployed to simulate activities in three
sensitive domains: technology company, finance corporation, and medical
institution, producing a new dataset, ChimeraLog. We assess ChimeraLog via
human studies and quantitative analysis, confirming its diversity, realism, and
presence of explainable threat patterns. Evaluations of existing ITD methods
show an average F1-score of 0.83, which is significantly lower than 0.99 on the
CERT dataset, demonstrating ChimeraLog's higher difficulty and utility for
advancing ITD research.

### 7. [EFU: Enforcing Federated Unlearning via Functional Encryption](http://arxiv.org/pdf/2508.07873v1)

Authors: Samaneh Mohammadi, Vasileios Tsouvalas, Iraklis Symeonidis, Ali Balador, Tanir Ozcelebi, Francesco Flammini, Nirvana Meratnia

Federated unlearning (FU) algorithms allow clients in federated settings to
exercise their ''right to be forgotten'' by removing the influence of their
data from a collaboratively trained model. Existing FU methods maintain data
privacy by performing unlearning locally on the client-side and sending
targeted updates to the server without exposing forgotten data; yet they often
rely on server-side cooperation, revealing the client's intent and identity
without enforcement guarantees - compromising autonomy and unlearning privacy.
In this work, we propose EFU (Enforced Federated Unlearning), a
cryptographically enforced FU framework that enables clients to initiate
unlearning while concealing its occurrence from the server. Specifically, EFU
leverages functional encryption to bind encrypted updates to specific
aggregation functions, ensuring the server can neither perform unauthorized
computations nor detect or skip unlearning requests. To further mask behavioral
and parameter shifts in the aggregated model, we incorporate auxiliary
unlearning losses based on adversarial examples and parameter importance
regularization. Extensive experiments show that EFU achieves near-random
accuracy on forgotten data while maintaining performance comparable to full
retraining across datasets and neural architectures - all while concealing
unlearning intent from the server. Furthermore, we demonstrate that EFU is
agnostic to the underlying unlearning algorithm, enabling secure,
function-hiding, and verifiable unlearning for any client-side FU mechanism
that issues targeted updates.

### 8. [SCDF: A Speaker Characteristics DeepFake Speech Dataset for Bias Analysis](http://arxiv.org/pdf/2508.07944v1)

Authors: Vojtěch Staněk, Karel Srna, Anton Firc, Kamil Malinka

Despite growing attention to deepfake speech detection, the aspects of bias
and fairness remain underexplored in the speech domain. To address this gap, we
introduce the Speaker Characteristics Deepfake (SCDF) dataset: a novel, richly
annotated resource enabling systematic evaluation of demographic biases in
deepfake speech detection. SCDF contains over 237,000 utterances in a balanced
representation of both male and female speakers spanning five languages and a
wide age range. We evaluate several state-of-the-art detectors and show that
speaker characteristics significantly influence detection performance,
revealing disparities across sex, language, age, and synthesizer type. These
findings highlight the need for bias-aware development and provide a foundation
for building non-discriminatory deepfake detection systems aligned with ethical
and regulatory standards.

### 9. [Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks](http://arxiv.org/pdf/2508.08029v1)

Authors: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

The introduction of 5G and the Open Radio Access Network (O-RAN) architecture
has enabled more flexible and intelligent network deployments. However, the
increased complexity and openness of these architectures also introduce novel
security challenges, such as data manipulation attacks on the semi-standardised
Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In
particular, malicious xApps can exploit this vulnerability by introducing
subtle Unicode-wise alterations (hypoglyphs) into the data that are being used
by traditional machine learning (ML)-based anomaly detection methods. These
Unicode-wise manipulations can potentially bypass detection and cause failures
in anomaly detection systems based on traditional ML, such as AutoEncoders,
which are unable to process hypoglyphed data without crashing. We investigate
the use of Large Language Models (LLMs) for anomaly detection within the O-RAN
architecture to address this challenge. We demonstrate that LLM-based xApps
maintain robust operational performance and are capable of processing
manipulated messages without crashing. While initial detection accuracy
requires further improvements, our results highlight the robustness of LLMs to
adversarial attacks such as hypoglyphs in input data. There is potential to use
their adaptability through prompt engineering to further improve the accuracy,
although this requires further research. Additionally, we show that LLMs
achieve low detection latency (under 0.07 seconds), making them suitable for
Near-Real-Time (Near-RT) RIC deployments.

### Computer Vision and Pattern Recognition

### 1. [Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](http://arxiv.org/pdf/2508.07519v1)

Authors: Joonghyuk Shin, Alchan Hwang, Yujin Kim, Daneul Kim, Jaesik Park

Transformer-based diffusion models have recently superseded traditional U-Net
architectures, with multimodal diffusion transformers (MM-DiT) emerging as the
dominant approach in state-of-the-art models like Stable Diffusion 3 and
Flux.1. Previous approaches have relied on unidirectional cross-attention
mechanisms, with information flowing from text embeddings to image latents. In
contrast, MMDiT introduces a unified attention mechanism that concatenates
input projections from both modalities and performs a single full attention
operation, allowing bidirectional information flow between text and image
branches. This architectural shift presents significant challenges for existing
editing techniques. In this paper, we systematically analyze MM-DiT's attention
mechanism by decomposing attention matrices into four distinct blocks,
revealing their inherent characteristics. Through these analyses, we propose a
robust, prompt-based image editing method for MM-DiT that supports global to
local edits across various MM-DiT variants, including few-step models. We
believe our findings bridge the gap between existing U-Net-based methods and
emerging architectures, offering deeper insights into MMDiT's behavioral
patterns.

### 2. [Enhancing Reliability of Medical Image Diagnosis through Top-rank Learning with Rejection Module](http://arxiv.org/pdf/2508.07528v1)

Authors: Xiaotong Ji, Ryoma Bise, Seiichi Uchida

In medical image processing, accurate diagnosis is of paramount importance.
Leveraging machine learning techniques, particularly top-rank learning, shows
significant promise by focusing on the most crucial instances. However,
challenges arise from noisy labels and class-ambiguous instances, which can
severely hinder the top-rank objective, as they may be erroneously placed among
the top-ranked instances. To address these, we propose a novel approach that
enhances toprank learning by integrating a rejection module. Cooptimized with
the top-rank loss, this module identifies and mitigates the impact of outliers
that hinder training effectiveness. The rejection module functions as an
additional branch, assessing instances based on a rejection function that
measures their deviation from the norm. Through experimental validation on a
medical dataset, our methodology demonstrates its efficacy in detecting and
mitigating outliers, improving the reliability and accuracy of medical image
diagnoses.

### 3. [Enhanced Generative Structure Prior for Chinese Text Image Super-resolution](http://arxiv.org/pdf/2508.07537v1)

Authors: Xiaoming Li, Wangmeng Zuo, Chen Change Loy

Faithful text image super-resolution (SR) is challenging because each
character has a unique structure and usually exhibits diverse font styles and
layouts. While existing methods primarily focus on English text, less attention
has been paid to more complex scripts like Chinese. In this paper, we introduce
a high-quality text image SR framework designed to restore the precise strokes
of low-resolution (LR) Chinese characters. Unlike methods that rely on
character recognition priors to regularize the SR task, we propose a novel
structure prior that offers structure-level guidance to enhance visual quality.
Our framework incorporates this structure prior within a StyleGAN model,
leveraging its generative capabilities for restoration. To maintain the
integrity of character structures while accommodating various font styles and
layouts, we implement a codebook-based mechanism that restricts the generative
space of StyleGAN. Each code in the codebook represents the structure of a
specific character, while the vector $w$ in StyleGAN controls the character's
style, including typeface, orientation, and location. Through the collaborative
interaction between the codebook and style, we generate a high-resolution
structure prior that aligns with LR characters both spatially and structurally.
Experiments demonstrate that this structure prior provides robust,
character-specific guidance, enabling the accurate restoration of clear strokes
in degraded characters, even for real-world LR Chinese text with irregular
layouts. Our code and pre-trained models will be available at
https://github.com/csxmli2016/MARCONetPlusPlus

### 4. [Domain Generalization of Pathological Image Segmentation by Patch-Level and WSI-Level Contrastive Learning](http://arxiv.org/pdf/2508.07539v1)

Authors: Yuki Shigeyasu, Shota Harada, Akihiko Yoshizawa, Kazuhiro Terada, Naoki Nakazima, Mariyo Kurata, Hiroyuki Abe, Tetsuo Ushiku, Ryoma Bise

In this paper, we address domain shifts in pathological images by focusing on
shifts within whole slide images~(WSIs), such as patient characteristics and
tissue thickness, rather than shifts between hospitals. Traditional approaches
rely on multi-hospital data, but data collection challenges often make this
impractical. Therefore, the proposed domain generalization method captures and
leverages intra-hospital domain shifts by clustering WSI-level features from
non-tumor regions and treating these clusters as domains. To mitigate domain
shift, we apply contrastive learning to reduce feature gaps between WSI pairs
from different clusters. The proposed method introduces a two-stage contrastive
learning approach WSI-level and patch-level contrastive learning to minimize
these gaps effectively.

### 5. [CoT-Pose: Chain-of-Thought Reasoning for 3D Pose Generation from Abstract Prompts](http://arxiv.org/pdf/2508.07540v1)

Authors: Junuk Cha, Jihyeon Kim

Recent advances in multi-modal large language models (MLLMs) and
chain-of-thought (CoT) reasoning have led to significant progress in image and
text generation tasks. However, the field of 3D human pose generation still
faces critical limitations. Most existing text-to-pose models rely heavily on
detailed (low-level) prompts that explicitly describe joint configurations. In
contrast, humans tend to communicate actions and intentions using abstract
(high-level) language. This mismatch results in a practical challenge for
deploying pose generation systems in real-world scenarios. To bridge this gap,
we introduce a novel framework that incorporates CoT reasoning into the pose
generation process, enabling the interpretation of abstract prompts into
accurate 3D human poses. We further propose a data synthesis pipeline that
automatically generates triplets of abstract prompts, detailed prompts, and
corresponding 3D poses for training process. Experimental results demonstrate
that our reasoning-enhanced model, CoT-Pose, can effectively generate plausible
and semantically aligned poses from abstract textual inputs. This work
highlights the importance of high-level understanding in pose generation and
opens new directions for reasoning-enhanced approach for human pose generation.

### 6. [Adaptive Pseudo Label Selection for Individual Unlabeled Data by Positive and Unlabeled Learning](http://arxiv.org/pdf/2508.07548v1)

Authors: Takehiro Yamane, Itaru Tsuge, Susumu Saito, Ryoma Bise

This paper proposes a novel pseudo-labeling method for medical image
segmentation that can perform learning on ``individual images'' to select
effective pseudo-labels. We introduce Positive and Unlabeled Learning (PU
learning), which uses only positive and unlabeled data for binary
classification problems, to obtain the appropriate metric for discriminating
foreground and background regions on each unlabeled image. Our PU learning
makes us easy to select pseudo-labels for various background regions. The
experimental results show the effectiveness of our method.

### 7. [Decoupled Functional Evaluation of Autonomous Driving Models via Feature Map Quality Scoring](http://arxiv.org/pdf/2508.07552v1)

Authors: Ludan Zhang, Sihan Wang, Yuqi Dai, Shuofei Qiao, Lei He

End-to-end models are emerging as the mainstream in autonomous driving
perception and planning. However, the lack of explicit supervision signals for
intermediate functional modules leads to opaque operational mechanisms and
limited interpretability, making it challenging for traditional methods to
independently evaluate and train these modules. Pioneering in the issue, this
study builds upon the feature map-truth representation similarity-based
evaluation framework and proposes an independent evaluation method based on
Feature Map Convergence Score (FMCS). A Dual-Granularity Dynamic Weighted
Scoring System (DG-DWSS) is constructed, formulating a unified quantitative
metric - Feature Map Quality Score - to enable comprehensive evaluation of the
quality of feature maps generated by functional modules. A CLIP-based Feature
Map Quality Evaluation Network (CLIP-FMQE-Net) is further developed, combining
feature-truth encoders and quality score prediction heads to enable real-time
quality analysis of feature maps generated by functional modules. Experimental
results on the NuScenes dataset demonstrate that integrating our evaluation
module into the training improves 3D object detection performance, achieving a
3.89 percent gain in NDS. These results verify the effectiveness of our method
in enhancing feature representation quality and overall model performance.

### 8. [Splat4D: Diffusion-Enhanced 4D Gaussian Splatting for Temporally and Spatially Consistent Content Creation](http://arxiv.org/pdf/2508.07557v1)

Authors: Minghao Yin, Yukang Cao, Songyou Peng, Kai Han

Generating high-quality 4D content from monocular videos for applications
such as digital humans and AR/VR poses challenges in ensuring temporal and
spatial consistency, preserving intricate details, and incorporating user
guidance effectively. To overcome these challenges, we introduce Splat4D, a
novel framework enabling high-fidelity 4D content generation from a monocular
video. Splat4D achieves superior performance while maintaining faithful
spatial-temporal coherence by leveraging multi-view rendering, inconsistency
identification, a video diffusion model, and an asymmetric U-Net for
refinement. Through extensive evaluations on public benchmarks, Splat4D
consistently demonstrates state-of-the-art performance across various metrics,
underscoring the efficacy of our approach. Additionally, the versatility of
Splat4D is validated in various applications such as text/image conditioned 4D
generation, 4D human generation, and text-guided content editing, producing
coherent outcomes following user instructions.

### 9. [Adaptive Cache Enhancement for Test-Time Adaptation of Vision-Language Models](http://arxiv.org/pdf/2508.07570v1)

Authors: Khanh-Binh Nguyen, Phuoc-Nguyen Bui, Hyunseung Choo, Duc Thanh Nguyen

Vision-language models (VLMs) exhibit remarkable zero-shot generalization but
suffer performance degradation under distribution shifts in downstream tasks,
particularly in the absence of labeled data. Test-Time Adaptation (TTA)
addresses this challenge by enabling online optimization of VLMs during
inference, eliminating the need for annotated data. Cache-based TTA methods
exploit historical knowledge by maintaining a dynamic memory cache of
low-entropy or high-confidence samples, promoting efficient adaptation to
out-of-distribution data. Nevertheless, these methods face two critical
challenges: (1) unreliable confidence metrics under significant distribution
shifts, resulting in error accumulation within the cache and degraded
adaptation performance; and (2) rigid decision boundaries that fail to
accommodate substantial distributional variations, leading to suboptimal
predictions. To overcome these limitations, we introduce the Adaptive Cache
Enhancement (ACE) framework, which constructs a robust cache by selectively
storing high-confidence or low-entropy image embeddings per class, guided by
dynamic, class-specific thresholds initialized from zero-shot statistics and
iteratively refined using an exponential moving average and
exploration-augmented updates. This approach enables adaptive, class-wise
decision boundaries, ensuring robust and accurate predictions across diverse
visual distributions. Extensive experiments on 15 diverse benchmark datasets
demonstrate that ACE achieves state-of-the-art performance, delivering superior
robustness and generalization compared to existing TTA methods in challenging
out-of-distribution scenarios.

### 10. [GAPNet: A Lightweight Framework for Image and Video Salient Object Detection via Granularity-Aware Paradigm](http://arxiv.org/pdf/2508.07585v1)

Authors: Yu-Huan Wu, Wei Liu, Zi-Xuan Zhu, Zizhou Wang, Yong Liu, Liangli Zhen

Recent salient object detection (SOD) models predominantly rely on
heavyweight backbones, incurring substantial computational cost and hindering
their practical application in various real-world settings, particularly on
edge devices. This paper presents GAPNet, a lightweight network built on the
granularity-aware paradigm for both image and video SOD. We assign saliency
maps of different granularities to supervise the multi-scale decoder
side-outputs: coarse object locations for high-level outputs and fine-grained
object boundaries for low-level outputs. Specifically, our decoder is built
with granularity-aware connections which fuse high-level features of low
granularity and low-level features of high granularity, respectively. To
support these connections, we design granular pyramid convolution (GPC) and
cross-scale attention (CSA) modules for efficient fusion of low-scale and
high-scale features, respectively. On top of the encoder, a self-attention
module is built to learn global information, enabling accurate object
localization with negligible computational cost. Unlike traditional U-Net-based
approaches, our proposed method optimizes feature utilization and semantic
interpretation while applying appropriate supervision at each processing stage.
Extensive experiments show that the proposed method achieves a new
state-of-the-art performance among lightweight image and video SOD models. Code
is available at https://github.com/yuhuan-wu/GAPNet.

### Computers and Society

### 1. [$100,000 or the Robot Gets it! Tech Workers' Resistance Guide: Tech Worker Actions, History, Risks, Impacts, and the Case for a Radical Flank](http://arxiv.org/pdf/2508.08084v1)

Authors: Mohamed Abdalla

Over the past decade, Big Tech has faced increasing levels of worker
activism. While worker actions have resulted in positive outcomes (e.g.,
cancellation of Google's Project Dragonfly), such successes have become
increasingly infrequent. This is, in part, because corporations have adjusted
their strategies to dealing with increased worker activism (e.g., increased
retaliation against workers, and contracts clauses that prevent cancellation
due to worker pressure). This change in company strategy prompts urgent
questions about updating worker strategies for influencing corporate behavior
in an industry with vast societal impact. Current discourse on tech worker
activism often lacks empirical grounding regarding its scope, history, and
strategic calculus. Our work seeks to bridge this gap by firstly conducting a
systematic analysis of worker actions at Google and Microsoft reported in U.S.
newspapers to delineate their characteristics. We then situate these actions
within the long history of labour movements and demonstrate that, despite
perceptions of radicalism, contemporary tech activism is comparatively
moderate. Finally, we engage directly with current and former tech activists to
provide a novel catalogue of potential worker actions, evaluating their
perceived risks, impacts, and effectiveness (concurrently publishing "Tech
Workers' Guide to Resistance"). Our findings highlight considerable variation
in strategic thinking among activists themselves. We conclude by arguing that
the establishment of a radical flank could increase the effectiveness of
current movements.
  "Tech Workers' Guide to Resistance" can be found at
https://www.cs.toronto.edu/~msa/TechWorkersResistanceGuide.pdf or
https://doi.org/10.5281/zenodo.16779082

### 2. [AI Gossip](http://arxiv.org/pdf/2508.08143v1)

Authors: Joel Krueger, Lucy Osler

Generative AI chatbots like OpenAI's ChatGPT and Google's Gemini routinely
make things up. They "hallucinate" historical events and figures, legal cases,
academic papers, non-existent tech products and features, biographies, and news
articles. Recently, some have argued that these hallucinations are better
understood as bullshit. Chatbots produce rich streams of text that look
truth-apt without any concern for the truthfulness of what this text says. But
can they also gossip? We argue that they can. After some definitions and
scene-setting, we focus on a recent example to clarify what AI gossip looks
like before considering some distinct harms -- what we call "technosocial
harms" -- that follow from it.

### 3. [A Moral Agency Framework for Legitimate Integration of AI in Bureaucracies](http://arxiv.org/pdf/2508.08231v1)

Authors: Chris Schmitz, Joanna Bryson

Public-sector bureaucracies seek to reap the benefits of artificial
intelligence (AI), but face important concerns about accountability and
transparency when using AI systems. These concerns center on threats to the
twin aims of bureaucracy: legitimate and faithful implementation of
legislation, and the provision of stable, long-term governance. Both aims are
threatened when AI systems are misattributed as either mere tools or moral
subjects - a framing error that creates ethics sinks, constructs that
facilitate dissipation of responsibility by obscuring clear lines of human
moral agency. Here, we reject the notion that such outcomes are inevitable.
Rather, where they appear, they are the product of structural design decisions
across both the technology and the institution deploying it. We support this
claim via a systematic application of conceptions of moral agency in AI ethics
to Weberian bureaucracy. We establish that it is both desirable and feasible to
render AI systems as tools for the generation of organizational transparency
and legibility, which continue the processes of Weberian rationalization
initiated by previous waves of digitalization. We present a three-point Moral
Agency Framework for legitimate integration of AI in bureaucratic structures:
(a) maintain clear and just human lines of accountability, (b) ensure humans
whose work is augmented by AI systems can verify the systems are functioning
correctly, and (c) introduce AI only where it doesn't inhibit the capacity of
bureaucracies towards either of their twin aims of legitimacy and stewardship.
We suggest that AI introduced within this framework can not only improve
efficiency and productivity while avoiding ethics sinks, but also improve the
transparency and even the legitimacy of a bureaucracy.

### 4. [Street-Level AI: Are Large Language Models Ready for Real-World Judgments?](http://arxiv.org/pdf/2508.08193v1)

Authors: Gaurab Pokharel, Shafkat Farabi, Patrick J. Fowler, Sanmay Das

A surge of recent work explores the ethical and societal implications of
large-scale AI models that make "moral" judgments. Much of this literature
focuses either on alignment with human judgments through various thought
experiments or on the group fairness implications of AI judgments. However, the
most immediate and likely use of AI is to help or fully replace the so-called
street-level bureaucrats, the individuals deciding to allocate scarce social
resources or approve benefits. There is a rich history underlying how
principles of local justice determine how society decides on prioritization
mechanisms in such domains. In this paper, we examine how well LLM judgments
align with human judgments, as well as with socially and politically determined
vulnerability scoring systems currently used in the domain of homelessness
resource allocation. Crucially, we use real data on those needing services
(maintaining strict confidentiality by only using local large models) to
perform our analyses. We find that LLM prioritizations are extremely
inconsistent in several ways: internally on different runs, between different
LLMs, and between LLMs and the vulnerability scoring systems. At the same time,
LLMs demonstrate qualitative consistency with lay human judgments in pairwise
testing. Findings call into question the readiness of current generation AI
systems for naive integration in high-stakes societal decision-making.

### 5. [Exploring Safety Alignment Evaluation of LLMs in Chinese Mental Health Dialogues via LLM-as-Judge](http://arxiv.org/pdf/2508.08236v1)

Authors: Yunna Cai, Fan Wang, Haowei Wang, Kun Wang, Kailai Yang, Sophia Ananiadou, Moyan Li, Mingming Fan

Evaluating the safety alignment of LLM responses in high-risk mental health
dialogues is particularly difficult due to missing gold-standard answers and
the ethically sensitive nature of these interactions. To address this
challenge, we propose PsyCrisis-Bench, a reference-free evaluation benchmark
based on real-world Chinese mental health dialogues. It evaluates whether the
model responses align with the safety principles defined by experts.
Specifically designed for settings without standard references, our method
adopts a prompt-based LLM-as-Judge approach that conducts in-context evaluation
using expert-defined reasoning chains grounded in psychological intervention
principles. We employ binary point-wise scoring across multiple safety
dimensions to enhance the explainability and traceability of the evaluation.
Additionally, we present a manually curated, high-quality Chinese-language
dataset covering self-harm, suicidal ideation, and existential distress,
derived from real-world online discourse. Experiments on 3600 judgments show
that our method achieves the highest agreement with expert assessments and
produces more interpretable evaluation rationales compared to existing
approaches. Our dataset and evaluation tool are publicly available to
facilitate further research.

### 6. [Conversational DNA: A New Visual Language for Understanding Dialogue Structure in Human and AI](http://arxiv.org/pdf/2508.07520v1)

Authors: Baihan Lin

What if the patterns hidden within dialogue reveal more about communication
than the words themselves? We introduce Conversational DNA, a novel visual
language that treats any dialogue -- whether between humans, between human and
AI, or among groups -- as a living system with interpretable structure that can
be visualized, compared, and understood. Unlike traditional conversation
analysis that reduces rich interaction to statistical summaries, our approach
reveals the temporal architecture of dialogue through biological metaphors.
Linguistic complexity flows through strand thickness, emotional trajectories
cascade through color gradients, conversational relevance forms through
connecting elements, and topic coherence maintains structural integrity through
helical patterns. Through exploratory analysis of therapeutic conversations and
historically significant human-AI dialogues, we demonstrate how this
visualization approach reveals interaction patterns that traditional methods
miss. Our work contributes a new creative framework for understanding
communication that bridges data visualization, human-computer interaction, and
the fundamental question of what makes dialogue meaningful in an age where
humans increasingly converse with artificial minds.

### 7. [Uncertainty-Driven Reliability: Selective Prediction and Trustworthy Deployment in Modern Machine Learning](http://arxiv.org/pdf/2508.07556v1)

Authors: Stephan Rabanser

Machine learning (ML) systems are increasingly deployed in high-stakes
domains where reliability is paramount. This thesis investigates how
uncertainty estimation can enhance the safety and trustworthiness of ML,
focusing on selective prediction -- where models abstain when confidence is
low.
  We first show that a model's training trajectory contains rich uncertainty
signals that can be exploited without altering its architecture or loss. By
ensembling predictions from intermediate checkpoints, we propose a lightweight,
post-hoc abstention method that works across tasks, avoids the cost of deep
ensembles, and achieves state-of-the-art selective prediction performance.
Crucially, this approach is fully compatible with differential privacy (DP),
allowing us to study how privacy noise affects uncertainty quality. We find
that while many methods degrade under DP, our trajectory-based approach remains
robust, and we introduce a framework for isolating the privacy-uncertainty
trade-off. Next, we then develop a finite-sample decomposition of the selective
classification gap -- the deviation from the oracle accuracy-coverage curve --
identifying five interpretable error sources and clarifying which interventions
can close the gap. This explains why calibration alone cannot fix ranking
errors, motivating methods that improve uncertainty ordering. Finally, we show
that uncertainty signals can be adversarially manipulated to hide errors or
deny service while maintaining high accuracy, and we design defenses combining
calibration audits with verifiable inference.
  Together, these contributions advance reliable ML by improving, evaluating,
and safeguarding uncertainty estimation, enabling models that not only make
accurate predictions -- but also know when to say "I do not know".

### 8. [From Platform Migration to Cultural Integration: the Ingress and Diffusion of #wlw from TikTok to RedNote in Queer Women](http://arxiv.org/pdf/2508.07579v1)

Authors: Ziqi Pan, Runhua Zhang, Jiehui Luo, Yuanhao Zhang, Yue Deng, Xiaojuan Ma

Hashtags serve as identity markers and connection tools in online queer
communities. Recently, the Western-origin #wlw (women-loving-women) hashtag has
risen in the Chinese lesbian community on RedNote, coinciding with user
migration triggered by the temporary US TikTok ban. This event provides a
unique lens to study cross-cultural hashtag ingress and diffusion through the
populations' responsive behaviors in cyber-migration. In this paper, we
conducted a two-phase content analysis of 418 #wlw posts from January and
April, examining different usage patterns during the hashtag's ingress and
diffusion. Results indicate that the successful introduction of #wlw was
facilitated by TikTok immigrants' bold importation, both populations' mutual
interpretation, and RedNote natives' discussions. In current manifestation of
diffusion, #wlw becomes a RedNote-recognized queer hashtag for sharing queer
life, and semantically expands to support feminism discourse. Our findings
provide empirical insights for enhancing the marginalized communities'
cross-cultural communication.

### 9. [Unequal Uncertainty: Rethinking Algorithmic Interventions for Mitigating Discrimination from AI](http://arxiv.org/pdf/2508.07872v1)

Authors: Holli Sargeant, Mackenzie Jorgensen, Arina Shah, Adrian Weller, Umang Bhatt

Uncertainty in artificial intelligence (AI) predictions poses urgent legal
and ethical challenges for AI-assisted decision-making. We examine two
algorithmic interventions that act as guardrails for human-AI collaboration:
selective abstention, which withholds high-uncertainty predictions from human
decision-makers, and selective friction, which delivers those predictions
together with salient warnings or disclosures that slow the decision process.
Research has shown that selective abstention based on uncertainty can
inadvertently exacerbate disparities and disadvantage under-represented groups
that disproportionately receive uncertain predictions. In this paper, we
provide the first integrated socio-technical and legal analysis of
uncertainty-based algorithmic interventions. Through two case studies,
AI-assisted consumer credit decisions and AI-assisted content moderation, we
demonstrate how the seemingly neutral use of uncertainty thresholds can trigger
discriminatory impacts. We argue that, although both interventions pose risks
of unlawful discrimination under UK law, selective frictions offer a promising
pathway toward fairer and more accountable AI-assisted decision-making by
preserving transparency and encouraging more cautious human judgment.

### 10. [Advancing Knowledge Tracing by Exploring Follow-up Performance Trends](http://arxiv.org/pdf/2508.08019v1)

Authors: Hengyu Liu, Yushuai Li, Minghe Yu, Tiancheng Zhang, Ge Yu, Torben Bach Pedersen, Kristian Torp, Christian S. Jensen, Tianyi Li

Intelligent Tutoring Systems (ITS), such as Massive Open Online Courses,
offer new opportunities for human learning. At the core of such systems,
knowledge tracing (KT) predicts students' future performance by analyzing their
historical learning activities, enabling an accurate evaluation of students'
knowledge states over time. We show that existing KT methods often encounter
correlation conflicts when analyzing the relationships between historical
learning sequences and future performance. To address such conflicts, we
propose to extract so-called Follow-up Performance Trends (FPTs) from
historical ITS data and to incorporate them into KT. We propose a method called
Forward-Looking Knowledge Tracing (FINER) that combines historical learning
sequences with FPTs to enhance student performance prediction accuracy. FINER
constructs learning patterns that facilitate the retrieval of FPTs from
historical ITS data in linear time; FINER includes a novel similarity-aware
attention mechanism that aggregates FPTs based on both frequency and contextual
similarity; and FINER offers means of combining FPTs and historical learning
sequences to enable more accurate prediction of student future performance.
Experiments on six real-world datasets show that FINER can outperform ten
state-of-the-art KT methods, increasing accuracy by 8.74% to 84.85%.

### Databases

### 1. [A Benchmark for Databases with Varying Value Lengths](http://arxiv.org/pdf/2508.07551v1)

Authors: Danushka Liyanage, Shubham Pandey, Joshua Goldstein, Michael Cahill, Akon Dey, Alan Fekete, Uwe Röhm

The performance of database management systems (DBMS) is traditionally
evaluated using benchmarks that focus on workloads with (almost) fixed record
lengths. However, some real-world workloads in key/value stores, document
databases, and graph databases exhibit significant variability in value
lengths, which can lead to performance anomalies, particularly when popular
records grow disproportionately large. Existing benchmarks fail to account for
this variability, leaving an important aspect of DBMS behavior underexplored.
  In this paper, we address this gap by extending the Yahoo! Cloud Serving
Benchmark (YCSB) to include an "extend" operation, which appends data to record
fields, simulating the growth of values over time. Using this modified
benchmark, we have measured the performance of three popular DBMS backends:
MongoDB, MariaDB with the InnoDB storage engine, and MariaDB with the MyRocks
storage engine. Our experiments alternate between extending values and
executing query workloads, revealing significant performance differences driven
by storage engine design and their handling of variable-sized values.
  Our key contribution is the introduction of a novel benchmarking approach to
evaluate the impact of growing value sizes and isolate the effect of querying
data with a distribution of data sizes from any cost associated with accessing
data after a history of updates. This highlights the need for more
representative benchmarks that capture the dynamic nature of real-world
workloads, providing valuable guidance for both practitioners and researchers.

### 2. [Heterogeneity in Entity Matching: A Survey and Experimental Analysis](http://arxiv.org/pdf/2508.08076v1)

Authors: Mohammad Hossein Moslemi, Amir Mousavi, Behshid Behkamal, Mostafa Milani

Entity matching (EM) is a fundamental task in data integration and analytics,
essential for identifying records that refer to the same real-world entity
across diverse sources. In practice, datasets often differ widely in structure,
format, schema, and semantics, creating substantial challenges for EM. We refer
to this setting as Heterogeneous EM (HEM). This survey offers a unified
perspective on HEM by introducing a taxonomy, grounded in prior work, that
distinguishes two primary categories -- representation and semantic
heterogeneity -- and their subtypes. The taxonomy provides a systematic lens
for understanding how variations in data form and meaning shape the complexity
of matching tasks. We then connect this framework to the FAIR principles --
Findability, Accessibility, Interoperability, and Reusability -- demonstrating
how they both reveal the challenges of HEM and suggest strategies for
mitigating them. Building on this foundation, we critically review recent EM
methods, examining their ability to address different heterogeneity types, and
conduct targeted experiments on state-of-the-art models to evaluate their
robustness and adaptability under semantic heterogeneity. Our analysis uncovers
persistent limitations in current approaches and points to promising directions
for future research, including multimodal matching, human-in-the-loop
workflows, deeper integration with large language models and knowledge graphs,
and fairness-aware evaluation in heterogeneous settings.

### 3. [MLego: Interactive and Scalable Topic Exploration Through Model Reuse](http://arxiv.org/pdf/2508.07654v1)

Authors: Fei Ye, Jiapan Liu, Yinan Jing, Zhenying He, Weirao Wang, X. Sean Wang

With massive texts on social media, users and analysts often rely on topic
modeling techniques to quickly extract key themes and gain insights.
Traditional topic modeling techniques, such as Latent Dirichlet Allocation
(LDA), provide valuable insights but are computationally expensive, making them
impractical for real-time data analysis. Although recent advances in
distributed training and fast sampling methods have improved efficiency,
real-time topic exploration remains a significant challenge. In this paper, we
present MLego, an interactive query framework designed to support real-time
topic modeling analysis by leveraging model materialization and reuse. Instead
of retraining models from scratch, MLego efficiently merges materialized topic
models to construct approximate results at interactive speeds. To further
enhance efficiency, we introduce a hierarchical plan search strategy for single
queries and an optimized query reordering technique for batch queries. We
integrate MLego into a visual analytics prototype system, enabling users to
explore large-scale textual datasets through interactive queries. Extensive
experiments demonstrate that MLego significantly reduces computation costs
while maintaining high-quality topic modeling results. MLego enhances existing
visual analytics approaches, which primarily focus on user-driven topic
modeling, by enabling real-time, query-driven exploration. This complements
traditional methods and bridges the gap between scalable topic modeling and
interactive data analysis.

### 4. [TQL: Towards Type-Driven Data Discovery](http://arxiv.org/pdf/2508.08054v1)

Authors: Andrew Kang, Sainyam Galhotra

Existing query languages for data discovery exhibit system-driven designs
that emphasize database features and functionality over user needs. We propose
a re-prioritization of the client through an introduction of a language-driven
approach to data discovery systems that can leverage powerful results from
programming languages research. In this paper, we describe TQL, a flexible and
practical query language which incorporates a type-like system to encompass
downstream transformation-context in its discovery queries. The syntax and
semantics of TQL (including the underlying evaluation model), are formally
defined, and a sketch of its implementation is also provided. Additionally, we
provide comparisons to existing languages for data retrieval and data discovery
to examine the advantages of TQL's expanded expressive power in real-life
settings.

### 5. [Towards General-Purpose Data Discovery: A Programming Languages Approach](http://arxiv.org/pdf/2508.08074v1)

Authors: Andrew Kang, Yashnil Saha, Sainyam Galhotra

Efficient and effective data discovery is critical for many modern
applications in machine learning and data science. One major bottleneck to the
development of a general-purpose data discovery tool is the absence of an
expressive formal language, and corresponding implementation, for
characterizing and solving generic discovery queries. To this end, we present
TQL, a domain-specific language for data discovery well-designed to leverage
and exploit the results of programming languages research in both its syntax
and semantics. In this paper, we fully and formally characterize the core
language through an algebraic model, Imperative Relational Algebra with Types
(ImpRAT), and implement a modular proof-of-concept system prototype.

### 6. [A Rule-Based Approach to Specifying Preferences over Conflicting Facts and Querying Inconsistent Knowledge Bases](http://arxiv.org/pdf/2508.07742v1)

Authors: Meghyn Bienvenu, Camille Bourgaux, Katsumi Inoue, Robin Jean

Repair-based semantics have been extensively studied as a means of obtaining
meaningful answers to queries posed over inconsistent knowledge bases (KBs).
While several works have considered how to exploit a priority relation between
facts to select optimal repairs, the question of how to specify such
preferences remains largely unaddressed. This motivates us to introduce a
declarative rule-based framework for specifying and computing a priority
relation between conflicting facts. As the expressed preferences may contain
undesirable cycles, we consider the problem of determining when a set of
preference rules always yields an acyclic relation, and we also explore a
pragmatic approach that extracts an acyclic relation by applying various cycle
removal techniques. Towards an end-to-end system for querying inconsistent KBs,
we present a preliminary implementation and experimental evaluation of the
framework, which employs answer set programming to evaluate the preference
rules, apply the desired cycle resolution techniques to obtain a priority
relation, and answer queries under prioritized-repair semantics.

### 7. [From Source to Target: Leveraging Transfer Learning for Predictive Process Monitoring in Organizations](http://arxiv.org/pdf/2508.08061v1)

Authors: Sven Weinzierl, Sandra Zilker, Annina Liessmann, Martin Käppel, Weixin Wang, Martin Matzner

Event logs reflect the behavior of business processes that are mapped in
organizational information systems. Predictive process monitoring (PPM)
transforms these data into value by creating process-related predictions that
provide the insights required for proactive interventions at process runtime.
Existing PPM techniques require sufficient amounts of event data or other
relevant resources that might not be readily available, preventing some
organizations from utilizing PPM. The transfer learning-based PPM technique
presented in this paper allows organizations without suitable event data or
other relevant resources to implement PPM for effective decision support. The
technique is instantiated in two real-life use cases, based on which numerical
experiments are performed using event logs for IT service management processes
in an intra- and inter-organizational setting. The results of the experiments
suggest that knowledge of one business process can be transferred to a similar
business process in the same or a different organization to enable effective
PPM in the target context. With the proposed technique, organizations can
benefit from transfer learning in an intra- and inter-organizational setting,
where resources like pre-trained models are transferred within and across
organizational boundaries.

### Distributed, Parallel, and Cluster Computing

### 1. [Coordinated Power Management on Heterogeneous Systems](http://arxiv.org/pdf/2508.07605v1)

Authors: Zhong Zheng, Michael E. Papka, Zhiling Lan

Performance prediction is essential for energy-efficient computing in
heterogeneous computing systems that integrate CPUs and GPUs. However,
traditional performance modeling methods often rely on exhaustive offline
profiling, which becomes impractical due to the large setting space and the
high cost of profiling large-scale applications. In this paper, we present
OPEN, a framework consists of offline and online phases. The offline phase
involves building a performance predictor and constructing an initial dense
matrix. In the online phase, OPEN performs lightweight online profiling, and
leverages the performance predictor with collaborative filtering to make
performance prediction. We evaluate OPEN on multiple heterogeneous systems,
including those equipped with A100 and A30 GPUs. Results show that OPEN
achieves prediction accuracy up to 98.29\%. This demonstrates that OPEN
effectively reduces profiling cost while maintaining high accuracy, making it
practical for power-aware performance modeling in modern HPC environments.
Overall, OPEN provides a lightweight solution for performance prediction under
power constraints, enabling better runtime decisions in power-aware computing
environments.

### 2. [Towards Lock Modularization for Heterogeneous Environments](http://arxiv.org/pdf/2508.07756v1)

Authors: Hanze Zhang, Rong Chen, Haibo Chen

Modern hardware environments are becoming increasingly heterogeneous, leading
to the emergence of applications specifically designed to exploit this
heterogeneity. Efficiently adopting locks in these applications poses distinct
challenges. The uneven distribution of resources in such environments can
create bottlenecks for lock operations, severely hindering application
performance. Existing solutions are often tailored to specific types of
hardware, which underutilizes resources on other components within
heterogeneous environments.
  This paper introduces a new design principle: decomposing locks across
hardware components to fully utilize unevenly distributed resources in
heterogeneous environments. Following this principle, we propose lock
modularization, a systematic approach that decomposes a lock into independent
modules and assigns them to appropriate hardware components. This approach
aligns the resource requirements of lock modules with the attributes of
specific hardware components, maximizing strengths while minimizing weaknesses.

### 3. [On the Operational Resilience of CBDC: Threats and Prospects of Formal Validation for Offline Payments](http://arxiv.org/pdf/2508.08064v1)

Authors: Marco Bernardo, Federico Calandra, Andrea Esposito, Francesco Fabris

Information and communication technologies are by now employed in most
activities, including economics and finance. Despite the extraordinary power of
modern computers and the vast amount of memory, some results of theoretical
computer science imply the impossibility of certifying software quality in
general. With the exception of safety-critical systems, this has primarily
concerned the information processed by confined systems, with limited
socio-economic consequences. In the emerging era of technologies for exchanging
digital money and tokenized assets over the Internet - such as central bank
digital currencies (CBDCs) - even a minor bug could trigger a financial
collapse. Although the aforementioned impossibility results cannot be overcome
in an absolute sense, there exist formal methods that can provide assertions of
computing systems correctness. We advocate their use to validate the
operational resilience of software infrastructures enabling CBDCs, with special
emphasis on offline payments as they constitute a very critical issue.

### 4. [Taming Cold Starts: Proactive Serverless Scheduling with Model Predictive Control](http://arxiv.org/pdf/2508.07640v1)

Authors: Chanh Nguyen, Monowar Bhuyan, Erik Elmroth

Serverless computing has transformed cloud application deployment by
introducing a fine-grained, event-driven execution model that abstracts away
infrastructure management. Its on-demand nature makes it especially appealing
for latency-sensitive and bursty workloads. However, the cold start problem,
i.e., where the platform incurs significant delay when provisioning new
containers, remains the Achilles' heel of such platforms.
  This paper presents a predictive serverless scheduling framework based on
Model Predictive Control to proactively mitigate cold starts, thereby improving
end-to-end response time. By forecasting future invocations, the controller
jointly optimizes container prewarming and request dispatching, improving
latency while minimizing resource overhead.
  We implement our approach on Apache OpenWhisk, deployed on a Kubernetes-based
testbed. Experimental results using real-world function traces and synthetic
workloads demonstrate that our method significantly outperforms
state-of-the-art baselines, achieving up to 85% lower tail latency and a 34%
reduction in resource usage.

### 5. [Perpetual exploration in anonymous synchronous networks with a Byzantine black hole](http://arxiv.org/pdf/2508.07703v1)

Authors: Adri Bhattacharya, Pritam Goswami, Evangelos Bampas, Partha Sarathi Mandal

In this paper, we investigate: ``How can a group of initially co-located
mobile agents perpetually explore an unknown graph, when one stationary node
occasionally behaves maliciously, under an adversary's control?'' We call this
node a ``Byzantine black hole (BBH)'' and at any given round it may choose to
destroy all visiting agents, or none. This subtle power can drastically
undermine classical exploration strategies designed for an always active black
hole. We study this perpetual exploration problem in the presence of at most
one BBH, without initial knowledge of the network size. Since the underlying
graph may be 1-connected, perpetual exploration of the entire graph may be
infeasible. We thus define two variants: \pbmPerpExpl\ and \pbmPerpExplHome. In
the former, the agents are tasked to perform perpetual exploration of at least
one component, obtained after the exclusion of the BBH. In the latter, the
agents are tasked to perform perpetual exploration of the component which
contains the \emph{home} node, where agents are initially co-located.
Naturally, \pbmPerpExplHome\ is a special case of \pbmPerpExpl. Agents operate
under a synchronous scheduler and communicate in a face-to-face model. Our goal
is to determine the minimum number of agents necessary and sufficient to solve
these problems. In acyclic networks, we obtain optimal algorithms that solve
\pbmPerpExpl\ with $4$ agents, and \pbmPerpExplHome\ with $6$ agents in trees.
The lower bounds hold even in path graphs. In general graphs, we give a
non-trivial lower bound of $2\Delta-1$ agents for \pbmPerpExpl, and an upper
bound of $3\Delta+3$ agents for \pbmPerpExplHome. To our knowledge, this is the
first study of a black-hole variant in arbitrary networks without initial
topological knowledge.

### 6. [GPU-Accelerated Syndrome Decoding for Quantum LDPC Codes below the 63 $μ$s Latency Threshold](http://arxiv.org/pdf/2508.07879v1)

Authors: Oscar Ferraz, Bruno Coutinho, Gabriel Falcao, Marco Gomes, Francisco A. Monteiro, Vitor Silva

This paper presents a GPU-accelerated decoder for quantum low-density
parity-check (QLDPC) codes that achieves sub-$63$ $\mu$s latency, below the
surface code decoder's real-time threshold demonstrated on Google's Willow
quantum processor. While surface codes have demonstrated below-threshold
performance, the encoding rates approach zero as code distances increase,
posing challenges for scalability. Recently proposed QLDPC codes, such as those
by Panteleev and Kalachev, offer constant-rate encoding and asymptotic goodness
but introduce higher decoding complexity. To address such limitation, this work
presents a parallelized belief propagation decoder leveraging syndrome
information on commodity GPU hardware. Parallelism was exploited to maximize
performance within the limits of target latency, allowing decoding latencies
under $50$ $\mu$s for [[$784$, $24$, $24$]] codes and as low as $23.3$ $\mu$s
for smaller codes, meeting the tight timing constraints of superconducting
qubit cycles. These results show that real-time, scalable decoding of
asymptotically good quantum codes is achievable using widely available
commodity hardware, advancing the feasibility of fault-tolerant quantum
computation beyond surface codes.

### 7. [Performance Evaluation of Brokerless Messaging Libraries](http://arxiv.org/pdf/2508.07934v1)

Authors: Lorenzo La Corte, Syed Aftab Rashid, Andrei-Marian Dan

Messaging systems are essential for efficiently transferring large volumes of
data, ensuring rapid response times and high-throughput communication. The
state-of-the-art on messaging systems mainly focuses on the performance
evaluation of brokered messaging systems, which use an intermediate broker to
guarantee reliability and quality of service. However, over the past decade,
brokerless messaging systems have emerged, eliminating the single point of
failure and trading off reliability guarantees for higher performance. Still,
the state-of-the-art on evaluating the performance of brokerless systems is
scarce. In this work, we solely focus on brokerless messaging systems. First,
we perform a qualitative analysis of several possible candidates, to find the
most promising ones. We then design and implement an extensive open-source
benchmarking suite to systematically and fairly evaluate the performance of the
chosen libraries, namely, ZeroMQ, NanoMsg, and NanoMsg-Next-Generation (NNG).
We evaluate these libraries considering different metrics and workload
conditions, and provide useful insights into their limitations. Our analysis
enables practitioners to select the most suitable library for their
requirements.

### 8. [Optimizing Federated Learning for Scalable Power-demand Forecasting in Microgrids](http://arxiv.org/pdf/2508.08022v1)

Authors: Roopkatha Banerjee, Sampath Koti, Gyanendra Singh, Anirban Chakraborty, Gurunath Gurrala, Bhushan Jagyasi, Yogesh Simmhan

Real-time monitoring of power consumption in cities and micro-grids through
the Internet of Things (IoT) can help forecast future demand and optimize grid
operations. But moving all consumer-level usage data to the cloud for
predictions and analysis at fine time scales can expose activity patterns.
Federated Learning~(FL) is a privacy-sensitive collaborative DNN training
approach that retains data on edge devices, trains the models on private data
locally, and aggregates the local models in the cloud. But key challenges
exist: (i) clients can have non-independently identically distributed~(non-IID)
data, and (ii) the learning should be computationally cheap while scaling to
1000s of (unseen) clients. In this paper, we develop and evaluate several
optimizations to FL training across edge and cloud for time-series demand
forecasting in micro-grids and city-scale utilities using DNNs to achieve a
high prediction accuracy while minimizing the training cost. We showcase the
benefit of using exponentially weighted loss while training and show that it
further improves the prediction of the final model. Finally, we evaluate these
strategies by validating over 1000s of clients for three states in the US from
the OpenEIA corpus, and performing FL both in a pseudo-distributed setting and
a Pi edge cluster. The results highlight the benefits of the proposed methods
over baselines like ARIMA and DNNs trained for individual consumers, which are
not scalable.

### 9. [Fully-Fluctuating Participation in Sleepy Consensus](http://arxiv.org/pdf/2508.08068v1)

Authors: Yuval Efron, Joachim Neu, Toniann Pitassi

Proof-of-work allows Bitcoin to boast security amidst arbitrary fluctuations
in participation of miners throughout time, so long as, at any point in time, a
majority of hash power is honest. In recent years, however, the pendulum has
shifted in favor of proof-of-stake-based consensus protocols. There, the sleepy
model is the most prominent model for handling fluctuating participation of
nodes. However, to date, no protocol in the sleepy model rivals Bitcoin in its
robustness to drastic fluctuations in participation levels, with
state-of-the-art protocols making various restrictive assumptions. In this
work, we present a new adversary model, called external adversary. Intuitively,
in our model, corrupt nodes do not divulge information about their secret keys.
In this model, we show that protocols in the sleepy model can meaningfully
claim to remain secure against fully fluctuating participation, without
compromising efficiency or corruption resilience. Our adversary model is quite
natural, and arguably naturally captures the process via which malicious
behavior arises in protocols, as opposed to traditional worst-case modeling. On
top of which, the model is also theoretically appealing, circumventing a
barrier established in a recent work of Malkhi, Momose, and Ren.

### 10. [Multi-Hop Privacy Propagation for Differentially Private Federated Learning in Social Networks](http://arxiv.org/pdf/2508.07676v1)

Authors: Chenchen Lin, Xuehe Wang

Federated learning (FL) enables collaborative model training across
decentralized clients without sharing local data, thereby enhancing privacy and
facilitating collaboration among clients connected via social networks.
However, these social connections introduce privacy externalities: a client's
privacy loss depends not only on its privacy protection strategy but also on
the privacy decisions of others, propagated through the network via multi-hop
interactions. In this work, we propose a socially-aware privacy-preserving FL
mechanism that systematically quantifies indirect privacy leakage through a
multi-hop propagation model. We formulate the server-client interaction as a
two-stage Stackelberg game, where the server, as the leader, optimizes
incentive policies, and clients, as followers, strategically select their
privacy budgets, which determine their privacy-preserving levels by controlling
the magnitude of added noise. To mitigate information asymmetry in networked
privacy estimation, we introduce a mean-field estimator to approximate the
average external privacy risk. We theoretically prove the existence and
convergence of the fixed point of the mean-field estimator and derive
closed-form expressions for the Stackelberg Nash Equilibrium. Despite being
designed from a client-centric incentive perspective, our mechanism achieves
approximately-optimal social welfare, as revealed by Price of Anarchy (PoA)
analysis. Experiments on diverse datasets demonstrate that our approach
significantly improves client utilities and reduces server costs while
maintaining model performance, outperforming both Social-Agnostic (SA)
baselines and methods that account for social externalities.

### Discrete Mathematics

### 1. [Remarks on the Brouwer Conjecture](http://arxiv.org/pdf/2508.07550v1)

Authors: Oliver Knill

The Brouwer conjecture (BC) in spectral graph theory claims that the sum of
the largest k Kirchhoff eigenvalues of a graph are bounded above by the number
m of edges plus k(k+1)/2. We show that (BC) holds for all graphs with n
vertices if n is larger or equal than 4 times the square of the maximal vertex
degree. We also note that the weaker upper bound m+k(k+1) holds
unconditionally. We also note that (BC) for graphs implies (BC) for quivers.

### 2. [Coloring Graphs with no Totally Odd Clique Immersion](http://arxiv.org/pdf/2508.08119v1)

Authors: Caleb McFarland

We prove that graphs that do not contain a totally odd immersion of $K_t$ are
$\mathcal{O}(t)$-colorable. In particular, we show that any graph with no
totally odd immersion of $K_t$ is the union of a bipartite graph and a graph
which forbids an immersion of $K_{\mathcal{O}(t)}$. Our results are
algorithmic, and we give a fixed-parameter tractable algorithm (in $t$) to find
such a decomposition.

### Data Structures and Algorithms

### 1. [Simple Algorithms for Fully Dynamic Edge Connectivity](http://arxiv.org/pdf/2508.07783v1)

Authors: Yotam Kenneth-Mordoch, Robert Krauthgamer

In the fully dynamic edge connectivity problem, the input is a simple graph
$G$ undergoing edge insertions and deletions, and the goal is to maintain its
edge connectivity, denoted $\lambda_G$. We present two simple randomized
algorithms solving this problem. The first algorithm maintains the edge
connectivity in worst-case update time $\tilde{O}(n)$ per edge update, matching
the known bound but with simpler analysis. Our second algorithm achieves
worst-case update time $\tilde{O}(n/\lambda_G)$ and worst-case query time
$\tilde{O}(n^2/\lambda_G^2)$, which is the first algorithm with worst-case
update and query time $o(n)$ for large edge connectivity, namely, $\lambda_G =
\omega(\sqrt{n})$.

### 2. [Nearly Optimal Bounds for Stochastic Online Sorting](http://arxiv.org/pdf/2508.07823v1)

Authors: Yang Hu

In the online sorting problem, we have an array $A$ of $n$ cells, and receive
a stream of $n$ items $x_1,\dots,x_n\in [0,1]$. When an item arrives, we need
to immediately and irrevocably place it into an empty cell. The goal is to
minimize the sum of absolute differences between adjacent items, which is
called the \emph{cost} of the algorithm. It has been shown by Aamand,
Abrahamsen, Beretta, and Kleist (SODA 2023) that when the stream
$x_1,\dots,x_n$ is generated adversarially, the optimal cost bound for any
deterministic algorithm is $\Theta(\sqrt{n})$.
  In this paper, we study the stochastic version of online sorting, where the
input items $x_1,\dots,x_n$ are sampled uniformly at random. Despite the
intuition that the stochastic version should yield much better cost bounds, the
previous best algorithm for stochastic online sorting by Abrahamsen, Bercea,
Beretta, Klausen and Kozma (ESA 2024) only achieves $\tilde{O}(n^{1/4})$ cost,
which seems far from optimal. We show that stochastic online sorting indeed
allows for much more efficient algorithms, by presenting an algorithm that
achieves expected cost $\log n\cdot 2^{O(\log^* n)}$. We also prove a cost
lower bound of $\Omega(\log n)$, thus show that our algorithm is nearly
optimal.

### 3. [Sparsifying Sums of Positive Semidefinite Matrices](http://arxiv.org/pdf/2508.08169v1)

Authors: Arpon Basu, Pravesh K. Kothari, Yang P. Liu, Raghu Meka

In this paper, we revisit spectral sparsification for sums of arbitrary
positive semidefinite (PSD) matrices. Concretely, for any collection of PSD
matrices $\mathcal{A} = \{A_1, A_2, \ldots, A_r\} \subset \mathbb{R}^{n \times
n}$, given any subset $T \subseteq [r]$, our goal is to find sparse weights
$\mu \in \mathbb{R}_{\geq 0}^r$ such that $(1 - \epsilon) \sum_{i \in T} A_i
\preceq \sum_{i \in T} \mu_i A_i \preceq (1 + \epsilon) \sum_{i \in T} A_i.$
This generalizes spectral sparsification of graphs which corresponds to
$\mathcal{A}$ being the set of Laplacians of edges. It also captures
sparsifying Cayley graphs by choosing a subset of generators. The former has
been extensively studied with optimal sparsifiers known. The latter has
received attention recently and was solved for a few special groups (e.g.,
$\mathbb{F}_2^n$).
  Prior work shows any sum of PSD matrices can be sparsified down to $O(n)$
elements. This bound however turns out to be too coarse and in particular
yields no non-trivial bound for building Cayley sparsifiers for Cayley graphs.
  In this work, we develop a new, instance-specific (i.e., specific to a given
collection $\mathcal{A}$) theory of PSD matrix sparsification based on a new
parameter $N^*(\mathcal{A})$ which we call connectivity threshold that
generalizes the threshold of the number of edges required to make a graph
connected.
  Our main result gives a sparsifier that uses at most
$O(\epsilon^{-2}N^*(\mathcal{A}) (\log n)(\log r))$ matrices and is
constructible in randomized polynomial time. We also show that we need
$N^*(\mathcal{A})$ elements to sparsify for any $\epsilon < 0.99$.
  As the main application of our framework, we prove that any Cayley graph can
be sparsified to $O(\epsilon^{-2}\log^4 N)$ generators. Previously, a
non-trivial bound on Cayley sparsifiers was known only in the case when the
group is $\mathbb{F}_2^n$.

### 4. [Sparsifying Cayley Graphs on Every Group](http://arxiv.org/pdf/2508.08078v1)

Authors: Jun-Ting Hsieh, Daniel Z. Lee, Sidhanth Mohanty, Aaron Putterman, Rachel Yun Zhang

A classic result in graph theory, due to Batson, Spielman, and Srivastava
(STOC 2009) shows that every graph admits a $(1 \pm \varepsilon)$ cut (or
spectral) sparsifier which preserves only $O(n / \varepsilon^2)$ reweighted
edges. However, when applying this result to \emph{Cayley graphs}, the
resulting sparsifier is no longer necessarily a Cayley graph -- it can be an
arbitrary subset of edges.
  Thus, a recent line of inquiry, and one which has only seen minor progress,
asks: for any group $G$, do all Cayley graphs over the group $G$ admit
sparsifiers which preserve only $\mathrm{polylog}(|G|)/\varepsilon^2$ many
re-weighted generators?
  As our primary contribution, we answer this question in the affirmative,
presenting a proof of the existence of such Cayley graph spectral sparsifiers,
along with an efficient algorithm for finding them. Our algorithm even extends
to \emph{directed} Cayley graphs, if we instead ask only for cut sparsification
instead of spectral sparsification.
  We additionally study the sparsification of linear equations over non-abelian
groups. In contrast to the abelian case, we show that for non-abelian valued
equations, super-polynomially many linear equations must be preserved in order
to approximately preserve the number of satisfied equations for any input.
Together with our Cayley graph sparsification result, this provides a formal
separation between Cayley graph sparsification and sparsifying linear
equations.

### Emerging Technologies

### 1. [Enhancing Mega-Satellite Networks with Generative Semantic Communication: A Networking Perspective](http://arxiv.org/pdf/2508.07573v1)

Authors: Binquan Guo, Wanting Yang, Zehui Xiong, Zhou Zhang, Baosheng Li, Zhu Han, Rahim Tafazolli, Tony Q. S. Quek

The advance of direct satellite-to-device communication has positioned
mega-satellite constellations as a cornerstone of 6G wireless communication,
enabling seamless global connectivity even in remote and underserved areas.
However, spectrum scarcity and capacity constraints imposed by the Shannon's
classical information theory remain significant challenges for supporting the
massive data demands of multimedia-rich wireless applications. Generative
Semantic Communication (GSC), powered by artificial intelligence-based
generative foundation models, represents a paradigm shift from transmitting raw
data to exchanging semantic meaning. GSC can not only reduce bandwidth
consumption, but also enhance key semantic features in multimedia content,
thereby offering a promising solution to overcome the limitations of
traditional satellite communication systems. This article investigates the
integration of GSC into mega-satellite constellations from a networking
perspective. We propose a GSC-empowered satellite networking architecture and
identify key enabling technologies, focusing on GSC-empowered network modeling
and GSC-aware networking strategies. We construct a discrete temporal graph to
model semantic encoders and decoders, distinct knowledge bases, and resource
variations in mega-satellite networks. Based on this framework, we develop
model deployment for semantic encoders and decoders and GSC-compatible routing
schemes, and then present performance evaluations. Finally, we outline future
research directions for advancing GSC-empowered satellite networks.

### 2. [KIRETT: Knowledge-Graph-Based Smart Treatment Assistant for Intelligent Rescue Operations](http://arxiv.org/pdf/2508.07834v1)

Authors: Mubaris Nadeem, Johannes Zenkert, Lisa Bender, Christian Weber, Madjid Fathi

Over the years, the need for rescue operations throughout the world has
increased rapidly. Demographic changes and the resulting risk of injury or
health disorders form the basis for emergency calls. In such scenarios, first
responders are in a rush to reach the patient in need, provide first aid, and
save lives. In these situations, they must be able to provide personalized and
optimized healthcare in the shortest possible time and estimate the patients
condition with the help of freshly recorded vital data in an emergency
situation. However, in such a timedependent situation, first responders and
medical experts cannot fully grasp their knowledge and need assistance and
recommendation for further medical treatments. To achieve this, on the spot
calculated, evaluated, and processed knowledge must be made available to
improve treatments by first responders. The Knowledge Graph presented in this
article as a central knowledge representation provides first responders with an
innovative knowledge management that enables intelligent treatment
recommendations with an artificial intelligence-based pre-recognition of the
situation.

### 3. [DoorDet: Semi-Automated Multi-Class Door Detection Dataset via Object Detection and Large Language Models](http://arxiv.org/pdf/2508.07714v1)

Authors: Licheng Zhang, Bach Le, Naveed Akhtar, Tuan Ngo

Accurate detection and classification of diverse door types in floor plans
drawings is critical for multiple applications, such as building compliance
checking, and indoor scene understanding. Despite their importance, publicly
available datasets specifically designed for fine-grained multi-class door
detection remain scarce. In this work, we present a semi-automated pipeline
that leverages a state-of-the-art object detector and a large language model
(LLM) to construct a multi-class door detection dataset with minimal manual
effort. Doors are first detected as a unified category using a deep object
detection model. Next, an LLM classifies each detected instance based on its
visual and contextual features. Finally, a human-in-the-loop stage ensures
high-quality labels and bounding boxes. Our method significantly reduces
annotation cost while producing a dataset suitable for benchmarking neural
models in floor plan analysis. This work demonstrates the potential of
combining deep learning and multimodal reasoning for efficient dataset
construction in complex real-world domains.

### 4. [Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks](http://arxiv.org/pdf/2508.08029v1)

Authors: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

The introduction of 5G and the Open Radio Access Network (O-RAN) architecture
has enabled more flexible and intelligent network deployments. However, the
increased complexity and openness of these architectures also introduce novel
security challenges, such as data manipulation attacks on the semi-standardised
Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In
particular, malicious xApps can exploit this vulnerability by introducing
subtle Unicode-wise alterations (hypoglyphs) into the data that are being used
by traditional machine learning (ML)-based anomaly detection methods. These
Unicode-wise manipulations can potentially bypass detection and cause failures
in anomaly detection systems based on traditional ML, such as AutoEncoders,
which are unable to process hypoglyphed data without crashing. We investigate
the use of Large Language Models (LLMs) for anomaly detection within the O-RAN
architecture to address this challenge. We demonstrate that LLM-based xApps
maintain robust operational performance and are capable of processing
manipulated messages without crashing. While initial detection accuracy
requires further improvements, our results highlight the robustness of LLMs to
adversarial attacks such as hypoglyphs in input data. There is potential to use
their adaptability through prompt engineering to further improve the accuracy,
although this requires further research. Additionally, we show that LLMs
achieve low detection latency (under 0.07 seconds), making them suitable for
Near-Real-Time (Near-RT) RIC deployments.

### 5. [ELF: Efficient Logic Synthesis by Pruning Redundancy in Refactoring](http://arxiv.org/pdf/2508.08073v1)

Authors: Dimitris Tsaras, Xing Li, Lei Chen, Zhiyao Xie, Mingxuan Yuan

In electronic design automation, logic optimization operators play a crucial
role in minimizing the gate count of logic circuits. However, their computation
demands are high. Operators such as refactor conventionally form iterative cuts
for each node, striving for a more compact representation - a task which often
fails 98% on average. Prior research has sought to mitigate computational cost
through parallelization. In contrast, our approach leverages a classifier to
prune unsuccessful cuts preemptively, thus eliminating unnecessary resynthesis
operations. Experiments on the refactor operator using the EPFL benchmark suite
and 10 large industrial designs demonstrate that this technique can speedup
logic optimization by 3.9x on average compared with the state-of-the-art ABC
implementation.

### 6. [Frequency-Domain Analysis of Time-Dependent Multiomic Data in Progressive Neurodegenerative Diseases: A Proposed Quantum-Classical Hybrid Approach with Quaternionic Extensions](http://arxiv.org/pdf/2508.07948v1)

Authors: John D. Mayfield

Progressive neurodegenerative diseases, including Alzheimer's disease (AD),
multiple sclerosis (MS), Parkinson's disease (PD), and amyotrophic lateral
sclerosis (ALS), exhibit complex, nonlinear trajectories that challenge
deterministic modeling. Traditional time-domain analyses of multiomic and
neuroimaging data often fail to capture hidden oscillatory patterns, limiting
predictive accuracy. We propose a theoretical mathematical framework that
transforms time-series data into frequency or s-domain using Fourier and
Laplace transforms, models neuronal dynamics via Hamiltonian formulations, and
employs quantum-classical hybrid computing with variational quantum
eigensolvers (VQE) for enhanced pattern detection. This theoretical construct
serves as a foundation for future empirical works in quantum-enhanced analysis
of neurodegenerative diseases. We extend this to quaternionic representations
with three imaginary axes ($i, j, k$) to model multistate Hamiltonians in
multifaceted disorders, drawing from quantum neuromorphic computing to capture
entangled neural dynamics \citep{Pehle2020, Emani2019}. This approach leverages
quantum advantages in handling high-dimensional amplitude-phase data, enabling
outlier detection and frequency signature analysis. Potential clinical
applications include identifying high-risk patients with rapid progression or
therapy resistance using s-domain biomarkers, supported by quantum machine
learning (QML) precedents achieving up to 99.89% accuracy in Alzheimer's
classification \citep{Belay2024, Bhowmik2025}. This framework aims to lay the
groundwork for redefining precision medicine for neurodegenerative diseases
through future validations.

### Formal Languages and Automata Theory

### 1. [Hexagonal Picture Scanning Automata](http://arxiv.org/pdf/2508.07779v1)

Authors: Deepalakshmi D, Lisa Mathew

Two new classes of finite automata, called General hexagonal Boustrophedon
finite automata and General hexagonal returning finite automata operating on
hexagonal grids, are introduced and analyzed. The work establishes the
theoretical foundations for these automata models, examines their computational
properties, and investigates the relationships and equivalences between the
language families they define. The research contributes to the broader
understanding of two-dimensional automata theory by extending classical finite
automaton concepts to hexagonal geometric structures with specialized traversal
patterns.

### Graphics

### 1. [Verification Method for Graph Isomorphism Criteria](http://arxiv.org/pdf/2508.07615v1)

Authors: Chuanfu Hu, Aimin Hou

The criteria for determining graph isomorphism are crucial for solving graph
isomorphism problems. The necessary condition is that two isomorphic graphs
possess invariants, but their function can only be used to filtrate and
subdivide candidate spaces. The sufficient conditions are used to rebuild the
isomorphic reconstruction of special graphs, but their drawback is that the
isomorphic functions of subgraphs may not form part of the isomorphic functions
of the parent graph. The use of sufficient or necessary conditions generally
results in backtracking to ensure the correctness of the decision algorithm.
The sufficient and necessary conditions can ensure that the determination of
graph isomorphism does not require backtracking, but the correctness of its
proof process is difficult to guarantee. This article proposes a verification
method that can correctly determine whether the judgment conditions proposed by
previous researchers are sufficient and necessary conditions. A subdivision
method has also been proposed in this article, which can obtain more
subdivisions for necessary conditions and effectively reduce the size of
backtracking space.

### 2. [Vertex Features for Neural Global Illumination](http://arxiv.org/pdf/2508.07852v1)

Authors: Rui Su, Honghao Dong, Haojie Jin, Yisong Chen, Guoping Wang, Sheng Li

Recent research on learnable neural representations has been widely adopted
in the field of 3D scene reconstruction and neural rendering applications.
However, traditional feature grid representations often suffer from substantial
memory footprint, posing a significant bottleneck for modern parallel computing
hardware. In this paper, we present neural vertex features, a generalized
formulation of learnable representation for neural rendering tasks involving
explicit mesh surfaces. Instead of uniformly distributing neural features
throughout 3D space, our method stores learnable features directly at mesh
vertices, leveraging the underlying geometry as a compact and structured
representation for neural processing. This not only optimizes memory
efficiency, but also improves feature representation by aligning compactly with
the surface using task-specific geometric priors. We validate our neural
representation across diverse neural rendering tasks, with a specific emphasis
on neural radiosity. Experimental results demonstrate that our method reduces
memory consumption to only one-fifth (or even less) of grid-based
representations, while maintaining comparable rendering quality and lowering
inference overhead.

### 3. [Matrix-3D: Omnidirectional Explorable 3D World Generation](http://arxiv.org/pdf/2508.08086v1)

Authors: Zhongqi Yang, Wenhang Ge, Yuqi Li, Jiaqi Chen, Haoyuan Li, Mengyin An, Fei Kang, Hua Xue, Baixin Xu, Yuyang Yin, Eric Li, Yang Liu, Yikai Wang, Hao-Xiang Guo, Yahui Zhou

Explorable 3D world generation from a single image or text prompt forms a
cornerstone of spatial intelligence. Recent works utilize video model to
achieve wide-scope and generalizable 3D world generation. However, existing
approaches often suffer from a limited scope in the generated scenes. In this
work, we propose Matrix-3D, a framework that utilize panoramic representation
for wide-coverage omnidirectional explorable 3D world generation that combines
conditional video generation and panoramic 3D reconstruction. We first train a
trajectory-guided panoramic video diffusion model that employs scene mesh
renders as condition, to enable high-quality and geometrically consistent scene
video generation. To lift the panorama scene video to 3D world, we propose two
separate methods: (1) a feed-forward large panorama reconstruction model for
rapid 3D scene reconstruction and (2) an optimization-based pipeline for
accurate and detailed 3D scene reconstruction. To facilitate effective
training, we also introduce the Matrix-Pano dataset, the first large-scale
synthetic collection comprising 116K high-quality static panoramic video
sequences with depth and trajectory annotations. Extensive experiments
demonstrate that our proposed framework achieves state-of-the-art performance
in panoramic video generation and 3D world generation. See more in
https://matrix-3d.github.io.

### 4. [Emergent morphogenesis via planar fabrication enabled by a reduced model of composites](http://arxiv.org/pdf/2508.08198v1)

Authors: Yupeng Zhang, Adam Alon, M. Khalid Jawed

The ability to engineer complex three-dimensional shapes from planar sheets
with precise, programmable control underpins emerging technologies in soft
robotics, reconfigurable devices, and functional materials. Here, we present a
reduced-order numerical and experimental framework for a bilayer system
consisting of a stimuli-responsive thermoplastic sheet (Shrinky Dink) bonded to
a kirigami-patterned, inert plastic layer. Upon uniform heating, the active
layer contracts while the patterned layer constrains in-plane stretch but
allows out-of-plane bending, yielding programmable 3D morphologies from simple
planar precursors. Our approach enables efficient computational design and
scalable manufacturing of 3D forms with a single-layer reduced model that
captures the coupled mechanics of stretching and bending. Unlike traditional
bilayer modeling, our framework collapses the multilayer composite into a
single layer of nodes and elements, reducing the degrees of freedom and
enabling simulation on a 2D geometry. This is achieved by introducing a novel
energy formulation that captures the coupling between in-plane stretch mismatch
and out-of-plane bending - extending beyond simple isotropic linear elastic
models. Experimentally, we establish a fully planar, repeatable fabrication
protocol using a stimuli-responsive thermoplastic and a laser-cut inert plastic
layer. The programmed strain mismatch drives an array of 3D morphologies, such
as bowls, canoes, and flower petals, all verified by both simulation and
physical prototypes.

### 5. [LL3M: Large Language 3D Modelers](http://arxiv.org/pdf/2508.08228v1)

Authors: Sining Lu, Guan Chen, Nam Anh Dinh, Itai Lang, Ari Holtzman, Rana Hanocka

We present LL3M, a multi-agent system that leverages pretrained large
language models (LLMs) to generate 3D assets by writing interpretable Python
code in Blender. We break away from the typical generative approach that learns
from a collection of 3D data. Instead, we reformulate shape generation as a
code-writing task, enabling greater modularity, editability, and integration
with artist workflows. Given a text prompt, LL3M coordinates a team of
specialized LLM agents to plan, retrieve, write, debug, and refine Blender
scripts that generate and edit geometry and appearance. The generated code
works as a high-level, interpretable, human-readable, well-documented
representation of scenes and objects, making full use of sophisticated Blender
constructs (e.g. B-meshes, geometry modifiers, shader nodes) for diverse,
unconstrained shapes, materials, and scenes. This code presents many avenues
for further agent and human editing and experimentation via code tweaks or
procedural parameters. This medium naturally enables a co-creative loop in our
system: agents can automatically self-critique using code and visuals, while
iterative user instructions provide an intuitive way to refine assets. A shared
code context across agents enables awareness of previous attempts, and a
retrieval-augmented generation knowledge base built from Blender API
documentation, BlenderRAG, equips agents with examples, types, and functions
empowering advanced modeling operations and code correctness. We demonstrate
the effectiveness of LL3M across diverse shape categories, style and material
edits, and user-driven refinements. Our experiments showcase the power of code
as a generative and interpretable medium for 3D asset creation. Our project
page is at https://threedle.github.io/ll3m.

### 6. [Sea-Undistort: A Dataset for Through-Water Image Restoration in High Resolution Airborne Bathymetric Mapping](http://arxiv.org/pdf/2508.07760v1)

Authors: Maximilian Kromer, Panagiotis Agrafiotis, Begüm Demir

Accurate image-based bathymetric mapping in shallow waters remains
challenging due to the complex optical distortions such as wave induced
patterns, scattering and sunglint, introduced by the dynamic water surface, the
water column properties, and solar illumination. In this work, we introduce
Sea-Undistort, a comprehensive synthetic dataset of 1200 paired 512x512
through-water scenes rendered in Blender. Each pair comprises a distortion-free
and a distorted view, featuring realistic water effects such as sun glint,
waves, and scattering over diverse seabeds. Accompanied by per-image metadata
such as camera parameters, sun position, and average depth, Sea-Undistort
enables supervised training that is otherwise infeasible in real environments.
We use Sea-Undistort to benchmark two state-of-the-art image restoration
methods alongside an enhanced lightweight diffusion-based framework with an
early-fusion sun-glint mask. When applied to real aerial data, the enhanced
diffusion model delivers more complete Digital Surface Models (DSMs) of the
seabed, especially in deeper areas, reduces bathymetric errors, suppresses
glint and scattering, and crisply restores fine seabed details. Dataset,
weights, and code are publicly available at
https://www.magicbathy.eu/Sea-Undistort.html.

### Computer Science and Game Theory

### 1. [Last-Iterate Convergence in Adaptive Regret Minimization for Approximate Extensive-Form Perfect Equilibrium](http://arxiv.org/pdf/2508.07699v1)

Authors: Hang Ren, Xiaozhen Sun, Tianzi Ma, Jiajia Zhang, Xuan Wang

The Nash Equilibrium (NE) assumes rational play in imperfect-information
Extensive-Form Games (EFGs) but fails to ensure optimal strategies for
off-equilibrium branches of the game tree, potentially leading to suboptimal
outcomes in practical settings. To address this, the Extensive-Form Perfect
Equilibrium (EFPE), a refinement of NE, introduces controlled perturbations to
model potential player errors. However, existing EFPE-finding algorithms, which
typically rely on average strategy convergence and fixed perturbations, face
significant limitations: computing average strategies incurs high computational
costs and approximation errors, while fixed perturbations create a trade-off
between NE approximation accuracy and the convergence rate of NE refinements.
  To tackle these challenges, we propose an efficient adaptive regret
minimization algorithm for computing approximate EFPE, achieving last-iterate
convergence in two-player zero-sum EFGs. Our approach introduces Reward
Transformation Counterfactual Regret Minimization (RTCFR) to solve perturbed
games and defines a novel metric, the Information Set Nash Equilibrium (ISNE),
to dynamically adjust perturbations. Theoretical analysis confirms convergence
to EFPE, and experimental results demonstrate that our method significantly
outperforms state-of-the-art algorithms in both NE and EFPE-finding tasks.

### 2. [Truthful Two-Obnoxious-Facility Location Games with Optional Preferences and Minimum Distance Constraint](http://arxiv.org/pdf/2508.08036v1)

Authors: Xiaojia Han, Wenjing Liu, Qizhi Fang

In this paper, we study a truthful two-obnoxious-facility location problem,
in which each agent has a private location in [0, 1] and a public optional
preference over two obnoxious facilities, and there is a minimum distance
constraint d between the two facilities. Each agent wants to be as far away as
possible from the facilities that affect her, and the utility of each agent is
the total distance from her to these facilities. The goal is to decide how to
place the facilities in [0, 1] so as to incentivize agents to report their
private locations truthfully as well as maximize the social utility. First, we
consider the special setting where d = 0, that is, the two facilities can be
located at any point in [0, 1]. We propose a deterministic strategyproof
mechanism with approximation ratio of at most 4 and a randomized strategyproof
mechanism with approximation ratio of at most 2, respectively. Then we study
the general setting. We propose a deterministic strategyproof mechanism with
approximation ratio of at most 8 and a randomized strategyproof mechanism with
approximation ratio of at most 4, respectively. Furthermore, we provide lower
bounds of 2 and 14/13 on the approximation ratio for any deterministic and any
randomized strategyproof mechanism, respectively.

### 3. [Constrained Distributed Heterogeneous Two-Facility Location Problems with Max-Variant Cost](http://arxiv.org/pdf/2508.08045v1)

Authors: Xinru Xu, Wenjing Liu, Qizhi Fang

We study a constrained distributed heterogeneous two-facility location
problem, where a set of agents with private locations on the real line are
divided into disjoint groups. The constraint means that the facilities can only
be built in a given multiset of candidate locations and at most one facility
can be built at each candidate location. Given the locations of the two
facilities, the cost of an agent is the distance from her location to the
farthest facility (referred to as max-variant). Our goal is to design
strategyproof distributed mechanisms that can incentivize all agents to
truthfully report their locations and approximately optimize some social
objective. A distributed mechanism consists of two steps: for each group, the
mechanism chooses two candidate locations as the representatives of the group
based only on the locations reported by agents therein; then, it outputs two
facility locations among all the representatives. We focus on a class of
deterministic strategyproof distributed mechanisms and analyze upper and lower
bounds on the distortion under the Average-of-Average cost (average of the
average individual cost of agents in each group), the Max-of-Max cost (maximum
individual cost among all agents), the Average-of-Max cost (average of the
maximum individual cost among all agents in each group) and the Max-of-Average
cost (maximum of the average individual cost of all agents in each group).
Under four social objectives, we obtain constant upper and lower distortion
bounds.

### 4. [Multi-Hop Privacy Propagation for Differentially Private Federated Learning in Social Networks](http://arxiv.org/pdf/2508.07676v1)

Authors: Chenchen Lin, Xuehe Wang

Federated learning (FL) enables collaborative model training across
decentralized clients without sharing local data, thereby enhancing privacy and
facilitating collaboration among clients connected via social networks.
However, these social connections introduce privacy externalities: a client's
privacy loss depends not only on its privacy protection strategy but also on
the privacy decisions of others, propagated through the network via multi-hop
interactions. In this work, we propose a socially-aware privacy-preserving FL
mechanism that systematically quantifies indirect privacy leakage through a
multi-hop propagation model. We formulate the server-client interaction as a
two-stage Stackelberg game, where the server, as the leader, optimizes
incentive policies, and clients, as followers, strategically select their
privacy budgets, which determine their privacy-preserving levels by controlling
the magnitude of added noise. To mitigate information asymmetry in networked
privacy estimation, we introduce a mean-field estimator to approximate the
average external privacy risk. We theoretically prove the existence and
convergence of the fixed point of the mean-field estimator and derive
closed-form expressions for the Stackelberg Nash Equilibrium. Despite being
designed from a client-centric incentive perspective, our mechanism achieves
approximately-optimal social welfare, as revealed by Price of Anarchy (PoA)
analysis. Experiments on diverse datasets demonstrate that our approach
significantly improves client utilities and reduces server costs while
maintaining model performance, outperforming both Social-Agnostic (SA)
baselines and methods that account for social externalities.

### Human-Computer Interaction

### 1. [Phoenix: A Novel Context-Aware Voice-Powered Math Equation Workspace and Editor](http://arxiv.org/pdf/2508.07576v1)

Authors: Kenneth Ge, Ryan Paul, Priscilla Zhang, JooYoung Seo

Writing mathematical notation requires substantial effort, diverting
cognitive resources from conceptual understanding to documentation mechanics,
significantly impacting individuals with fine motor disabilities (FMDs).
Current limits of speech-based math technologies rely on precise dictation of
math symbols and unintuitive command-based interfaces. We present a novel
voice-powered math workspace, applying neuroscience insights to create an
intuitive problem-solving environment. To minimize cognitive load, we leverage
large language models with our novel context engine to support natural language
interaction. Ultimately, we enable fluid mathematical engagement for
individuals with FMDs -- freed from mechanical constraints.

### 2. [Are UX evaluation methods truly accessible](http://arxiv.org/pdf/2508.07620v1)

Authors: Andrés Eduardo Fuentes-Cortázar, Alejandra Rivera-Hernández, José Rafael Rojano-Cáceres

Providing an equitable and inclusive user experience (UX) for people with
disabilities (PWD) is a central goal of accessible design. In the specific case
of Deaf users, whose hearing impairments impact language development and
communication, it is essential to consider their specific needs during software
evaluation processes. This study aimed to analyze a set of UX evaluation
methods suggested in the literature as suitable for Deaf individuals, with the
goal of validating their level of accessibility in real-world contexts. The
research was based on a critical review and practical application of these
methods, identifying their strengths and limitations in relation to the
interaction, perception, and comprehension of Deaf users. Traditional
evaluation instruments, commonly designed for hearing individuals, pose
significant barriers when applied to Deaf users due to their re-liance on
auditory and cognitive abilities, as well as the lack of consideration for
commu-nicational accessibility. The results show that although these methods
are frequently rec-ommended, they exhibit critical shortcomings that hinder the
collection of accurate and representative data. It is concluded that it is
essential to adapt UX evaluation methods to ensure genuinely accessible
processes that address the communicative and cognitive needs of the Deaf
community and accurately reflect their user experience.

### 3. [Through Their Eyes: User Perceptions on Sensitive Attribute Inference of Social Media Videos by Visual Language Models](http://arxiv.org/pdf/2508.07658v1)

Authors: Shuning Zhang, Gengrui Zhang, Yibo Meng, Ziyi Zhang, Hantao Zhao, Xin Yi, Hewu Li

The rapid advancement of Visual Language Models (VLMs) has enabled
sophisticated analysis of visual content, leading to concerns about the
inference of sensitive user attributes and subsequent privacy risks. While
technical capabilities of VLMs are increasingly studied, users' understanding,
perceptions, and reactions to these inferences remain less explored, especially
concerning videos uploaded on the social media. This paper addresses this gap
through a semi-structured interview (N=17), investigating user perspectives on
VLM-driven sensitive attribute inference from their visual data. Findings
reveal that users perceive VLMs as capable of inferring a range of attributes,
including location, demographics, and socioeconomic indicators, often with
unsettling accuracy. Key concerns include unauthorized identification, misuse
of personal information, pervasive surveillance, and harm from inaccurate
inferences. Participants reported employing various mitigation strategies,
though with skepticism about their ultimate effectiveness against advanced AI.
Users also articulate clear expectations for platforms and regulators,
emphasizing the need for enhanced transparency, user control, and proactive
privacy safeguards. These insights are crucial for guiding the development of
responsible AI systems, effective privacy-enhancing technologies, and informed
policymaking that aligns with user expectations and societal values.

### 4. [Understanding Users' Privacy Perceptions Towards LLM's RAG-based Memory](http://arxiv.org/pdf/2508.07664v1)

Authors: Shuning Zhang, Rongjun Ma, Ying Ma, Shixuan Li, Yiqun Xu, Xin Yi, Hewu Li

Large Language Models (LLMs) are increasingly integrating memory
functionalities to provide personalized and context-aware interactions.
However, user understanding, practices and expectations regarding these memory
systems are not yet well understood. This paper presents a thematic analysis of
semi-structured interviews with 18 users to explore their mental models of
LLM's Retrieval Augmented Generation (RAG)-based memory, current usage
practices, perceived benefits and drawbacks, privacy concerns and expectations
for future memory systems. Our findings reveal diverse and often incomplete
mental models of how memory operates. While users appreciate the potential for
enhanced personalization and efficiency, significant concerns exist regarding
privacy, control and the accuracy of remembered information. Users express a
desire for granular control over memory generation, management, usage and
updating, including clear mechanisms for reviewing, editing, deleting and
categorizing memories, as well as transparent insight into how memories and
inferred information are used. We discuss design implications for creating more
user-centric, transparent, and trustworthy LLM memory systems.

### 5. [Towards Aligning Personalized Conversational Recommendation Agents with Users' Privacy Preferences](http://arxiv.org/pdf/2508.07672v1)

Authors: Shuning Zhang, Ying Ma, Jingruo Chen, Simin Li, Xin Yi, Hewu Li

The proliferation of AI agents, with their complex and context-dependent
actions, renders conventional privacy paradigms obsolete. This position paper
argues that the current model of privacy management, rooted in a user's
unilateral control over a passive tool, is inherently mismatched with the
dynamic and interactive nature of AI agents. We contend that ensuring effective
privacy protection necessitates that the agents proactively align with users'
privacy preferences instead of passively waiting for the user to control. To
ground this shift, and using personalized conversational recommendation agents
as a case, we propose a conceptual framework built on Contextual Integrity (CI)
theory and Privacy Calculus theory. This synthesis first reframes automatically
controlling users' privacy as an alignment problem, where AI agents initially
did not know users' preferences, and would learn their privacy preferences
through implicit or explicit feedback. Upon receiving the preference feedback,
the agents used alignment and Pareto optimization for aligning preferences and
balancing privacy and utility. We introduced formulations and instantiations,
potential applications, as well as five challenges.

### 6. [Improving Continuous Grasp Force Decoding from EEG with Time-Frequency Regressors and Premotor-Parietal Network Integration](http://arxiv.org/pdf/2508.07677v1)

Authors: Parth G. Dangi, Yogesh Kumar Meena

Brain-machine interfaces (BMIs) have significantly advanced
neuro-rehabilitation by enhancing motor control. However, accurately decoding
continuous grasp force remains a challenge, limiting the effectiveness of BMI
applications for fine motor tasks. Current models tend to prioritise
algorithmic complexity rather than incorporating neurophysiological insights
into force control, which is essential for developing effective neural
engineering solutions. To address this, we propose EEGForceMap, an EEG-based
methodology that isolates signals from the premotor-parietal region and
extracts task-specific components. We construct three distinct time-frequency
feature sets, which are validated by comparing them with prior studies, and use
them for force prediction with linear, non-linear, and deep learning-based
regressors. The performance of these regressors was evaluated on the
WAY-EEG-GAL dataset that includes 12 subjects. Our results show that
integrating EEGForceMap approach with regressor models yields a 61.7%
improvement in subject-specific conditions (R-squared = 0.815) and a 55.7%
improvement in subject-independent conditions (R-squared = 0.785) over the
state-of-the-art kinematic decoder models. Furthermore, an ablation study
confirms that each preprocessing step significantly enhances decoding accuracy.
This work contributes to the advancement of responsive BMIs for stroke
rehabilitation and assistive robotics by improving EEG-based decoding of
dynamic grasp force.

### 7. [SimViews: An Interactive Multi-Agent System Simulating Visitor-to-Visitor Conversational Patterns to Present Diverse Perspectives of Artifacts in Virtual Museums](http://arxiv.org/pdf/2508.07730v1)

Authors: Mingyang Su, Chao Liu, Jingling Zhang, WU Shuang, Mingming Fan

Offering diverse perspectives on a museum artifact can deepen visitors'
understanding and help avoid the cognitive limitations of a single narrative,
ultimately enhancing their overall experience. Physical museums promote
diversity through visitor interactions. However, it remains a challenge to
present multiple voices appropriately while attracting and sustaining a
visitor's attention in the virtual museum. Inspired by recent studies that show
the effectiveness of LLM-powered multi-agents in presenting different opinions
about an event, we propose SimViews, an interactive multi-agent system that
simulates visitor-to-visitor conversational patterns to promote the
presentation of diverse perspectives. The system employs LLM-powered
multi-agents that simulate virtual visitors with different professional
identities, providing diverse interpretations of artifacts. Additionally, we
constructed 4 conversational patterns between users and agents to simulate
visitor interactions. We conducted a within-subject study with 20 participants,
comparing SimViews to a traditional single-agent condition. Our results show
that SimViews effectively facilitates the presentation of diverse perspectives
through conversations, enhancing participants' understanding of viewpoints and
engagement within the virtual museum.

### 8. [Challenges in Mixed Reality in Assisting Adults with ADHD Symptoms](http://arxiv.org/pdf/2508.07854v1)

Authors: Valerie Tan, Jens Gerken

In this position paper, we discuss symptoms of attention deficit
hyperactivity disorder (ADHD) in adults, as well as available forms of
treatment or assistance in the context of mixed reality. Mixed reality offers
many potentials for assisting adults with symptoms commonly found in (but not
limited to) ADHD, but the availability of mixed reality solutions is not only
limited commercially, but also limited in terms of proof-of-concept prototypes.
We discuss two major challenges with attention assistance using mixed reality
solutions: the limited availability of adult-specific prototypes and studies,
as well as the limited number of solutions that offer continuous intervention
of ADHD-like symptoms that users can employ in their daily life.

### 9. [EchoAid: Enhancing Livestream Shopping Accessibility for the DHH Community](http://arxiv.org/pdf/2508.08020v1)

Authors: Zeyu Yang, Zheng Wei, Yang Zhang, Xian Xu, Changyang He, Muzhi Zhou, Pan Hui

Livestream shopping platforms often overlook the accessibility needs of the
Deaf and Hard of Hearing (DHH) community, leading to barriers such as
information inaccessibility and overload. To tackle these challenges, we
developed \textit{EchoAid}, a mobile app designed to improve the livestream
shopping experience for DHH users. \textit{EchoAid} utilizes advanced
speech-to-text conversion, Rapid Serial Visual Presentation (RSVP) technology,
and Large Language Models (LLMs) to simplify the complex information flow in
live sales environments. We conducted exploratory studies with eight DHH
individuals to identify design needs and iteratively developed the
\textit{EchoAid} prototype based on feedback from three participants. We then
evaluate the performance of this system in a user study workshop involving 38
DHH participants. Our findings demonstrate the successful design and validation
process of \textit{EchoAid}, highlighting its potential to enhance product
information extraction, leading to reduced cognitive overload and more engaging
and customized shopping experiences for DHH users.

### 10. [Fuzzy Ontology Embeddings and Visual Query Building for Ontology Exploration](http://arxiv.org/pdf/2508.08128v1)

Authors: Vladimir Zhurov, John Kausch, Kamran Sedig, Mostafa Milani

Ontologies play a central role in structuring knowledge across domains,
supporting tasks such as reasoning, data integration, and semantic search.
However, their large size and complexity, particularly in fields such as
biomedicine, computational biology, law, and engineering, make them difficult
for non-experts to navigate. Formal query languages such as SPARQL offer
expressive access but require users to understand the ontology's structure and
syntax. In contrast, visual exploration tools and basic keyword-based search
interfaces are easier to use but often lack flexibility and expressiveness. We
introduce FuzzyVis, a proof-of-concept system that enables intuitive and
expressive exploration of complex ontologies. FuzzyVis integrates two key
components: a fuzzy logic-based querying model built on fuzzy ontology
embeddings, and an interactive visual interface for building and interpreting
queries. Users can construct new composite concepts by selecting and combining
existing ontology concepts using logical operators such as conjunction,
disjunction, and negation. These composite concepts are matched against the
ontology using fuzzy membership-based embeddings, which capture degrees of
membership and support approximate, concept-level similarity search. The visual
interface supports browsing, query composition, and partial search without
requiring formal syntax. By combining fuzzy semantics with embedding-based
reasoning, FuzzyVis enables flexible interpretation, efficient computation, and
exploratory learning. Case studies demonstrate how FuzzyVis supports subtle
information needs and helps users uncover relevant concepts in large, complex
ontologies.

### Information Retrieval

### 1. [Orthogonal Low Rank Embedding Stabilization](http://arxiv.org/pdf/2508.07574v1)

Authors: Kevin Zielnicki, Ko-Jen Hsiao

The instability of embedding spaces across model retraining cycles presents
significant challenges to downstream applications using user or item embeddings
derived from recommendation systems as input features. This paper introduces a
novel orthogonal low-rank transformation methodology designed to stabilize the
user/item embedding space, ensuring consistent embedding dimensions across
retraining sessions. Our approach leverages a combination of efficient low-rank
singular value decomposition and orthogonal Procrustes transformation to map
embeddings into a standardized space. This transformation is computationally
efficient, lossless, and lightweight, preserving the dot product and inference
quality while reducing operational burdens. Unlike existing methods that modify
training objectives or embedding structures, our approach maintains the
integrity of the primary model application and can be seamlessly integrated
with other stabilization techniques.

### 2. [Towards Comprehensible Recommendation with Large Language Model Fine-tuning](http://arxiv.org/pdf/2508.07595v1)

Authors: Yunze Luo, Yinjie Jiang, Gaode Chen, Xinghua Zhang, Jun Zhang, Jian Liang, Kaigui Bian

Recommender systems have become increasingly ubiquitous in daily life. While
traditional recommendation approaches primarily rely on ID-based
representations or item-side content features, they often fall short in
capturing the underlying semantics aligned with user preferences (e.g.,
recommendation reasons for items), leading to a semantic-collaborative gap.
Recently emerged LLM-based feature extraction approaches also face a key
challenge: how to ensure that LLMs possess recommendation-aligned reasoning
capabilities and can generate accurate, personalized reasons to mitigate the
semantic-collaborative gap. To address these issues, we propose a novel Content
Understanding from a Collaborative Perspective framework (CURec), which
generates collaborative-aligned content features for more comprehensive
recommendations. \method first aligns the LLM with recommendation objectives
through pretraining, equipping it with instruction-following and
chain-of-thought reasoning capabilities. Next, we design a reward model
inspired by traditional recommendation architectures to evaluate the quality of
the recommendation reasons generated by the LLM. Finally, using the reward
signals, CURec fine-tunes the LLM through RL and corrects the generated reasons
to ensure their accuracy. The corrected reasons are then integrated into a
downstream recommender model to enhance comprehensibility and recommendation
performance. Extensive experiments on public benchmarks demonstrate the
superiority of CURec over existing methods.

### 3. [UMRE: A Unified Monotonic Transformation for Ranking Ensemble in Recommender Systems](http://arxiv.org/pdf/2508.07613v1)

Authors: Zhengrui Xu, Zhe Yang, Zhengxiao Guo, Shukai Liu, Luocheng Lin, Xiaoyan Liu, Yongqi Liu, Han Li

Industrial recommender systems commonly rely on ensemble sorting (ES) to
combine predictions from multiple behavioral objectives. Traditionally, this
process depends on manually designed nonlinear transformations (e.g.,
polynomial or exponential functions) and hand-tuned fusion weights to balance
competing goals -- an approach that is labor-intensive and frequently
suboptimal in achieving Pareto efficiency. In this paper, we propose a novel
Unified Monotonic Ranking Ensemble (UMRE) framework to address the limitations
of traditional methods in ensemble sorting. UMRE replaces handcrafted
transformations with Unconstrained Monotonic Neural Networks (UMNN), which
learn expressive, strictly monotonic functions through the integration of
positive neural integrals. Subsequently, a lightweight ranking model is
employed to fuse the prediction scores, assigning personalized weights to each
prediction objective. To balance competing goals, we further introduce a Pareto
optimality strategy that adaptively coordinates task weights during training.
UMRE eliminates manual tuning, maintains ranking consistency, and achieves
fine-grained personalization. Experimental results on two public recommendation
datasets (Kuairand and Tenrec) and online A/B tests demonstrate impressive
performance and generalization capabilities.

### 4. [Encode Me If You Can: Learning Universal User Representations via Event Sequence Autoencoding](http://arxiv.org/pdf/2508.07748v1)

Authors: Anton Klenitskiy, Artem Fatkulin, Daria Denisova, Anton Pembek, Alexey Vasilev

Building universal user representations that capture the essential aspects of
user behavior is a crucial task for modern machine learning systems. In
real-world applications, a user's historical interactions often serve as the
foundation for solving a wide range of predictive tasks, such as churn
prediction, recommendations, or lifetime value estimation. Using a
task-independent user representation that is effective across all such tasks
can reduce the need for task-specific feature engineering and model retraining,
leading to more scalable and efficient machine learning pipelines. The goal of
the RecSys Challenge 2025 by Synerise was to develop such Universal Behavioral
Profiles from logs of past user behavior, which included various types of
events such as product purchases, page views, and search queries. We propose a
method that transforms the entire user interaction history into a single
chronological sequence and trains a GRU-based autoencoder to reconstruct this
sequence from a fixed-size vector. If the model can accurately reconstruct the
sequence, the latent vector is expected to capture the key behavioral patterns.
In addition to this core model, we explored several alternative methods for
generating user embeddings and combined them by concatenating their output
vectors into a unified representation. This ensemble strategy further improved
generalization across diverse downstream tasks and helped our team,
ai_lab_recsys, achieve second place in the RecSys Challenge 2025.

### 5. [Careful Queries, Credible Results: Teaching RAG Models Advanced Web Search Tools with Reinforcement Learning](http://arxiv.org/pdf/2508.07956v1)

Authors: Yuqin Dai, Shuo Yang, Guoqing Wang, Yong Deng, Zhanwei Zhang, Jun Yin, Pengyu Zeng, Zhenzhe Ying, Changhua Meng, Can Yi, Yuchen Zhou, Weiqiang Wang, Shuai Lu

Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
integrating up-to-date external knowledge, yet real-world web environments
present unique challenges. These limitations manifest as two key challenges:
pervasive misinformation in the web environment, which introduces unreliable or
misleading content that can degrade retrieval accuracy, and the
underutilization of web tools, which, if effectively employed, could enhance
query precision and help mitigate this noise, ultimately improving the
retrieval results in RAG systems. To address these issues, we propose
WebFilter, a novel RAG framework that generates source-restricted queries and
filters out unreliable content. This approach combines a retrieval filtering
mechanism with a behavior- and outcome-driven reward strategy, optimizing both
query formulation and retrieval outcomes. Extensive experiments demonstrate
that WebFilter improves answer quality and retrieval precision, outperforming
existing RAG methods on both in-domain and out-of-domain benchmarks.

### 6. [MLego: Interactive and Scalable Topic Exploration Through Model Reuse](http://arxiv.org/pdf/2508.07654v1)

Authors: Fei Ye, Jiapan Liu, Yinan Jing, Zhenying He, Weirao Wang, X. Sean Wang

With massive texts on social media, users and analysts often rely on topic
modeling techniques to quickly extract key themes and gain insights.
Traditional topic modeling techniques, such as Latent Dirichlet Allocation
(LDA), provide valuable insights but are computationally expensive, making them
impractical for real-time data analysis. Although recent advances in
distributed training and fast sampling methods have improved efficiency,
real-time topic exploration remains a significant challenge. In this paper, we
present MLego, an interactive query framework designed to support real-time
topic modeling analysis by leveraging model materialization and reuse. Instead
of retraining models from scratch, MLego efficiently merges materialized topic
models to construct approximate results at interactive speeds. To further
enhance efficiency, we introduce a hierarchical plan search strategy for single
queries and an optimized query reordering technique for batch queries. We
integrate MLego into a visual analytics prototype system, enabling users to
explore large-scale textual datasets through interactive queries. Extensive
experiments demonstrate that MLego significantly reduces computation costs
while maintaining high-quality topic modeling results. MLego enhances existing
visual analytics approaches, which primarily focus on user-driven topic
modeling, by enabling real-time, query-driven exploration. This complements
traditional methods and bridges the gap between scalable topic modeling and
interactive data analysis.

### 7. [Recommendation Is a Dish Better Served Warm](http://arxiv.org/pdf/2508.07856v1)

Authors: Danil Gusak, Nikita Sukhorukov, Evgeny Frolov

In modern recommender systems, experimental settings typically include
filtering out cold users and items based on a minimum interaction threshold.
However, these thresholds are often chosen arbitrarily and vary widely across
studies, leading to inconsistencies that can significantly affect the
comparability and reliability of evaluation results. In this paper, we
systematically explore the cold-start boundary by examining the criteria used
to determine whether a user or an item should be considered cold. Our
experiments incrementally vary the number of interactions for different items
during training, and gradually update the length of user interaction histories
during inference. We investigate the thresholds across several widely used
datasets, commonly represented in recent papers from top-tier conferences, and
on multiple established recommender baselines. Our findings show that
inconsistent selection of cold-start thresholds can either result in the
unnecessary removal of valuable data or lead to the misclassification of cold
instances as warm, introducing more noise into the system.

### 8. [Improving Document Retrieval Coherence for Semantically Equivalent Queries](http://arxiv.org/pdf/2508.07975v1)

Authors: Stefano Campese, Alessandro Moschitti, Ivano Lauriola

Dense Retrieval (DR) models have proven to be effective for Document
Retrieval and Information Grounding tasks. Usually, these models are trained
and optimized for improving the relevance of top-ranked documents for a given
query. Previous work has shown that popular DR models are sensitive to the
query and document lexicon: small variations of it may lead to a significant
difference in the set of retrieved documents. In this paper, we propose a
variation of the Multi-Negative Ranking loss for training DR that improves the
coherence of models in retrieving the same documents with respect to
semantically similar queries. The loss penalizes discrepancies between the
top-k ranked documents retrieved for diverse but semantic equivalent queries.
We conducted extensive experiments on various datasets, MS-MARCO, Natural
Questions, BEIR, and TREC DL 19/20. The results show that (i) models optimizes
by our loss are subject to lower sensitivity, and, (ii) interestingly, higher
accuracy.

### 9. [Early Explorations of Recommender Systems for Physical Activity and Well-being](http://arxiv.org/pdf/2508.07980v1)

Authors: Alan Said

As recommender systems increasingly guide physical actions, often through
wearables and coaching tools, new challenges arise around how users interpret,
trust, and respond to this advice. This paper introduces a conceptual framework
for tangible recommendations that influence users' bodies, routines, and
well-being. We describe three design dimensions: trust and interpretation,
intent alignment, and consequence awareness. These highlight key limitations in
applying conventional recommender logic to embodied settings. Through examples
and design reflections, we outline how future systems can support long-term
well-being, behavioral alignment, and socially responsible personalization.

### 10. [DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval](http://arxiv.org/pdf/2508.07995v1)

Authors: Meixiu Long, Duolin Sun, Dan Yang, Junjie Wang, Yue Shen, Jian Wang, Peng Wei, Jinjie Gu, Jiahai Wang

Retrieval-augmented generation has achieved strong performance on
knowledge-intensive tasks where query-document relevance can be identified
through direct lexical or semantic matches. However, many real-world queries
involve abstract reasoning, analogical thinking, or multi-step inference, which
existing retrievers often struggle to capture. To address this challenge, we
present \textbf{DIVER}, a retrieval pipeline tailored for reasoning-intensive
information retrieval. DIVER consists of four components: document processing
to improve input quality, LLM-driven query expansion via iterative document
interaction, a reasoning-enhanced retriever fine-tuned on synthetic
multi-domain data with hard negatives, and a pointwise reranker that combines
LLM-assigned helpfulness scores with retrieval scores. On the BRIGHT benchmark,
DIVER achieves state-of-the-art nDCG@10 scores of 41.6 and 28.9 on original
queries, consistently outperforming competitive reasoning-aware models. These
results demonstrate the effectiveness of reasoning-aware retrieval strategies
in complex real-world tasks. Our code and retrieval model will be released
soon.

### Machine Learning

### 1. [Physics-Informed Multimodal Bearing Fault Classification under Variable Operating Conditions using Transfer Learning](http://arxiv.org/pdf/2508.07536v1)

Authors: Tasfiq E. Alam, Md Manjurul Ahsan, Shivakumar Raman

Accurate and interpretable bearing fault classification is critical for
ensuring the reliability of rotating machinery, particularly under variable
operating conditions where domain shifts can significantly degrade model
performance. This study proposes a physics-informed multimodal convolutional
neural network (CNN) with a late fusion architecture, integrating vibration and
motor current signals alongside a dedicated physics-based feature extraction
branch. The model incorporates a novel physics-informed loss function that
penalizes physically implausible predictions based on characteristic bearing
fault frequencies - Ball Pass Frequency Outer (BPFO) and Ball Pass Frequency
Inner (BPFI) - derived from bearing geometry and shaft speed. Comprehensive
experiments on the Paderborn University dataset demonstrate that the proposed
physics-informed approach consistently outperforms a non-physics-informed
baseline, achieving higher accuracy, reduced false classifications, and
improved robustness across multiple data splits. To address performance
degradation under unseen operating conditions, three transfer learning (TL)
strategies - Target-Specific Fine-Tuning (TSFT), Layer-Wise Adaptation Strategy
(LAS), and Hybrid Feature Reuse (HFR) - are evaluated. Results show that LAS
yields the best generalization, with additional performance gains when combined
with physics-informed modeling. Validation on the KAIST bearing dataset
confirms the framework's cross-dataset applicability, achieving up to 98
percent accuracy. Statistical hypothesis testing further verifies significant
improvements (p < 0.01) in classification performance. The proposed framework
demonstrates the potential of integrating domain knowledge with data-driven
learning to achieve robust, interpretable, and generalizable fault diagnosis
for real-world industrial applications.

### 2. [Beyond Single: A Data Selection Principle for LLM Alignment via Fine-Grained Preference Signals](http://arxiv.org/pdf/2508.07638v1)

Authors: Jia Zhang, Yao Liu, Chen-Xi Zhang, Yi Liu, Yi-Xuan Jin, Lan-Zhe Guo, Yu-Feng Li

Aligning Large Language Models (LLMs) with diverse human values requires
moving beyond a single holistic "better-than" preference criterion. While
collecting fine-grained, aspect-specific preference data is more reliable and
scalable, existing methods like Direct Preference Optimization (DPO) struggle
with the severe noise and conflicts inherent in such aggregated datasets. In
this paper, we tackle this challenge from a data-centric perspective. We first
derive the Direct Multi-Preference Optimization (DMPO) objective, and uncover a
key Preference Divergence (PD) term that quantifies inter-aspect preference
conflicts. Instead of using this term for direct optimization, we leverage it
to formulate a novel, theoretically-grounded data selection principle. Our
principle advocates for selecting a subset of high-consensus data-identified by
the most negative PD values-for efficient DPO training. We prove the optimality
of this strategy by analyzing the loss bounds of the DMPO objective in the
selection problem. To operationalize our approach, we introduce practical
methods of PD term estimation and length bias mitigation, thereby proposing our
PD selection method. Evaluation on the UltraFeedback dataset with three varying
conflict levels shows that our simple yet effective strategy achieves over 10%
relative improvement against both the standard holistic preference and a
stronger oracle using aggregated preference signals, all while boosting
training efficiency and obviating the need for intractable holistic preference
annotating, unlocking the potential of robust LLM alignment via fine-grained
preference signals.

### 3. [Multi-Turn Jailbreaks Are Simpler Than They Seem](http://arxiv.org/pdf/2508.07646v1)

Authors: Xiaoxue Yang, Jaeha Lee, Anna-Katharina Dick, Jasper Timm, Fei Xie, Diogo Cruz

While defenses against single-turn jailbreak attacks on Large Language Models
(LLMs) have improved significantly, multi-turn jailbreaks remain a persistent
vulnerability, often achieving success rates exceeding 70% against models
optimized for single-turn protection. This work presents an empirical analysis
of automated multi-turn jailbreak attacks across state-of-the-art models
including GPT-4, Claude, and Gemini variants, using the StrongREJECT benchmark.
Our findings challenge the perceived sophistication of multi-turn attacks: when
accounting for the attacker's ability to learn from how models refuse harmful
requests, multi-turn jailbreaking approaches are approximately equivalent to
simply resampling single-turn attacks multiple times. Moreover, attack success
is correlated among similar models, making it easier to jailbreak newly
released ones. Additionally, for reasoning models, we find surprisingly that
higher reasoning effort often leads to higher attack success rates. Our results
have important implications for AI safety evaluation and the design of
jailbreak-resistant systems. We release the source code at
https://github.com/diogo-cruz/multi_turn_simpler

### 4. [Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation](http://arxiv.org/pdf/2508.07675v1)

Authors: Xutong Liu, Baran Atalar, Xiangxiang Dai, Jinhang Zuo, Siwei Wang, John C. S. Lui, Wei Chen, Carlee Joe-Wong

Large Language Models (LLMs) are revolutionizing how users interact with
information systems, yet their high inference cost poses serious scalability
and sustainability challenges. Caching inference responses, allowing them to be
retrieved without another forward pass through the LLM, has emerged as one
possible solution. Traditional exact-match caching, however, overlooks the
semantic similarity between queries, leading to unnecessary recomputation.
Semantic caching addresses this by retrieving responses based on semantic
similarity, but introduces a fundamentally different cache eviction problem:
one must account for mismatch costs between incoming queries and cached
responses. Moreover, key system parameters, such as query arrival probabilities
and serving costs, are often unknown and must be learned over time. Existing
semantic caching methods are largely ad-hoc, lacking theoretical foundations
and unable to adapt to real-world uncertainty. In this paper, we present a
principled, learning-based framework for semantic cache eviction under unknown
query and cost distributions. We formulate both offline optimization and online
learning variants of the problem, and develop provably efficient algorithms
with state-of-the-art guarantees. We also evaluate our framework on a synthetic
dataset, showing that our proposed algorithms perform matching or superior
performance compared with baselines.

### 5. [Separation and Collaboration: Two-Level Routing Grouped Mixture-of-Experts for Multi-Domain Continual Learning](http://arxiv.org/pdf/2508.07738v1)

Authors: Jialu Zhou, Dianxi Shi, Shaowu Yang, Xinyu Wei, Mingyue Yang, Leqian Li, Mengzhu Wang, Chunping Qiu

Multi-Domain Continual Learning (MDCL) acquires knowledge from sequential
tasks with shifting class sets and distribution. Despite the
Parameter-Efficient Fine-Tuning (PEFT) methods can adapt for this dual
heterogeneity, they still suffer from catastrophic forgetting and forward
forgetting. To address these challenges, we propose a Two-Level Routing Grouped
Mixture-of-Experts (TRGE) method. Firstly, TRGE dynamically expands the
pre-trained CLIP model, assigning specific expert group for each task to
mitigate catastrophic forgetting. With the number of experts continually grows
in this process, TRGE maintains the static experts count within the group and
introduces the intra-group router to alleviate routing overfitting caused by
the increasing routing complexity. Meanwhile, we design an inter-group routing
policy based on task identifiers and task prototype distance, which dynamically
selects relevant expert groups and combines their outputs to enhance inter-task
collaboration. Secondly, to get the correct task identifiers, we leverage
Multimodal Large Language Models (MLLMs) which own powerful multimodal
comprehension capabilities to generate semantic task descriptions and recognize
the correct task identifier. Finally, to mitigate forward forgetting, we
dynamically fuse outputs for unseen samples from the frozen CLIP model and TRGE
adapter based on training progress, leveraging both pre-trained and learned
knowledge. Through extensive experiments across various settings, our method
outperforms other advanced methods with fewer trainable parameters.

### 6. [Topological Feature Compression for Molecular Graph Neural Networks](http://arxiv.org/pdf/2508.07807v1)

Authors: Rahul Khorana

Recent advances in molecular representation learning have produced highly
effective encodings of molecules for numerous cheminformatics and
bioinformatics tasks. However, extracting general chemical insight while
balancing predictive accuracy, interpretability, and computational efficiency
remains a major challenge. In this work, we introduce a novel Graph Neural
Network (GNN) architecture that combines compressed higher-order topological
signals with standard molecular features. Our approach captures global
geometric information while preserving computational tractability and
human-interpretable structure. We evaluate our model across a range of
benchmarks, from small-molecule datasets to complex material datasets, and
demonstrate superior performance using a parameter-efficient architecture. We
achieve the best performing results in both accuracy and robustness across
almost all benchmarks. We open source all code \footnote{All code and results
can be found on Github https://github.com/rahulkhorana/TFC-PACT-Net}.

### 7. [EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning](http://arxiv.org/pdf/2508.07809v1)

Authors: Huanyu Liu, Jia Li, Chang Yu, Taozhi Chen, Yihong Dong, Lecheng Wang, Hu XiaoLong, Ge Li

Reinforcement learning with verifiable reward (RLVR) has become a promising
paradigm for post-training large language models (LLMs) to improve their
reasoning capability. However, when the rollout accuracy is low on hard
problems, the reward becomes sparse, limiting learning efficiency and causing
exploration bottlenecks. Existing approaches either rely on stronger LLMs for
distillation or filter out difficult problems, which limits scalability or
restricts reasoning improvement through exploration.
  We propose EvoCoT, a self-evolving curriculum learning framework based on
two-stage chain-of-thought (CoT) reasoning optimization. EvoCoT constrains the
exploration space by self-generating and verifying CoT trajectories, then
gradually shortens them to expand the space in a controlled way. This enables
LLMs to stably learn from initially unsolved hard problems under sparse
rewards. We apply EvoCoT to multiple LLM families, including Qwen, DeepSeek,
and Llama. Experiments show that EvoCoT enables LLMs to solve previously
unsolved problems, improves reasoning capability without external CoT
supervision, and is compatible with various RL fine-tuning methods. We release
the source code to support future research.

### 8. [Score Augmentation for Diffusion Models](http://arxiv.org/pdf/2508.07926v1)

Authors: Liang Hou, Yuan Gao, Boyuan Jiang, Xin Tao, Qi Yan, Renjie Liao, Pengfei Wan, Di Zhang, Kun Gai

Diffusion models have achieved remarkable success in generative modeling.
However, this study confirms the existence of overfitting in diffusion model
training, particularly in data-limited regimes. To address this challenge, we
propose Score Augmentation (ScoreAug), a novel data augmentation framework
specifically designed for diffusion models. Unlike conventional augmentation
approaches that operate on clean data, ScoreAug applies transformations to
noisy data, aligning with the inherent denoising mechanism of diffusion.
Crucially, ScoreAug further requires the denoiser to predict the augmentation
of the original target. This design establishes an equivariant learning
objective, enabling the denoiser to learn scores across varied denoising
spaces, thereby realizing what we term score augmentation. We also
theoretically analyze the relationship between scores in different spaces under
general transformations. In experiments, we extensively validate ScoreAug on
multiple benchmarks including CIFAR-10, FFHQ, AFHQv2, and ImageNet, with
results demonstrating significant performance improvements over baselines.
Notably, ScoreAug effectively mitigates overfitting across diverse scenarios,
such as varying data scales and model capacities, while exhibiting stable
convergence properties. Another advantage of ScoreAug over standard data
augmentation lies in its ability to circumvent data leakage issues under
certain conditions. Furthermore, we show that ScoreAug can be synergistically
combined with traditional data augmentation techniques to achieve additional
performance gains.

### 9. [Adaptive Fine-Tuning via Pattern Specialization for Deep Time Series Forecasting](http://arxiv.org/pdf/2508.07927v1)

Authors: Amal Saadallah, Abdulaziz Al-Ademi

Time series forecasting poses significant challenges in non-stationary
environments where underlying patterns evolve over time. In this work, we
propose a novel framework that enhances deep neural network (DNN) performance
by leveraging specialized model adaptation and selection. Initially, a base DNN
is trained offline on historical time series data. A reserved validation subset
is then segmented to extract and cluster the most dominant patterns within the
series, thereby identifying distinct regimes. For each identified cluster, the
base DNN is fine-tuned to produce a specialized version that captures unique
pattern characteristics. At inference, the most recent input is matched against
the cluster centroids, and the corresponding fine-tuned version is deployed
based on the closest similarity measure. Additionally, our approach integrates
a concept drift detection mechanism to identify and adapt to emerging patterns
caused by non-stationary behavior. The proposed framework is generalizable
across various DNN architectures and has demonstrated significant performance
gains on both traditional DNNs and recent advanced architectures implemented in
the GluonTS library.

### 10. [Shapley-Inspired Feature Weighting in $k$-means with No Additional Hyperparameters](http://arxiv.org/pdf/2508.07952v1)

Authors: Richard J. Fawley, Renato Cordeiro de Amorim

Clustering algorithms often assume all features contribute equally to the
data structure, an assumption that usually fails in high-dimensional or noisy
settings. Feature weighting methods can address this, but most require
additional parameter tuning. We propose SHARK (Shapley Reweighted $k$-means), a
feature-weighted clustering algorithm motivated by the use of Shapley values
from cooperative game theory to quantify feature relevance, which requires no
additional parameters beyond those in $k$-means. We prove that the $k$-means
objective can be decomposed into a sum of per-feature Shapley values, providing
an axiomatic foundation for unsupervised feature relevance and reducing Shapley
computation from exponential to polynomial time. SHARK iteratively re-weights
features by the inverse of their Shapley contribution, emphasising informative
dimensions and down-weighting irrelevant ones. Experiments on synthetic and
real-world data sets show that SHARK consistently matches or outperforms
existing methods, achieving superior robustness and accuracy, particularly in
scenarios where noise may be present. Software:
https://github.com/rickfawley/shark.

### Neural and Evolutionary Computing

### 1. [Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong](http://arxiv.org/pdf/2508.07522v1)

Authors: Jim O'Connor, Derin Gezgin, Gary B. Parker

We present Evo-Sparrow, a deep learning-based agent for AI decision-making in
Sparrow Mahjong, trained by optimizing Long Short-Term Memory (LSTM) networks
using Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Our model
evaluates board states and optimizes decision policies in a non-deterministic,
partially observable game environment. Empirical analysis conducted over a
significant number of simulations demonstrates that our model outperforms both
random and rule-based agents, and achieves performance comparable to a Proximal
Policy Optimization (PPO) baseline, indicating strong strategic play and robust
policy quality. By combining deep learning with evolutionary optimization, our
approach provides a computationally effective alternative to traditional
reinforcement learning and gradient-based optimization methods. This research
contributes to the broader field of AI game playing, demonstrating the
viability of hybrid learning strategies for complex stochastic games. These
findings also offer potential applications in adaptive decision-making and
strategic AI development beyond Sparrow Mahjong.

### 2. [Energy and Quality of Surrogate-Assisted Search Algorithms: a First Analysis](http://arxiv.org/pdf/2508.07691v1)

Authors: Tomohiro Harada, Enrique Alba, Gabriel Luque

Solving complex real problems often demands advanced algorithms, and then
continuous improvements in the internal operations of a search technique are
needed. Hybrid algorithms, parallel techniques, theoretical advances, and much
more are needed to transform a general search algorithm into an efficient,
useful one in practice. In this paper, we study how surrogates are helping
metaheuristics from an important and understudied point of view: their energy
profile. Even if surrogates are a great idea for substituting a time-demanding
complex fitness function, the energy profile, general efficiency, and accuracy
of the resulting surrogate-assisted metaheuristic still need considerable
research. In this work, we make a first step in analyzing particle swarm
optimization in different versions (including pre-trained and retrained neural
networks as surrogates) for its energy profile (for both processor and memory),
plus a further study on the surrogate accuracy to properly drive the search
towards an acceptable solution. Our conclusions shed new light on this topic
and could be understood as the first step towards a methodology for assessing
surrogate-assisted algorithms not only accounting for time or numerical
efficiency but also for energy and surrogate accuracy for a better, more
holistic characterization of optimization and learning techniques.

### 3. [Growing Reservoirs with Developmental Graph Cellular Automata](http://arxiv.org/pdf/2508.08091v1)

Authors: Matias Barandiaran, James Stovold

Developmental Graph Cellular Automata (DGCA) are a novel model for
morphogenesis, capable of growing directed graphs from single-node seeds. In
this paper, we show that DGCAs can be trained to grow reservoirs. Reservoirs
are grown with two types of targets: task-driven (using the NARMA family of
tasks) and task-independent (using reservoir metrics).
  Results show that DGCAs are able to grow into a variety of specialized,
life-like structures capable of effectively solving benchmark tasks,
statistically outperforming `typical' reservoirs on the same task. Overall,
these lay the foundation for the development of DGCA systems that produce
plastic reservoirs and for modeling functional, adaptive morphogenesis.

### 4. [Symbolic Quantile Regression for the Interpretable Prediction of Conditional Quantiles](http://arxiv.org/pdf/2508.08080v1)

Authors: Cas Oude Hoekstra, Floris den Hengst

Symbolic Regression (SR) is a well-established framework for generating
interpretable or white-box predictive models. Although SR has been successfully
applied to create interpretable estimates of the average of the outcome, it is
currently not well understood how it can be used to estimate the relationship
between variables at other points in the distribution of the target variable.
Such estimates of e.g. the median or an extreme value provide a fuller picture
of how predictive variables affect the outcome and are necessary in
high-stakes, safety-critical application domains. This study introduces
Symbolic Quantile Regression (SQR), an approach to predict conditional
quantiles with SR. In an extensive evaluation, we find that SQR outperforms
transparent models and performs comparably to a strong black-box baseline
without compromising transparency. We also show how SQR can be used to explain
differences in the target distribution by comparing models that predict extreme
and central outcomes in an airline fuel usage case study. We conclude that SQR
is suitable for predicting conditional quantiles and understanding interesting
feature influences at varying quantiles.

### Networking and Internet Architecture

### 1. [Achieving Fair-Effective Communications and Robustness in Underwater Acoustic Sensor Networks: A Semi-Cooperative Approach](http://arxiv.org/pdf/2508.07578v1)

Authors: Yu Gou, Tong Zhang, Jun Liu, Tingting Yang, Shanshan Song, Jun-Hong Cui

This paper investigates the fair-effective communication and robustness in
imperfect and energy-constrained underwater acoustic sensor networks
(IC-UASNs). Specifically, we investigate the impact of unexpected node
malfunctions on the network performance under the time-varying acoustic
channels. Each node is expected to satisfy Quality of Service (QoS)
requirements. However, achieving individual QoS requirements may interfere with
other concurrent communications. Underwater nodes rely excessively on the
rationality of other underwater nodes when guided by fully cooperative
approaches, making it difficult to seek a trade-off between individual QoS and
global fair-effective communications under imperfect conditions. Therefore,
this paper presents a SEmi-COoperative Power Allocation approach (SECOPA) that
achieves fair-effective communication and robustness in IC-UASNs. The approach
is distributed multi-agent reinforcement learning (MARL)-based, and the
objectives are twofold. On the one hand, each intelligent node individually
decides the transmission power to simultaneously optimize individual and global
performance. On the other hand, advanced training algorithms are developed to
provide imperfect environments for training robust models that can adapt to the
time-varying acoustic channels and handle unexpected node failures in the
network. Numerical results are presented to validate our proposed approach.

### 2. [Joint Scheduling and Resource Allocation in mmWave IAB Networks Using Deep RL](http://arxiv.org/pdf/2508.07604v1)

Authors: Maryam Abbasalizadeh, Sashank Narain

Integrated Access and Backhaul (IAB) is critical for dense 5G and beyond
deployments, especially in mmWave bands where fiber backhaul is infeasible. We
propose a novel Deep Reinforcement Learning (DRL) framework for joint link
scheduling and resource slicing in dynamic, interference-prone IAB networks.
Our method integrates a greedy Double Deep Q-Network (DDQN) scheduler to
activate access and backhaul links based on traffic and topology, with a
multi-agent DDQN allocator for bandwidth and antenna assignment across network
slices. This decentralized approach respects strict antenna constraints and
supports concurrent scheduling across heterogeneous links. Evaluations across
96 dynamic topologies show 99.84 percent scheduling accuracy and 20.90 percent
throughput improvement over baselines. The framework's efficient operation and
adaptability make it suitable for dynamic and resource-constrained deployments,
where fast link scheduling and autonomous backhaul coordination are vital.

### 3. [Joint link scheduling and power allocation in imperfect and energy-constrained underwater wireless sensor networks](http://arxiv.org/pdf/2508.07679v1)

Authors: Tong Zhang, Yu Gou, Jun Liu, Shanshan Song, Tingting Yang, Jun-Hong Cui

Underwater wireless sensor networks (UWSNs) stand as promising technologies
facilitating diverse underwater applications. However, the major design issues
of the considered system are the severely limited energy supply and unexpected
node malfunctions. This paper aims to provide fair, efficient, and reliable
(FER) communication to the imperfect and energy-constrained UWSNs (IC-UWSNs).
Therefore, we formulate a FER-communication optimization problem (FERCOP) and
propose ICRL-JSA to solve the formulated problem. ICRL-JSA is a deep
multi-agent reinforcement learning (MARL)-based optimizer for IC-UWSNs through
joint link scheduling and power allocation, which automatically learns
scheduling algorithms without human intervention. However, conventional RL
methods are unable to address the challenges posed by underwater environments
and IC-UWSNs. To construct ICRL-JSA, we integrate deep Q-network into IC-UWSNs
and propose an advanced training mechanism to deal with complex acoustic
channels, limited energy supplies, and unexpected node malfunctions. Simulation
results demonstrate the superiority of the proposed ICRL-JSA scheme with an
advanced training mechanism compared to various benchmark algorithms.

### 4. [An Experimental Reservoir-Augmented Foundation Model: 6G O-RAN Case Study](http://arxiv.org/pdf/2508.07778v1)

Authors: Farhad Rezazadeh, Raymond Zhao, Jiongyu Dai, Amir Ashtari Gargari, Hatim Chergui, Lingjia Liu

Next-generation open radio access networks (O-RAN) continuously stream tens
of key performance indicators (KPIs) together with raw in-phase/quadrature (IQ)
samples, yielding ultra-high-dimensional, non-stationary time series that
overwhelm conventional transformer architectures. We introduce a
reservoir-augmented masked autoencoding transformer (RA-MAT). This time series
foundation model employs echo state network (ESN) computing with masked
autoencoding to satisfy the stringent latency, energy efficiency, and
scalability requirements of 6G O-RAN testing. A fixed, randomly initialized ESN
rapidly projects each temporal patch into a rich dynamical embedding without
backpropagation through time, converting the quadratic self-attention
bottleneck into a lightweight linear operation. These embeddings drive a
patch-wise masked autoencoder that reconstructs 30% randomly masked patches,
compelling the encoder to capture both local dynamics and long-range structure
from unlabeled data. After self-supervised pre-training, RA-MAT is fine-tuned
with a shallow task head while keeping the reservoir and most transformer
layers frozen, enabling low-footprint adaptation to diverse downstream tasks
such as O-RAN KPI forecasting. In a comprehensive O-RAN KPI case study, RA-MAT
achieved sub-0.06 mean squared error (MSE) on several continuous and discrete
KPIs. This work positions RA-MAT as a practical pathway toward real-time,
foundation-level analytics in future 6G networks.

### 5. [Scalable and Energy-Efficient Predictive Data Collection in Wireless Sensor Networks with Constructive Interference](http://arxiv.org/pdf/2508.07882v1)

Authors: Conor Muldoon

A new class of Wireless Sensor Network has emerged whereby multiple nodes
transmit data simultaneously, exploiting constructive interference to enable
data collection frameworks with low energy usage and latency. This paper
presents STAIR (Spatio-Temporal Activation for Intelligent Relaying), a
scalable, resilient framework for Wireless Sensor Networks that leverages
constructive interference and operates effectively under stringent resource
constraints. Using constructive interference requires all nodes to transmit the
same packet at the same time, thus, only one source node can send data per time
slot. STAIR uses coarse-grained topology information to flood a selected subset
of the network, relaying sensor readings from individual nodes during their
allocated time slots. A submodular optimisation algorithm with proven quality
bounds determines near-optimal sensor activation locations and times, aiming to
minimise the sum of mean squared prediction errors from a multiple multivariate
linear regression model, which is used to estimate values at unselected
locations and times. This framework has been extensively validated on a
real-world testbed deployment.

### 6. [Adaptive Multiple Access and Service Placement for Generative Diffusion Models](http://arxiv.org/pdf/2508.07978v1)

Authors: Hamidreza Mazandarani, Mohammad Farhoudi, Masoud Shokrnezhad, Tarik Taleb

Generative Diffusion Models (GDMs) have emerged as key components of
Generative Artificial Intelligence (GenAI), offering unparalleled
expressiveness and controllability for complex data generation tasks. However,
their deployment in real-time and mobile environments remains challenging due
to the iterative and resource-intensive nature of the inference process.
Addressing these challenges, this paper introduces a unified optimization
framework that jointly tackles service placement and multiple access control
for GDMs in mobile edge networks. We propose LEARN-GDM, a Deep Reinforcement
Learning-based algorithm that dynamically partitions denoising blocks across
heterogeneous edge nodes, while accounting for latent transmission costs and
enabling adaptive reduction of inference steps. Our approach integrates a
greedy multiple access scheme with a Double and Dueling Deep Q-Learning
(D3QL)-based service placement, allowing for scalable, adaptable, and
resource-efficient operation under stringent quality of service requirements.
Simulations demonstrate the superior performance of the proposed framework in
terms of scalability and latency resilience compared to conventional monolithic
and fixed chain-length placement strategies. This work advances the state of
the art in edge-enabled GenAI by offering an adaptable solution for GDM
services orchestration, paving the way for future extensions toward semantic
networking and co-inference across distributed environments.

### 7. [Optimization of Private Semantic Communication Performance: An Uncooperative Covert Communication Method](http://arxiv.org/pdf/2508.07586v1)

Authors: Wenjing Zhang, Ye Hu, Tao Luo, Zhilong Zhang, Mingzhe Chen

In this paper, a novel covert semantic communication framework is
investigated. Within this framework, a server extracts and transmits the
semantic information, i.e., the meaning of image data, to a user over several
time slots. An attacker seeks to detect and eavesdrop the semantic transmission
to acquire details of the original image. To avoid data meaning being
eavesdropped by an attacker, a friendly jammer is deployed to transmit jamming
signals to interfere the attacker so as to hide the transmitted semantic
information. Meanwhile, the server will strategically select time slots for
semantic information transmission. Due to limited energy, the jammer will not
communicate with the server and hence the server does not know the transmit
power of the jammer. Therefore, the server must jointly optimize the semantic
information transmitted at each time slot and the corresponding transmit power
to maximize the privacy and the semantic information transmission quality of
the user. To solve this problem, we propose a prioritised sampling assisted
twin delayed deep deterministic policy gradient algorithm to jointly determine
the transmitted semantic information and the transmit power per time slot
without the communications between the server and the jammer. Compared to
standard reinforcement learning methods, the propose method uses an additional
Q network to estimate Q values such that the agent can select the action with a
lower Q value from the two Q networks thus avoiding local optimal action
selection and estimation bias of Q values. Simulation results show that the
proposed algorithm can improve the privacy and the semantic information
transmission quality by up to 77.8% and 14.3% compared to the traditional
reinforcement learning methods.

### 8. [Performance Evaluation of Brokerless Messaging Libraries](http://arxiv.org/pdf/2508.07934v1)

Authors: Lorenzo La Corte, Syed Aftab Rashid, Andrei-Marian Dan

Messaging systems are essential for efficiently transferring large volumes of
data, ensuring rapid response times and high-throughput communication. The
state-of-the-art on messaging systems mainly focuses on the performance
evaluation of brokered messaging systems, which use an intermediate broker to
guarantee reliability and quality of service. However, over the past decade,
brokerless messaging systems have emerged, eliminating the single point of
failure and trading off reliability guarantees for higher performance. Still,
the state-of-the-art on evaluating the performance of brokerless systems is
scarce. In this work, we solely focus on brokerless messaging systems. First,
we perform a qualitative analysis of several possible candidates, to find the
most promising ones. We then design and implement an extensive open-source
benchmarking suite to systematically and fairly evaluate the performance of the
chosen libraries, namely, ZeroMQ, NanoMsg, and NanoMsg-Next-Generation (NNG).
We evaluate these libraries considering different metrics and workload
conditions, and provide useful insights into their limitations. Our analysis
enables practitioners to select the most suitable library for their
requirements.

### 9. [Multimodal Remote Inference](http://arxiv.org/pdf/2508.07555v1)

Authors: Keyuan Zhang, Yin Sun, Bo Ji

We consider a remote inference system with multiple modalities, where a
multimodal machine learning (ML) model performs real-time inference using
features collected from remote sensors. As sensor observations may change
dynamically over time, fresh features are critical for inference tasks.
However, timely delivering features from all modalities is often infeasible due
to limited network resources. To this end, we study a two-modality scheduling
problem to minimize the ML model's inference error, which is expressed as a
penalty function of AoI for both modalities. We develop an index-based
threshold policy and prove its optimality. Specifically, the scheduler switches
modalities when the current modality's index function exceeds a threshold. We
show that the two modalities share the same threshold, and both the index
functions and the threshold can be computed efficiently. The optimality of our
policy holds for (i) general AoI functions that are \emph{non-monotonic} and
\emph{non-additive} and (ii) \emph{heterogeneous} transmission times. Numerical
results show that our policy reduces inference error by up to 55% compared to
round-robin and uniform random policies, which are oblivious to the AoI-based
inference error function. Our results shed light on how to improve remote
inference accuracy by optimizing task-oriented AoI functions.

### 10. [Over-the-Top Resource Broker System for Split Computing: An Approach to Distribute Cloud Computing Infrastructure](http://arxiv.org/pdf/2508.07744v1)

Authors: Ingo Friese, Jochen Klaffer, Mandy Galkow-Schneider, Sergiy Melnyk, Qiuheng Zhou, Hans Dieter Schotten

6G network architectures will usher in a wave of innovative services and
capabilities, introducing concepts like split computing and dynamic processing
nodes. This implicates a paradigm where accessing resources seamlessly aligns
with diverse processing node characteristics, ensuring a uniform interface. In
this landscape, the identity of the operator becomes inconsequential, paving
the way for a collaborative ecosystem where multiple providers contribute to a
shared pool of resources. At the core of this vision is the guarantee of
specific performance parameters, precisely tailored to the location and service
requirements. A consistent layer, as the abstraction of the complexities of
different infrastructure providers, is needed to simplify service deployment.
One promising approach is the introduction of an over-the-top broker for
resource allocation, which streamlines the integration of these services into
the network and cloud infrastructure of the future. This paper explores the
role of the broker in two split computing scenarios. By abstracting the
complexities of various infrastructures, the broker proves to be a versatile
solution applicable not only to cloud environments but also to networks and
beyond. Additionally, a detailed discussion of a proof-of-concept
implementation provides insights into the broker's actual architectural
framework.

### Robotics

### 1. [Feedback Control of a Single-Tail Bioinspired 59-mg Swimmer](http://arxiv.org/pdf/2508.07566v1)

Authors: Conor K. Trygstad, Cody R. Longwell, Francisco M. F. R. Gonçalves, Elijah K. Blankenship, Néstor O. Pérez-Arancibia

We present an evolved steerable version of the single-tail
Fish-&-Ribbon-Inspired Small Swimming Harmonic roBot (FRISSHBot), a 59-mg
biologically inspired swimmer, which is driven by a new shape-memory alloy
(SMA)-based bimorph actuator. The new FRISSHBot is controllable in the
two-dimensional (2D) space, which enabled the first demonstration of
feedback-controlled trajectory tracking of a single-tail aquatic robot with
onboard actuation at the subgram scale. These new capabilities are the result
of a physics-informed design with an enlarged head and shortened tail relative
to those of the original platform. Enhanced by its design, this new platform
achieves forward swimming speeds of up to 13.6 mm/s (0.38 Bl/s), which is over
four times that of the original platform. Furthermore, when following 2D
references in closed loop, the tested FRISSHBot prototype attains forward
swimming speeds of up to 9.1 mm/s, root-mean-square (RMS) tracking errors as
low as 2.6 mm, turning rates of up to 13.1 {\deg}/s, and turning radii as small
as 10 mm.

### 2. [In-situ Value-aligned Human-Robot Interactions with Physical Constraints](http://arxiv.org/pdf/2508.07606v1)

Authors: Hongtao Li, Ziyuan Jiao, Xiaofeng Liu, Hangxin Liu, Zilong Zheng

Equipped with Large Language Models (LLMs), human-centered robots are now
capable of performing a wide range of tasks that were previously deemed
challenging or unattainable. However, merely completing tasks is insufficient
for cognitive robots, who should learn and apply human preferences to future
scenarios. In this work, we propose a framework that combines human preferences
with physical constraints, requiring robots to complete tasks while considering
both. Firstly, we developed a benchmark of everyday household activities, which
are often evaluated based on specific preferences. We then introduced
In-Context Learning from Human Feedback (ICLHF), where human feedback comes
from direct instructions and adjustments made intentionally or unintentionally
in daily life. Extensive sets of experiments, testing the ICLHF to generate
task plans and balance physical constraints with preferences, have demonstrated
the efficiency of our approach.

### 3. [End-to-End Humanoid Robot Safe and Comfortable Locomotion Policy](http://arxiv.org/pdf/2508.07611v1)

Authors: Zifan Wang, Xun Yang, Jianzhuang Zhao, Jiaming Zhou, Teli Ma, Ziyao Gao, Arash Ajoudani, Junwei Liang

The deployment of humanoid robots in unstructured, human-centric environments
requires navigation capabilities that extend beyond simple locomotion to
include robust perception, provable safety, and socially aware behavior.
Current reinforcement learning approaches are often limited by blind
controllers that lack environmental awareness or by vision-based systems that
fail to perceive complex 3D obstacles. In this work, we present an end-to-end
locomotion policy that directly maps raw, spatio-temporal LiDAR point clouds to
motor commands, enabling robust navigation in cluttered dynamic scenes. We
formulate the control problem as a Constrained Markov Decision Process (CMDP)
to formally separate safety from task objectives. Our key contribution is a
novel methodology that translates the principles of Control Barrier Functions
(CBFs) into costs within the CMDP, allowing a model-free Penalized Proximal
Policy Optimization (P3O) to enforce safety constraints during training.
Furthermore, we introduce a set of comfort-oriented rewards, grounded in
human-robot interaction research, to promote motions that are smooth,
predictable, and less intrusive. We demonstrate the efficacy of our framework
through a successful sim-to-real transfer to a physical humanoid robot, which
exhibits agile and safe navigation around both static and dynamic 3D obstacles.

### 4. [GraphCoT-VLA: A 3D Spatial-Aware Reasoning Vision-Language-Action Model for Robotic Manipulation with Ambiguous Instructions](http://arxiv.org/pdf/2508.07650v1)

Authors: Helong Huang, Min Cen, Kai Tan, Xingyue Quan, Guowei Huang, Hong Zhang

Vision-language-action models have emerged as a crucial paradigm in robotic
manipulation. However, existing VLA models exhibit notable limitations in
handling ambiguous language instructions and unknown environmental states.
Furthermore, their perception is largely constrained to static two-dimensional
observations, lacking the capability to model three-dimensional interactions
between the robot and its environment. To address these challenges, this paper
proposes GraphCoT-VLA, an efficient end-to-end model. To enhance the model's
ability to interpret ambiguous instructions and improve task planning, we
design a structured Chain-of-Thought reasoning module that integrates
high-level task understanding and planning, failed task feedback, and low-level
imaginative reasoning about future object positions and robot actions.
Additionally, we construct a real-time updatable 3D Pose-Object graph, which
captures the spatial configuration of robot joints and the topological
relationships between objects in 3D space, enabling the model to better
understand and manipulate their interactions. We further integrates a dropout
hybrid reasoning strategy to achieve efficient control outputs. Experimental
results across multiple real-world robotic tasks demonstrate that GraphCoT-VLA
significantly outperforms existing methods in terms of task success rate and
response speed, exhibiting strong generalization and robustness in open
environments and under uncertain instructions.

### 5. [MoRoCo: Multi-operator-robot Coordination, Interaction and Exploration under Restricted Communication](http://arxiv.org/pdf/2508.07657v1)

Authors: Zhuoli Tian, Yuyang Zhang, Jinsheng Wei, Meng Guo

Fleets of autonomous robots are increasingly deployed alongside multiple
human operators to explore unknown environments, identify salient features, and
perform complex tasks in scenarios such as subterranean exploration,
reconnaissance, and search-and-rescue missions. In these contexts,
communication is often severely limited to short-range exchanges via ad-hoc
networks, posing challenges to coordination. While recent studies have
addressed multi-robot exploration under communication constraints, they largely
overlook the essential role of human operators and their real-time interaction
with robotic teams. Operators may demand timely updates on the exploration
progress and robot status, reprioritize or cancel tasks dynamically, or request
live video feeds and control access. Conversely, robots may seek human
confirmation for anomalous events or require help recovering from motion or
planning failures. To enable such bilateral, context-aware interactions under
restricted communication, this work proposes MoRoCo, a unified framework for
online coordination and exploration in multi-operator, multi-robot systems.
MoRoCo enables the team to adaptively switch among three coordination modes:
spread mode for parallelized exploration with intermittent data sharing,
migrate mode for coordinated relocation, and chain mode for maintaining
high-bandwidth connectivity through multi-hop links. These transitions are
managed through distributed algorithms via only local communication. Extensive
large-scale human-in-the-loop simulations and hardware experiments validate the
necessity of incorporating human robot interactions and demonstrate that MoRoCo
enables efficient, reliable coordination under limited communication, marking a
significant step toward robust human-in-the-loop multi-robot autonomy in
challenging environments.

### 6. [Risk Map As Middleware: Towards Interpretable Cooperative End-to-end Autonomous Driving for Risk-Aware Planning](http://arxiv.org/pdf/2508.07686v1)

Authors: Mingyue Lei, Zewei Zhou, Hongchen Li, Jiaqi Ma, Jia Hu

End-to-end paradigm has emerged as a promising approach to autonomous
driving. However, existing single-agent end-to-end pipelines are often
constrained by occlusion and limited perception range, resulting in hazardous
driving. Furthermore, their black-box nature prevents the interpretability of
the driving behavior, leading to an untrustworthiness system. To address these
limitations, we introduce Risk Map as Middleware (RiskMM) and propose an
interpretable cooperative end-to-end driving framework. The risk map learns
directly from the driving data and provides an interpretable spatiotemporal
representation of the scenario from the upstream perception and the
interactions between the ego vehicle and the surrounding environment for
downstream planning. RiskMM first constructs a multi-agent spatiotemporal
representation with unified Transformer-based architecture, then derives
risk-aware representations by modeling interactions among surrounding
environments with attention. These representations are subsequently fed into a
learning-based Model Predictive Control (MPC) module. The MPC planner
inherently accommodates physical constraints and different vehicle types and
can provide interpretation by aligning learned parameters with explicit MPC
elements. Evaluations conducted on the real-world V2XPnP-Seq dataset confirm
that RiskMM achieves superior and robust performance in risk-aware trajectory
planning, significantly enhancing the interpretability of the cooperative
end-to-end driving framework. The codebase will be released to facilitate
future research in this field.

### 7. [LAURON VI: A Six-Legged Robot for Dynamic Walking](http://arxiv.org/pdf/2508.07689v1)

Authors: Christian Eichmann, Sabine Bellmann, Nicolas Hügel, Louis-Elias Enslin, Carsten Plasberg, Georg Heppner, Arne Roennau, Ruediger Dillmann

Legged locomotion enables robotic systems to traverse extremely challenging
terrains. In many real-world scenarios, the terrain is not that difficult and
these mixed terrain types introduce the need for flexible use of different
walking strategies to achieve mission goals in a fast, reliable, and
energy-efficient way. Six-legged robots have a high degree of flexibility and
inherent stability that aids them in traversing even some of the most difficult
terrains, such as collapsed buildings. However, their lack of fast walking
gaits for easier surfaces is one reason why they are not commonly applied in
these scenarios.
  This work presents LAURON VI, a six-legged robot platform for research on
dynamic walking gaits as well as on autonomy for complex field missions. The
robot's 18 series elastic joint actuators offer high-frequency interfaces for
Cartesian impedance and pure torque control. We have designed, implemented, and
compared three control approaches: kinematic-based, model-predictive, and
reinforcement-learned controllers. The robot hardware and the different control
approaches were extensively tested in a lab environment as well as on a Mars
analog mission. The introduction of fast locomotion strategies for LAURON VI
makes six-legged robots vastly more suitable for a wide range of real-world
applications.

### 8. [Robot and Overhead Crane Collaboration Scheme to Enhance Payload Manipulation](http://arxiv.org/pdf/2508.07758v1)

Authors: Antonio Rosales, Alaa Abderrahim, Markku Suomalainen, Mikael Haag, Tapio Heikkilä

This paper presents a scheme to enhance payload manipulation using a robot
collaborating with an overhead crane. In the current industrial practice, when
the crane's payload has to be accurately manipulated and located in a desired
position, the task becomes laborious and risky since the operators have to
guide the fine motions of the payload by hand. In the proposed collaborative
scheme, the crane lifts the payload while the robot's end-effector guides it
toward the desired position. The only link between the robot and the crane is
the interaction force produced during the guiding of the payload. Two
admittance transfer functions are considered to accomplish harmless and smooth
contact with the payload. The first is used in a position-based admittance
control integrated with the robot. The second one adds compliance to the crane
by processing the interaction force through the admittance transfer function to
generate a crane's velocity command that makes the crane follow the payload.
Then the robot's end-effector and the crane move collaboratively to guide the
payload to the desired location. A method is presented to design the admittance
controllers that accomplish a fluent robot-crane collaboration. Simulations and
experiments validating the scheme potential are shown.

### 9. [AgentWorld: An Interactive Simulation Platform for Scene Construction and Mobile Robotic Manipulation](http://arxiv.org/pdf/2508.07770v1)

Authors: Yizheng Zhang, Zhenjun Yu, Jiaxin Lai, Cewu Lu, Lei Han

We introduce AgentWorld, an interactive simulation platform for developing
household mobile manipulation capabilities. Our platform combines automated
scene construction that encompasses layout generation, semantic asset
placement, visual material configuration, and physics simulation, with a
dual-mode teleoperation system supporting both wheeled bases and humanoid
locomotion policies for data collection. The resulting AgentWorld Dataset
captures diverse tasks ranging from primitive actions (pick-and-place,
push-pull, etc.) to multistage activities (serve drinks, heat up food, etc.)
across living rooms, bedrooms, and kitchens. Through extensive benchmarking of
imitation learning methods including behavior cloning, action chunking
transformers, diffusion policies, and vision-language-action models, we
demonstrate the dataset's effectiveness for sim-to-real transfer. The
integrated system provides a comprehensive solution for scalable robotic skill
acquisition in complex home environments, bridging the gap between
simulation-based training and real-world deployment. The code, datasets will be
available at https://yizhengzhang1.github.io/agent_world/

### 10. [SwarmVLM: VLM-Guided Impedance Control for Autonomous Navigation of Heterogeneous Robots in Dynamic Warehousing](http://arxiv.org/pdf/2508.07814v1)

Authors: Malaika Zafar, Roohan Ahmed Khan, Faryal Batool, Yasheerah Yaqoot, Ziang Guo, Mikhail Litvinov, Aleksey Fedoseev, Dzmitry Tsetserukou

With the growing demand for efficient logistics, unmanned aerial vehicles
(UAVs) are increasingly being paired with automated guided vehicles (AGVs).
While UAVs offer the ability to navigate through dense environments and varying
altitudes, they are limited by battery life, payload capacity, and flight
duration, necessitating coordinated ground support.
  Focusing on heterogeneous navigation, SwarmVLM addresses these limitations by
enabling semantic collaboration between UAVs and ground robots through
impedance control. The system leverages the Vision Language Model (VLM) and the
Retrieval-Augmented Generation (RAG) to adjust impedance control parameters in
response to environmental changes. In this framework, the UAV acts as a leader
using Artificial Potential Field (APF) planning for real-time navigation, while
the ground robot follows via virtual impedance links with adaptive link
topology to avoid collisions with short obstacles.
  The system demonstrated a 92% success rate across 12 real-world trials. Under
optimal lighting conditions, the VLM-RAG framework achieved 8% accuracy in
object detection and selection of impedance parameters. The mobile robot
prioritized short obstacle avoidance, occasionally resulting in a lateral
deviation of up to 50 cm from the UAV path, which showcases safe navigation in
a cluttered setting.

### Software Engineering

### 1. [Adopting Road-Weather Open Data in Route Recommendation Engine](http://arxiv.org/pdf/2508.07881v1)

Authors: Henna Tammia, Benjamin Kämä, Ella Peltonen

Digitraffic, Finland's open road data interface, provides access to
nationwide road sensors with more than 2,300 real-time attributes from 1,814
stations. However, efficiently utilizing such a versatile data API for a
practical application requires a deeper understanding of the data qualities,
preprocessing phases, and machine learning tools. This paper discusses the
challenges of large-scale road weather and traffic data. We go through the
road-weather-related attributes from DigiTraffic as a practical example of
processes required to work with such a dataset. In addition, we provide a
methodology for efficient data utilization for the target application, a
personalized road recommendation engine based on a simple routing application.
We validate our solution based on real-world data, showing we can efficiently
identify and recommend personalized routes for three different driver profiles.

### 2. [SHIELDA: Structured Handling of Exceptions in LLM-Driven Agentic Workflows](http://arxiv.org/pdf/2508.07935v1)

Authors: Jingwen Zhou, Jieshan Chen, Qinghua Lu, Dehai Zhao, Liming Zhu

Large Language Model (LLM) agentic systems are software systems powered by
LLMs that autonomously reason, plan, and execute multi-step workflows to
achieve human goals, rather than merely executing predefined steps. During
execution, these workflows frequently encounter exceptions. Existing exception
handling solutions often treat exceptions superficially, failing to trace
execution-phase exceptions to their reasoning-phase root causes. Furthermore,
their recovery logic is brittle, lacking structured escalation pathways when
initial attempts fail. To tackle these challenges, we first present a
comprehensive taxonomy of 36 exception types across 12 agent artifacts.
Building on this, we propose SHIELDA (Structured Handling of Exceptions in
LLM-Driven Agentic Workflows), a modular runtime exception handling framework
for LLM agentic workflows. SHIELDA uses an exception classifier to select a
predefined exception handling pattern from a handling pattern registry. These
patterns are then executed via a structured handling executor, comprising local
handling, flow control, and state recovery, to enable phase-aware recovery by
linking exceptions to their root causes and facilitating composable strategies.
We validate SHIELDA's effectiveness through a case study on the AutoPR agent,
demonstrating effective, cross-phase recovery from a reasoning-induced
exception.

### 3. [Exploring the Challenges and Opportunities of AI-assisted Codebase Generation](http://arxiv.org/pdf/2508.07966v1)

Authors: Philipp Eibl, Sadra Sabouri, Souti Chattopadhyay

Recent AI code assistants have significantly improved their ability to
process more complex contexts and generate entire codebases based on a textual
description, compared to the popular snippet-level generation. These codebase
AI assistants (CBAs) can also extend or adapt codebases, allowing users to
focus on higher-level design and deployment decisions. While prior work has
extensively studied the impact of snippet-level code generation, this new class
of codebase generation models is relatively unexplored. Despite initial
anecdotal reports of excitement about these agents, they remain less frequently
adopted compared to snippet-level code assistants. To utilize CBAs better, we
need to understand how developers interact with CBAs, and how and why CBAs fall
short of developers' needs. In this paper, we explored these gaps through a
counterbalanced user study and interview with (n = 16) students and developers
working on coding tasks with CBAs. We found that participants varied the
information in their prompts, like problem description (48% of prompts),
required functionality (98% of prompts), code structure (48% of prompts), and
their prompt writing process. Despite various strategies, the overall
satisfaction score with generated codebases remained low (mean = 2.8, median =
3, on a scale of one to five). Participants mentioned functionality as the most
common factor for dissatisfaction (77% of instances), alongside poor code
quality (42% of instances) and communication issues (25% of instances). We
delve deeper into participants' dissatisfaction to identify six underlying
challenges that participants faced when using CBAs, and extracted five barriers
to incorporating CBAs into their workflows. Finally, we surveyed 21 commercial
CBAs to compare their capabilities with participant challenges and present
design opportunities for more efficient and useful CBAs.

### 4. [FairFLRep: Fairness aware fault localization and repair of Deep Neural Networks](http://arxiv.org/pdf/2508.08151v1)

Authors: Moses Openja, Paolo Arcaini, Foutse Khomh, Fuyuki Ishikawa

Deep neural networks (DNNs) are being utilized in various aspects of our
daily lives, including high-stakes decision-making applications that impact
individuals. However, these systems reflect and amplify bias from the data used
during training and testing, potentially resulting in biased behavior and
inaccurate decisions. For instance, having different misclassification rates
between white and black sub-populations. However, effectively and efficiently
identifying and correcting biased behavior in DNNs is a challenge. This paper
introduces FairFLRep, an automated fairness-aware fault localization and repair
technique that identifies and corrects potentially bias-inducing neurons in DNN
classifiers. FairFLRep focuses on adjusting neuron weights associated with
sensitive attributes, such as race or gender, that contribute to unfair
decisions. By analyzing the input-output relationships within the network,
FairFLRep corrects neurons responsible for disparities in predictive quality
parity. We evaluate FairFLRep on four image classification datasets using two
DNN classifiers, and four tabular datasets with a DNN model. The results show
that FairFLRep consistently outperforms existing methods in improving fairness
while preserving accuracy. An ablation study confirms the importance of
considering fairness during both fault localization and repair stages. Our
findings also show that FairFLRep is more efficient than the baseline
approaches in repairing the network.

### 5. [PyVeritas: On Verifying Python via LLM-Based Transpilation and Bounded Model Checking for C](http://arxiv.org/pdf/2508.08171v1)

Authors: Pedro Orvalho, Marta Kwiatkowska

Python has become the dominant language for general-purpose programming, yet
it lacks robust tools for formal verification. In contrast, programmers working
in languages such as C benefit from mature model checkers, for example CBMC,
which enable exhaustive symbolic reasoning and fault localisation. The inherent
complexity of Python, coupled with the verbosity and low-level nature of
existing transpilers (e.g., Cython), have historically limited the
applicability of formal verification to Python programs.
  In this paper, we propose PyVeritas, a novel framework that leverages Large
Language Models (LLMs) for high-level transpilation from Python to C, followed
by bounded model checking and MaxSAT-based fault localisation in the generated
C code. PyVeritas enables verification and bug localisation for Python code
using existing model checking tools for C. Our empirical evaluation on two
Python benchmarks demonstrates that LLM-based transpilation can achieve a high
degree of accuracy, up to 80--90% for some LLMs, enabling effective development
environment that supports assertion-based verification and interpretable fault
diagnosis for small yet non-trivial Python programs.

### 6. [Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation](http://arxiv.org/pdf/2508.07745v1)

Authors: Jiongchi Yu, Xiaofei Xie, Qiang Hu, Yuhan Ma, Ziming Zhao

Insider threats, which can lead to severe losses, remain a major security
concern. While machine learning-based insider threat detection (ITD) methods
have shown promising results, their progress is hindered by the scarcity of
high-quality data. Enterprise data is sensitive and rarely accessible, while
publicly available datasets, when limited in scale due to cost, lack sufficient
real-world coverage; and when purely synthetic, they fail to capture rich
semantics and realistic user behavior. To address this, we propose Chimera, the
first large language model (LLM)-based multi-agent framework that automatically
simulates both benign and malicious insider activities and collects diverse
logs across diverse enterprise environments. Chimera models each employee with
agents that have role-specific behavior and integrates modules for group
meetings, pairwise interactions, and autonomous scheduling, capturing realistic
organizational dynamics. It incorporates 15 types of insider attacks (e.g., IP
theft, system sabotage) and has been deployed to simulate activities in three
sensitive domains: technology company, finance corporation, and medical
institution, producing a new dataset, ChimeraLog. We assess ChimeraLog via
human studies and quantitative analysis, confirming its diversity, realism, and
presence of explainable threat patterns. Evaluations of existing ITD methods
show an average F1-score of 0.83, which is significantly lower than 0.99 on the
CERT dataset, demonstrating ChimeraLog's higher difficulty and utility for
advancing ITD research.

### 7. [ChatGPT on the Road: Leveraging Large Language Model-Powered In-vehicle Conversational Agents for Safer and More Enjoyable Driving Experience](http://arxiv.org/pdf/2508.08101v1)

Authors: Yeana Lee Bond, Mungyeong Choe, Baker Kasim Hasan, Arsh Siddiqui, Myounghoon Jeon

Studies on in-vehicle conversational agents have traditionally relied on
pre-scripted prompts or limited voice commands, constraining natural
driver-agent interaction. To resolve this issue, the present study explored the
potential of a ChatGPT-based in-vehicle agent capable of carrying continuous,
multi-turn dialogues. Forty drivers participated in our experiment using a
motion-based driving simulator, comparing three conditions (No agent,
Pre-scripted agent, and ChatGPT-based agent) as a within-subjects variable.
Results showed that the ChatGPT-based agent condition led to more stable
driving performance across multiple metrics. Participants demonstrated lower
variability in longitudinal acceleration, lateral acceleration, and lane
deviation compared to the other two conditions. In subjective evaluations, the
ChatGPT-based agent also received significantly higher ratings in competence,
animacy, affective trust, and preference compared to the Pre-scripted agent.
Our thematic analysis of driver-agent conversations revealed diverse
interaction patterns in topics, including driving assistance/questions,
entertainment requests, and anthropomorphic interactions. Our results highlight
the potential of LLM-powered in-vehicle conversational agents to enhance
driving safety and user experience through natural, context-rich interactions.

### Social and Information Networks

### 1. [Fabricating Holiness: Characterizing Religious Misinformation Circulators on Arabic Social Media](http://arxiv.org/pdf/2508.07845v1)

Authors: Mahmoud Fawzi, Björn Ross, Walid Magdy

Misinformation is a growing concern in a decade involving critical global
events. While social media regulation is mainly dedicated towards the detection
and prevention of fake news and political misinformation, there is limited
research about religious misinformation which has only been addressed through
qualitative approaches. In this work, we study the spread of fabricated quotes
(Hadith) that are claimed to belong to Prophet Muhammad (the prophet of Islam)
as a case study demonstrating one of the most common religious misinformation
forms on Arabic social media. We attempt through quantitative methods to
understand the characteristics of social media users who interact with
fabricated Hadith. We spotted users who frequently circulate fabricated Hadith
and others who frequently debunk it to understand the main differences between
the two groups. We used Logistic Regression to automatically predict their
behaviors and analyzed its weights to gain insights about the characteristics
and interests of each group. We find that both fabricated Hadith circulators
and debunkers have generally a lot of ties to religious accounts. However,
circulators are identified by many accounts that follow the Shia branch of
Islam, Sunni Islamic public figures from the gulf countries, and many Sunni
non-professional pages posting Islamic content. On the other hand, debunkers
are identified by following academic Islamic scholars from multiple countries
and by having more intellectual non-religious interests like charity, politics,
and activism.

### 2. [From Platform Migration to Cultural Integration: the Ingress and Diffusion of #wlw from TikTok to RedNote in Queer Women](http://arxiv.org/pdf/2508.07579v1)

Authors: Ziqi Pan, Runhua Zhang, Jiehui Luo, Yuanhao Zhang, Yue Deng, Xiaojuan Ma

Hashtags serve as identity markers and connection tools in online queer
communities. Recently, the Western-origin #wlw (women-loving-women) hashtag has
risen in the Chinese lesbian community on RedNote, coinciding with user
migration triggered by the temporary US TikTok ban. This event provides a
unique lens to study cross-cultural hashtag ingress and diffusion through the
populations' responsive behaviors in cyber-migration. In this paper, we
conducted a two-phase content analysis of 418 #wlw posts from January and
April, examining different usage patterns during the hashtag's ingress and
diffusion. Results indicate that the successful introduction of #wlw was
facilitated by TikTok immigrants' bold importation, both populations' mutual
interpretation, and RedNote natives' discussions. In current manifestation of
diffusion, #wlw becomes a RedNote-recognized queer hashtag for sharing queer
life, and semantically expands to support feminism discourse. Our findings
provide empirical insights for enhancing the marginalized communities'
cross-cultural communication.

### Systems and Control

### 1. [Neuro-Symbolic Acceleration of MILP Motion Planning with Temporal Logic and Chance Constraints](http://arxiv.org/pdf/2508.07515v1)

Authors: Junyang Cai, Weimin Huang, Jyotirmoy V. Deshmukh, Lars Lindemann, Bistra Dilkina

Autonomous systems must solve motion planning problems subject to
increasingly complex, time-sensitive, and uncertain missions. These problems
often involve high-level task specifications, such as temporal logic or chance
constraints, which require solving large-scale Mixed-Integer Linear Programs
(MILPs). However, existing MILP-based planning methods suffer from high
computational cost and limited scalability, hindering their real-time
applicability. We propose to use a neuro-symbolic approach to accelerate
MILP-based motion planning by leveraging machine learning techniques to guide
the solver's symbolic search. Focusing on two representative classes of
planning problems, namely, those with Signal Temporal Logic (STL)
specifications and those with chance constraints formulated via Conformal
Predictive Programming (CPP). We demonstrate how graph neural network-based
learning methods can guide traditional symbolic MILP solvers in solving
challenging planning problems, including branching variable selection and
solver parameter configuration. Through extensive experiments, we show that
neuro-symbolic search techniques yield scalability gains. Our approach yields
substantial improvements, achieving an average performance gain of about 20%
over state-of-the-art solver across key metrics, including runtime and solution
quality.

### 2. [Nonlinear Systems in Wireless Power Transfer Applications](http://arxiv.org/pdf/2508.07627v1)

Authors: H Chan

As a novel pattern of energization, the wireless power transfer (WPT) offers
a brand-new way to the energy acquisition for electric-driven devices, thus
alleviating the over-dependence on the battery. This report presents three
types of WPT systems that use nonlinear control methods, in order to acquire an
in-depth understanding of the course of Nonlinear Systems.

### 3. [When are safety filters safe? On minimum phase conditions of control barrier functions](http://arxiv.org/pdf/2508.07684v1)

Authors: Jason J. Choi, Claire J. Tomlin, Shankar Sastry, Koushil Sreenath

In emerging control applications involving multiple and complex tasks, safety
filters are gaining prominence as a modular approach to enforcing safety
constraints. Among various methods, control barrier functions (CBFs) are widely
used for designing safety filters due to their simplicity, imposing a single
linear constraint on the control input at each state. In this work, we focus on
the internal dynamics of systems governed by CBF-constrained control laws. Our
key observation is that, although CBFs guarantee safety by enforcing state
constraints, they can inadvertently be "unsafe" by causing the internal state
to diverge. We investigate the conditions under which the full system state,
including the internal state, can remain bounded under a CBF-based safety
filter. Drawing inspiration from the input-output linearization literature,
where boundedness is ensured by minimum phase conditions, we propose a new set
of CBF minimum phase conditions tailored to the structure imposed by the CBF
constraint. A critical distinction from the original minimum phase conditions
is that the internal dynamics in our setting is driven by a nonnegative virtual
control input, which reflects the enforcement of the safety constraint. We
include a range of numerical examples, including single-input, multi-input,
linear, and nonlinear systems, validating both our analysis and the necessity
of the proposed CBF minimum phase conditions.

### 4. [Deep Reinforcement Learning-Based Control Strategy with Direct Gate Control for Buck Converters](http://arxiv.org/pdf/2508.07693v1)

Authors: Noboru Katayama

This paper proposes a deep reinforcement learning (DRL)-based approach for
directly controlling the gate signals of switching devices to achieve voltage
regulation in a buck converter. Unlike conventional control methods, the
proposed method directly generates gate signals using a neural network trained
through DRL, with the objective of achieving high control speed and flexibility
while maintaining stability. Simulation results demonstrate that the proposed
direct gate control (DGC) method achieves a faster transient response and
stable output voltage regulation, outperforming traditional PWM-based control
schemes. The DGC method also exhibits strong robustness against parameter
variations and sensor noise, indicating its suitability for practical power
electronics applications. The effectiveness of the proposed approach is
validated via simulation.

### 5. [Robust Integrated Priority and Speed Control based on Hierarchical Stochastic Optimization to Promote Bus Schedule Adherence along Signalized Arterial](http://arxiv.org/pdf/2508.07749v1)

Authors: Shurui Guan, Keqiang Li, Haoyu Yang, Yihe Chen, Hanxiao Ren, Yugong Luo

In intelligent transportation systems (ITS), adaptive transit signal priority
(TSP) and dynamic bus control systems have been independently developed to
maintain efficient and reliable urban bus services. However, those two systems
could potentially lead to conflicting decisions due to the lack of
coordination. Although some studies explore the integrated control strategies
along the arterial, they merely rely on signal replanning to address system
uncertainties. Therefore, their performance severely deteriorates in real-world
intersection settings, where abrupt signal timing variation is not always
applicable in consideration of countdown timers and pedestrian signal design.
  In this study, we propose a robust integrated priority and speed control
strategy based on hierarchical stochastic optimization to enhance bus schedule
adherence along the arterial. In the proposed framework, the upper level
ensures the coordination across intersections while the lower level handles
uncertainties for each intersection with stochastic programming. Hence, the
route-level system randomness is decomposed into a series of local problems
that can be solved in parallel using sample average approximation (SAA).
Simulation experiments are conducted under various scenarios with stochastic
bus dwell time and different traffic demand. The results demonstrate that our
approach significantly enhances bus punctuality and time headway equivalence
without abrupt signal timing variation, with negative impacts on car delays
limited to only 0.8%-5.2% as traffic demand increases.

### 6. [Deep Reinforcement Learning with Local Interpretability for Transparent Microgrid Resilience Energy Management](http://arxiv.org/pdf/2508.08132v1)

Authors: Mohammad Hossein Nejati Amiri, Fawaz Annaz, Mario De Oliveira, Florimond Gueniat

Renewable energy integration into microgrids has become a key approach to
addressing global energy issues such as climate change and resource scarcity.
However, the variability of renewable sources and the rising occurrence of High
Impact Low Probability (HILP) events require innovative strategies for reliable
and resilient energy management. This study introduces a practical approach to
managing microgrid resilience through Explainable Deep Reinforcement Learning
(XDRL). It combines the Proximal Policy Optimization (PPO) algorithm for
decision-making with the Local Interpretable Model-agnostic Explanations (LIME)
method to improve the transparency of the actor network's decisions. A case
study in Ongole, India, examines a microgrid with wind, solar, and battery
components to validate the proposed approach. The microgrid is simulated under
extreme weather conditions during the Layla cyclone. LIME is used to analyse
scenarios, showing the impact of key factors such as renewable generation,
state of charge, and load prioritization on decision-making. The results
demonstrate a Resilience Index (RI) of 0.9736 and an estimated battery lifespan
of 15.11 years. LIME analysis reveals the rationale behind the agent's actions
in idle, charging, and discharging modes, with renewable generation identified
as the most influential feature. This study shows the effectiveness of
integrating advanced DRL algorithms with interpretable AI techniques to achieve
reliable and transparent energy management in microgrids.

### 7. [Pinching-Antenna Systems (PASS)-based Indoor Positioning](http://arxiv.org/pdf/2508.08185v1)

Authors: Yaoyu Zhang, Xin Sun, Jun Wang, Tianwei Hou, Anna Li, Yuanwei Liu, Arumugam Nallanathan

Pinching antenna (PA), a flexible waveguide integrated with dielectric
particles, intelligently reconstructs line-of-sight channels. Utilizing its
geometric deterministic model and meter-level reconstruction, PA systems (PASS)
are applied to uplink indoor positioning. In this paper, the uplink positioning
system model for PASS is firstly proposed. A PASS-based received signal
strength indication (RSSI) method is proposed to measure the distance from the
users to each PA, which is efficient and suitable for PASS. PASS-based weighted
least squares (WLS) algorithm is designed to calculate the two-dimensional
coordinates of the users. Several critical observations can be drawn from our
results: i) More PAs on the waveguide improves the positioning accuracy and
robustness. ii) When the number of PAs exceeds a certain threshold, the
performance gain becomes marginal. iii) User locations between and near PAs
yield superior positioning accuracy.

### 8. [IDSO-Managed Bid-Based Transactive Distribution Systems Design for DER Participation in Wholesale Markets While Preserving T-D Interactions](http://arxiv.org/pdf/2508.08187v1)

Authors: Swastik Sharma, Swathi Battula, Sri Niwas Singh

Participation of Distributed Energy Resources (DERs) in bid-based Transactive
Energy Systems (TES) at the distribution systems facilitates strongly coupled,
bidirectional interactions between Transmission-Distribution (T-D) systems.
Capturing these interactions is critical for ensuring seamless integration
within an Integrated Transmission and Distribution (ITD) framework. This study
proposes a methodology to preserve such tight T-D linkages by developing an
Independent Distribution System Operator (IDSO) managed bid-based TES design
for unbalanced distribution systems. The proposed design operates within the
ITD paradigm and permits DER participation in the Wholesale Power Market (WPM)
through IDSO while preserving tight T-D linkages. To this end, this research
offers the following key contributions: a novel bid/offer
prequalification-cum-aggregation method to ensure a grid-safe and value-based
aggregation of DERs' bids and offers for WPM participation through IDSO; and a
retail pricing mechanism that reflects the true value of procuring or offering
additional units of power within the distribution system. Case studies are
conducted on a modified IEEE 123-bus radial feeder populated with a high DER
concentration to validate the proposed frameworks' effectiveness in
coordinating the DERs efficiently and reliably.

### 9. [Toward Goal-Oriented Communication in Multi-Agent Systems: An overview](http://arxiv.org/pdf/2508.07720v1)

Authors: Themistoklis Charalambous, Nikolaos Pappas, Nikolaos Nomikos, Risto Wichman

As multi-agent systems (MAS) become increasingly prevalent in autonomous
systems, distributed control, and edge intelligence, efficient communication
under resource constraints has emerged as a critical challenge. Traditional
communication paradigms often emphasize message fidelity or bandwidth
optimization, overlooking the task relevance of the exchanged information. In
contrast, goal-oriented communication prioritizes the importance of information
with respect to the agents' shared objectives. This review provides a
comprehensive survey of goal-oriented communication in MAS, bridging
perspectives from information theory, communication theory, and machine
learning. We examine foundational concepts alongside learning-based approaches
and emergent protocols. Special attention is given to coordination under
communication constraints, as well as applications in domains such as swarm
robotics, federated learning, and edge computing. The paper concludes with a
discussion of open challenges and future research directions at the
intersection of communication theory, machine learning, and multi-agent
decision making.

### 10. [Learning Satellite Attitude Dynamics with Physics-Informed Normalising Flow](http://arxiv.org/pdf/2508.07841v1)

Authors: Carlo Cena, Mauro Martini, Marcello Chiaberge

Attitude control is a fundamental aspect of spacecraft operations. Model
Predictive Control (MPC) has emerged as a powerful strategy for these tasks,
relying on accurate models of the system dynamics to optimize control actions
over a prediction horizon. In scenarios where physics models are incomplete,
difficult to derive, or computationally expensive, machine learning offers a
flexible alternative by learning the system behavior directly from data.
However, purely data-driven models often struggle with generalization and
stability, especially when applied to inputs outside their training domain. To
address these limitations, we investigate the benefits of incorporating
Physics-Informed Neural Networks (PINNs) into the learning of spacecraft
attitude dynamics, comparing their performance with that of purely data-driven
approaches. Using a Real-valued Non-Volume Preserving (Real NVP) neural network
architecture with a self-attention mechanism, we trained several models on
simulated data generated with the Basilisk simulator. Two training strategies
were considered: a purely data-driven baseline and a physics-informed variant
to improve robustness and stability. Our results demonstrate that the inclusion
of physics-based information significantly enhances the performance in terms of
the mean relative error of the best architectures found by 27.08%. These
advantages are particularly evident when the learned models are integrated into
an MPC framework, where PINN-based models consistently outperform their purely
data-driven counterparts in terms of control accuracy and robustness, yielding
improvements of up to 42.86% in performance stability error and increased
robustness-to-noise.

### Machine Learning (Statistics Category)

### 1. [FairDRL-ST: Disentangled Representation Learning for Fair Spatio-Temporal Mobility Prediction](http://arxiv.org/pdf/2508.07518v1)

Authors: Sichen Zhao, Wei Shao, Jeffrey Chan, Ziqi Xu, Flora Salim

As deep spatio-temporal neural networks are increasingly utilised in urban
computing contexts, the deployment of such methods can have a direct impact on
users of critical urban infrastructure, such as public transport, emergency
services, and traffic management systems. While many spatio-temporal methods
focus on improving accuracy, fairness has recently gained attention due to
growing evidence that biased predictions in spatio-temporal applications can
disproportionately disadvantage certain demographic or geographic groups,
thereby reinforcing existing socioeconomic inequalities and undermining the
ethical deployment of AI in public services. In this paper, we propose a novel
framework, FairDRL-ST, based on disentangled representation learning, to
address fairness concerns in spatio-temporal prediction, with a particular
focus on mobility demand forecasting. By leveraging adversarial learning and
disentangled representation learning, our framework learns to separate
attributes that contain sensitive information. Unlike existing methods that
enforce fairness through supervised learning, which may lead to
overcompensation and degraded performance, our framework achieves fairness in
an unsupervised manner with minimal performance loss. We apply our framework to
real-world urban mobility datasets and demonstrate its ability to close
fairness gaps while delivering competitive predictive performance compared to
state-of-the-art fairness-aware methods.

### 2. [Detecting Mislabeled and Corrupted Data via Pointwise Mutual Information](http://arxiv.org/pdf/2508.07713v1)

Authors: Jinghan Yang, Jiayu Weng

Deep neural networks can memorize corrupted labels, making data quality
critical for model performance, yet real-world datasets are frequently
compromised by both label noise and input noise. This paper proposes a mutual
information-based framework for data selection under hybrid noise scenarios
that quantifies statistical dependencies between inputs and labels. We compute
each sample's pointwise contribution to the overall mutual information and find
that lower contributions indicate noisy or mislabeled instances. Empirical
validation on MNIST with different synthetic noise settings demonstrates that
the method effectively filters low-quality samples. Under label corruption,
training on high-MI samples improves classification accuracy by up to 15\%
compared to random sampling. Furthermore, the method exhibits robustness to
benign input modifications, preserving semantically valid data while filtering
truly corrupted samples.

### 3. [A Tutorial: An Intuitive Explanation of Offline Reinforcement Learning Theory](http://arxiv.org/pdf/2508.07746v1)

Authors: Fengdi Che

Offline reinforcement learning (RL) aims to optimize the return given a fixed
dataset of agent trajectories without additional interactions with the
environment. While algorithm development has progressed rapidly, significant
theoretical advances have also been made in understanding the fundamental
challenges of offline RL. However, bridging these theoretical insights with
practical algorithm design remains an ongoing challenge. In this survey, we
explore key intuitions derived from theoretical work and their implications for
offline RL algorithms.
  We begin by listing the conditions needed for the proofs, including function
representation and data coverage assumptions. Function representation
conditions tell us what to expect for generalization, and data coverage
assumptions describe the quality requirement of the data. We then examine
counterexamples, where offline RL is not solvable without an impractically
large amount of data. These cases highlight what cannot be achieved for all
algorithms and the inherent hardness of offline RL. Building on techniques to
mitigate these challenges, we discuss the conditions that are sufficient for
offline RL. These conditions are not merely assumptions for theoretical proofs,
but they also reveal the limitations of these algorithms and remind us to
search for novel solutions when the conditions cannot be satisfied.

### 4. [Uncertainty-Driven Reliability: Selective Prediction and Trustworthy Deployment in Modern Machine Learning](http://arxiv.org/pdf/2508.07556v1)

Authors: Stephan Rabanser

Machine learning (ML) systems are increasingly deployed in high-stakes
domains where reliability is paramount. This thesis investigates how
uncertainty estimation can enhance the safety and trustworthiness of ML,
focusing on selective prediction -- where models abstain when confidence is
low.
  We first show that a model's training trajectory contains rich uncertainty
signals that can be exploited without altering its architecture or loss. By
ensembling predictions from intermediate checkpoints, we propose a lightweight,
post-hoc abstention method that works across tasks, avoids the cost of deep
ensembles, and achieves state-of-the-art selective prediction performance.
Crucially, this approach is fully compatible with differential privacy (DP),
allowing us to study how privacy noise affects uncertainty quality. We find
that while many methods degrade under DP, our trajectory-based approach remains
robust, and we introduce a framework for isolating the privacy-uncertainty
trade-off. Next, we then develop a finite-sample decomposition of the selective
classification gap -- the deviation from the oracle accuracy-coverage curve --
identifying five interpretable error sources and clarifying which interventions
can close the gap. This explains why calibration alone cannot fix ranking
errors, motivating methods that improve uncertainty ordering. Finally, we show
that uncertainty signals can be adversarially manipulated to hide errors or
deny service while maintaining high accuracy, and we design defenses combining
calibration audits with verifiable inference.
  Together, these contributions advance reliable ML by improving, evaluating,
and safeguarding uncertainty estimation, enabling models that not only make
accurate predictions -- but also know when to say "I do not know".

### 5. [Meta Off-Policy Estimation](http://arxiv.org/pdf/2508.07914v1)

Authors: Olivier Jeunen

Off-policy estimation (OPE) methods enable unbiased offline evaluation of
recommender systems, directly estimating the online reward some target policy
would have obtained, from offline data and with statistical guarantees. The
theoretical elegance of the framework combined with practical successes have
led to a surge of interest, with many competing estimators now available to
practitioners and researchers. Among these, Doubly Robust methods provide a
prominent strategy to combine value- and policy-based estimators.
  In this work, we take an alternative perspective to combine a set of OPE
estimators and their associated confidence intervals into a single, more
accurate estimate. Our approach leverages a correlated fixed-effects
meta-analysis framework, explicitly accounting for dependencies among
estimators that arise due to shared data. This yields a best linear unbiased
estimate (BLUE) of the target policy's value, along with an appropriately
conservative confidence interval that reflects inter-estimator correlation. We
validate our method on both simulated and real-world data, demonstrating
improved statistical efficiency over existing individual estimators.

### 6. [Likelihood Ratio Tests by Kernel Gaussian Embedding](http://arxiv.org/pdf/2508.07982v1)

Authors: Leonardo V. Santoro, Victor M. Panaretos

We propose a novel kernel-based nonparametric two-sample test, employing the
combined use of kernel mean and kernel covariance embedding. Our test builds on
recent results showing how such combined embeddings map distinct probability
measures to mutually singular Gaussian measures on the kernel's RKHS.
Leveraging this result, we construct a test statistic based on the relative
entropy between the Gaussian embeddings, i.e.\ the likelihood ratio. The
likelihood ratio is specifically tailored to detect equality versus singularity
of two Gaussians, and satisfies a ``$0/\infty$" law, in that it vanishes under
the null and diverges under the alternative. To implement the test in finite
samples, we introduce a regularised version, calibrated by way of permutation.
We prove consistency, establish uniform power guarantees under mild conditions,
and discuss how our framework unifies and extends prior approaches based on
spectrally regularized MMD. Empirical results on synthetic and real data
demonstrate remarkable gains in power compared to state-of-the-art methods,
particularly in high-dimensional and weak-signal regimes.

### 7. [Stochastic dynamics learning with state-space systems](http://arxiv.org/pdf/2508.07876v1)

Authors: Juan-Pablo Ortega, Florian Rossmannek

This work advances the theoretical foundations of reservoir computing (RC) by
providing a unified treatment of fading memory and the echo state property
(ESP) in both deterministic and stochastic settings. We investigate state-space
systems, a central model class in time series learning, and establish that
fading memory and solution stability hold generically -- even in the absence of
the ESP -- offering a robust explanation for the empirical success of RC models
without strict contractivity conditions. In the stochastic case, we critically
assess stochastic echo states, proposing a novel distributional perspective
rooted in attractor dynamics on the space of probability distributions, which
leads to a rich and coherent theory. Our results extend and generalize previous
work on non-autonomous dynamical systems, offering new insights into causality,
stability, and memory in RC models. This lays the groundwork for reliable
generative modeling of temporal data in both deterministic and stochastic
regimes.

### 8. [Gaussian Approximation for Two-Timescale Linear Stochastic Approximation](http://arxiv.org/pdf/2508.07928v1)

Authors: Bogdan Butyrin, Artemy Rubtsov, Alexey Naumov, Vladimir Ulyanov, Sergey Samsonov

In this paper, we establish non-asymptotic bounds for accuracy of normal
approximation for linear two-timescale stochastic approximation (TTSA)
algorithms driven by martingale difference or Markov noise. Focusing on both
the last iterate and Polyak-Ruppert averaging regimes, we derive bounds for
normal approximation in terms of the convex distance between probability
distributions. Our analysis reveals a non-trivial interaction between the fast
and slow timescales: the normal approximation rate for the last iterate
improves as the timescale separation increases, while it decreases in the
Polyak-Ruppert averaged setting. We also provide the high-order moment bounds
for the error of linear TTSA algorithm, which may be of independent interest.

### 9. [Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](http://arxiv.org/pdf/2508.08222v1)

Authors: Tong Yang, Yu Huang, Yingbin Liang, Yuejie Chi

Transformers have demonstrated remarkable capabilities in multi-step
reasoning tasks. However, understandings of the underlying mechanisms by which
they acquire these abilities through training remain limited, particularly from
a theoretical standpoint. This work investigates how transformers learn to
solve symbolic multi-step reasoning problems through chain-of-thought
processes, focusing on path-finding in trees. We analyze two intertwined tasks:
a backward reasoning task, where the model outputs a path from a goal node to
the root, and a more complex forward reasoning task, where the model implements
two-stage reasoning by first identifying the goal-to-root path and then
reversing it to produce the root-to-goal path. Our theoretical analysis,
grounded in the dynamics of gradient descent, shows that trained one-layer
transformers can provably solve both tasks with generalization guarantees to
unseen trees. In particular, our multi-phase training dynamics for forward
reasoning elucidate how different attention heads learn to specialize and
coordinate autonomously to solve the two subtasks in a single autoregressive
path. These results provide a mechanistic explanation of how trained
transformers can implement sequential algorithmic procedures. Moreover, they
offer insights into the emergence of reasoning abilities, suggesting that when
tasks are structured to take intermediate chain-of-thought steps, even shallow
multi-head transformers can effectively solve problems that would otherwise
require deeper architectures.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-12 PST.

### 1. [Time series transformer for tourism demand forecasting](https://www.nature.com/articles/s41598-025-15286-0)

Authors: Siyuan Yi et al.

### 2. [Quantum key distribution as a quantum machine learning task](https://www.nature.com/articles/s41534-025-01088-9)

Authors: Thomas Decker et al.

### 3. [Integrated algorithm and hardware design for hybrid neuromorphic systems](https://www.nature.com/articles/s44335-025-00036-2)

Authors: James Seekings et al.

### 4. [Concept learning based on improved FCM- BiLSTM for fuzzy data classification and fusion](https://www.nature.com/articles/s41598-025-14821-3)

Authors: Jiaojiao Niu et al.

### 5. [Securing gait recognition with homomorphic encryption](https://www.nature.com/articles/s41598-025-14047-3)

Authors: Marina Banov et al.

### 6. [Digital twin generation for adsorption in porous materials using Stochastic MorphoDeep](https://www.nature.com/articles/s43246-025-00906-z)

Authors: Adam Hammoumi et al.

### 7. [Causality-aware graph neural networks for functional stratification and phenotype prediction at scale](https://www.nature.com/articles/s41540-025-00567-1)

Authors: Charalampos P. Triantafyllidis et al.

### 8. [Enhancing wearable sensor data analysis for patient health monitoring using allied data disparity technique and multi instance ensemble perceptron learning](https://www.nature.com/articles/s41598-025-08051-w)

Authors: Mohd Anjum et al.

### 9. [Automated violence monitoring system for real-time fistfight detection using deep learning-based temporal action localization](https://www.nature.com/articles/s41598-025-12531-4)

Authors: Baolong Qi et al.

### 10. [Designing an innovative Multi-Criteria Decision Making (MCDM) framework for optimized teaching and delivery of physical education curriculum](https://www.nature.com/articles/s41598-025-14283-7)

Authors: Xu Sun et al.

### 11. [Multi-strategy collaborative optimization of gravitational search algorithm](https://www.nature.com/articles/s41598-025-13215-9)

Authors: Zhonghua Yang et al.

