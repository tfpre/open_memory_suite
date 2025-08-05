Open Memory Suite & Frugal Memory Dispatcher: Comprehensive Plan and Methodology

Introduction and Core Thesis

Large Language Models (LLMs) excel at many tasks, but they struggle with long-term memory. Their built-in context windows are limited, and important information from earlier interactions can be forgotten or lead to expensive re-prompting. Recent research highlights that even state-of-the-art models like GPT-4 perform poorly on episodic memory tasks involving recall of specific past events or facts over long conversations. Simply increasing context length or fine-tuning internal model weights is not sufficient, due to trade-offs in explicit vs. implicit memory storage. The core thesis of this project is that external memory mechanisms, if managed intelligently, can dramatically improve an agent’s long-term recall without exorbitant cost or latency.

Open Memory Suite is our proposed solution comprising two primary contributions:



An Open Memory Benchmark – a standardized evaluation framework to measure how well different memory systems and policies help LLMs recall information, along with the associated cost (latency, monetary cost, etc.). This benchmark provides foundational capital for the community: a way to compare approaches on common ground.

A Frugal Memory Dispatcher – a policy module that decides how to handle incoming conversational content (store verbatim, summarize, or discard) in order to minimize memory cost and latency while preserving high recall. This component serves as demonstration capital, showing a practical way to cut memory usage by ~2× (or more) with negligible loss in relevant information.

The ultimate goal is to ship an open-source suite (tentatively the open_memory_suite repo and a frugal_memory Python package) implementing these ideas, complete with a leaderboard and a technical report. By the end of the project, researchers and developers should be able to install our package, plug it into their LLM applications (e.g. via a LangChain memory interface), and achieve significant cost savings in context management. At the same time, they can use the benchmark to fairly compare memory strategies (including our dispatcher vs. other baselines).

Why does this matter? In current LLM applications, maintaining long dialogues or knowledge of past interactions often involves either sending large histories to the model repeatedly (expensive and slow) or using external vector databases/knowledge graphs with ad-hoc strategies for what to store. Both approaches incur costs – token fees, latency, or the risk of missing important facts. By formulating this as a decision problem (store vs summarize vs drop each piece of memory) and training a cost-sensitive policy, we aim to optimize the trade-off. For example, memory layer services like Zep have shown it’s possible to dramatically reduce latency (by up to 90% in enterprise tasks) while improving recall by using structured memory stores. Our work builds on this idea, adding a focus on frugality: using just enough memory to achieve high recall. We draw inspiration from recent research like R³Mem, which compresses histories into “virtual memory tokens” to balance retention and retrieval, and from heuristic approaches in agent frameworks that decide when to flush or retain context. However, our contribution will be the first open benchmark to evaluate such strategies side by side, and a working memory dispatcher implementation that others can readily adopt.

In summary, the project’s thesis is: By intelligently triaging what an LLM “remembers” (via external memory stores or summaries), we can cut down memory cost and latency significantly (on the order of 50% or more) while maintaining at least ~90% of the original recall ability on relevant facts. This document presents a comprehensive plan for how we will achieve and validate this, targeted at our development team and research collaborators.



System Overview and Architecture

To tackle the problem, we have designed the system with clear modular components and workstreams. The Open Memory Suite consists of the following key modules:



Memory Adapters: Pluggable backends that interface with different memory storage solutions (graph-based memory, vector databases, on-the-fly compressed memory, etc.). These allow the benchmark to evaluate various memory mechanisms under a unified API.

Frugal Memory Dispatcher: A decision-making component (initially rule-based, later ML-enhanced) that routes each new piece of information to one of the memory adapters or to a summarization pipeline – or decides to not store it at all – based on cost/benefit analysis.

Benchmark Harness: A framework to run simulated conversations (from various datasets) through memory systems, log the outcomes, and compute metrics like recall and cost. This includes dataset curation and a standardized trace log format for recording what happens at each turn.

Metrics & Leaderboard: Tools to analyze performance (e.g. recall@K, latency, $$ spent) and a lightweight web UI (Streamlit app) to display and compare results from different strategies. This encourages external contributions and fair comparison.

LangChain Integration & Demo: A small adapter to use the Frugal Memory Dispatcher in existing LLM toolchains (like LangChain), and an interactive demo script that showcases how the system works in a realistic multi-turn conversation scenario.

Below is the high-level repository structure which reflects these components (subject to refinement as we implement):



open_memory_suite/

├── adapters/

│ ├── base.py # Abstract base class for MemoryAdapter

│ ├── zep_graph.py # Adapter for Zep's graph memory service

│ ├── faiss_store.py # Adapter for a FAISS-HNSW vector store

│ └── r3mem_wrapper.py # Adapter for R³Mem (reversible memory via prompts)

├── dispatcher/

│ ├── heuristics.py # Rule-based dispatcher (v0)

│ ├── triage_bert.py # Utilities for fine-tuning the BERT triage model

│ └── frugal.py # Main FrugalDispatcher implementation

├── benchmark/

│ ├── datasets/

│ │ └── sessions.jsonl # Curated conversational sessions for the benchmark

│ ├── metrics.py # Metric computations (recall, cost, etc.)

│ ├── run_harness.py # Orchestrates running a full benchmark evaluation

│ └── cost_table.yaml # Configurable cost/latency for various operations

├── langchain_plugin/

│ └── frugal_memory.py # ≤50 LOC shim to plug into LangChain as a Memory

├── streamlit_app/

│ └── leaderboard.py # Streamlit UI for the leaderboard

├── tests/

│ └── test_adapters.py # Unit tests (example for adapters)

└── pyproject.toml # Project config (uses Poetry for packaging)

Each of these components and their roles are explained in detail below, along with the methodological reasoning behind them.



Memory Adapters and Data Logging

Memory Adapters are classes that implement a common interface (defined in MemoryAdapter ABC) for storing and retrieving conversational data. The idea is to abstract over different memory implementations so that our benchmark can compare them easily and the dispatcher can use them interchangeably. We plan to implement three initial adapters, each exemplifying a distinct approach:



Zep Graph Adapter: An adapter wrapping the Zep memory service (via the zep-python SDK). Zep is a temporal knowledge graph based memory: it stores conversation facts in a graph database (Neo4j under the hood) with nodes/edges representing semantic entities and their relationships over time. This allows complex queries and cross-session memory. We choose Zep to represent a knowledge-graph approach to memory, which can capture structured facts and potentially support powerful recall (e.g., "what was the user’s request last week when discussing X?"). Zep’s design has shown superior accuracy on long-term tasks by maintaining structured context and drastically improving query speed. Our adapter will connect to a Zep server (or a local Graphiti instance) and allow storing messages as nodes and querying relevant context via Zep’s API. We must handle API rate limits (noted as a risk) – our mitigation is to use a local Neo4j instance for testing if needed.

Vector Store Adapter: An adapter using a FAISS HNSW index (a local vector database) to store embeddings of past messages. This represents a classic retrieval-augmented generation (RAG) approach: each turn’s text is embedded (e.g., with a SentenceTransformer) and added to the index, and on recalling we query the top-k similar embeddings. This is the simplest form of long-term memory, similar to what many LLM apps do with Pinecone or other vector DBs. We include it as a baseline – it’s fast and self-contained, but has no explicit knowledge of temporal ordering or semantics beyond vector similarity. It will help benchmark how a pure vector recall compares in cost/recall to more structured memory.

R³Mem Wrapper Adapter: A special adapter that doesn’t store data externally at all, but instead uses the R³Mem approach of compressing memory into the model’s context (a reversible prompt compression). R³Mem (Retain-Retrieve via Reversible Compression) is a research concept where the model itself is fed a compressed summary that can be expanded back when needed. In practice, implementing R³Mem fully may be complex, but we aim to simulate it by, for instance, using a smaller LLM that continually generates a condensed summary of the conversation so far (which can be “decompressed” by another prompt if needed). This adapter serves as a proxy for in-model memory: its cost is mainly additional tokens in prompts (instead of external storage calls). It allows us to test a scenario of summarizing everything (extreme compression) versus explicit storing. The prompt wrapper might simply maintain a running summary with some reversible format (perhaps using special tokens to delineate memory segments). We label it “draft” because it’s experimental; if it underperforms, it’s fine – it’s there to include an implicit memory baseline.

All adapters will produce a unified trace log as they operate. We define a trace schema (likely as JSON Lines, where each line is a JSON object for one turn or operation) recording key data:



turn_id or timestamp,

the input message,

what action was taken (stored, summarized, dropped),

latency of the memory operation (e.g., how long a DB query took or how long summarization took),

cost in cents (e.g., if calling an API or using tokens),

number of tokens processed or stored,

and any retrieval outcome (e.g., which memory items were retrieved to assist the LLM’s response, and whether the answer was correct – for measuring recall).

By using a simple JSONL for traces, we ensure uniform logging across different memory providers, which makes aggregation and analysis easier. This decision to have our own logging instead of relying on each vendor’s logs is crucial: vendor-specific logs might have different formats or missing fields, whereas our standardized trace captures exactly the metrics we care about for every turn. It also future-proofs the benchmark – any new memory adapter just needs to output the same fields, and it will work with the metrics pipeline.

Dataset Curation: To evaluate memory strategies, we need realistic conversation transcripts and queries. We plan to curate a diverse set of sessions (conversations), including:



PersonaChat dialogues: Open-domain chitchat dialogues from the PersonaChat dataset, which are rich in personal facts and casual context. We will use these to test memory of persona facts or earlier references in a conversation.

Synthetic episodic stories (EpiMemBench): We will leverage the Episodic Memories Benchmark (ICLR 2025) which provides synthetic stories designed to test recall of specific events and temporal reasoning. These stories will be converted into conversational form or QA form where the model is later asked something that requires recalling a detail from earlier in the story. Using this ensures we have ground-truth: we know what should be recallable. (The dataset is “free from contamination” and comes with known answers, which helps in measuring recall accuracy).

Knowledge work logs (Slack/Ticket data): If available (scrubbed of personal info), we will include some real-world conversational logs from support tickets or Slack chats. These often involve long discussions with tasks, where remembering a past decision or detail is important. They introduce domain-specific and noise challenges for memory systems.

These datasets will be normalized into a common format (likely the sessions.jsonl mentioned above), where each session is a list of turns with speaker labels, etc. We will also define evaluation queries for each session (e.g., a question at the end asking something from earlier) to test recall. The RA will work on collecting and cleaning these datasets, ensuring no privacy issues and that they’re legally usable (the plan notes that download mirrors and licenses are public, so no legal hold-ups).

By end of the dataset/adapters phase, we expect a working harness that can simulate a conversation of, say, 100 turns through all three adapters and output logs of their performance. This will validate that our scaffolding (S1) and basic adapters (S3) function correctly. The exit criterion M1 is: the harness can run 3 adapters on a 100-turn sample and log metrics like latency, cost, recall@k.



Metrics and Leaderboard

Metrics are at the heart of the Open Memory Benchmark. We need to quantify how well a memory strategy performs both in terms of utility (recall) and efficiency (cost/latency). The key metrics and evaluation methodology include:



Recall@k: This measures how many of the relevant pieces of information the memory system successfully provides to the LLM when needed, out of the top-k retrieved or considered. For example, if a conversation had 10 facts that are later queried, and our memory strategy allowed the LLM to recall 9 of them (perhaps by retrieving them into context or not forgetting them), recall@10 would be 0.9 (90%). We might use recall@k or a similar metric like question answering accuracy on the final queries. This is the benefit side of the trade-off.

Cost and Latency: We measure the cumulative cost of memory operations. Cost has multiple facets:

Monetary cost ($): If using external APIs (OpenAI embeddings, Zep cloud, etc.), how much money is spent per conversation on those calls. We will maintain a cost_table.yaml mapping operations (e.g., embedding 1k tokens, storing a vector, a Zep API call, a local GPU inference for DistilBERT, etc.) to cost (in fractions of cents). For open-source components on our own hardware, the cost can be set to 0 or an estimate of resource cost.

Latency (ms): How much wall-clock time the memory operations add. A good memory system should be fast so as not to slow down the chat noticeably. We’ll log latency for each operation and possibly use aggregated metrics like 95th percentile latency.

We might combine these or track them separately. In particular, we may introduce a combined metric like $-per-recall or a cost-vs-recall curve.

$-Recall Scatter Plot: One way we plan to present the results is a scatter plot where the x-axis is Recall (%) and the y-axis is Cost ($ or relative) for a given configuration (memory strategy). Each point is a run of a memory strategy on the benchmark. This visualization cleanly shows the trade-off: the top-right corner (high recall, low cost) is the ideal. Our aim is for the Frugal Dispatcher to push the curve towards that ideal compared to baselines (like “store everything” vs “summarize everything”). The notion of a “Pareto optimal” frontier might come up – we want our approach to dominate others if possible.

All these metrics will be tracked and visualized via a Weights & Biases (W&B) dashboard as well as the Streamlit leaderboard. We’ll use W&B for internal tracking (since it’s great for comparing experiment runs, e.g., seeing how a new dispatcher rule improves recall-cost trade-off), and then have the Streamlit app for an outward-facing leaderboard. The Streamlit app (leaderboard.py) will likely allow users to toggle certain views (for example, maybe filter by dataset or show a specific metric) and even upload their own results in a CSV/JSON format to add to the chart. The decision to use Streamlit is driven by simplicity: it allows us to create an interactive dashboard in a single Python file without needing a full web stack, which is overkill for our needs.

Exit criterion M2: By the midpoint of development, we plan to have a preliminary result showing that using even a simple rule-based dispatcher yields significant cost savings with only minor recall loss. Specifically, in a controlled test (perhaps on a toy conversation or subset of data), a notebook or dashboard should demonstrate ≥40% cost reduction (relative to a baseline of storing everything) while maintaining ≥90% recall. This would validate the concept before we invest in the learned model.



Frugal Memory Dispatcher (Rules-based and Learned)

The Frugal Memory Dispatcher is the centerpiece of the project. This module’s job is to analyze each incoming conversation turn and decide one of three actions:



Store it in full detail in the long-term memory store (for maximum recall).

Summarize it (store a condensed version, saving space/cost but preserving key info).

Toss it (drop it entirely, deeming it not important for future queries).

The intuition is analogous to human note-taking or an email triage system: not every piece of information is worth retaining verbatim. Some can be summarized (like we write an excerpt instead of saving the whole document), and some are trivial or irrelevant and can be forgotten. By filtering and compressing in this way, we reduce memory usage (and thus cost) significantly.

Initially, we will implement a rule-based heuristic dispatcher (heuristics.py). These rules will be based on simple logic and observations. For example, possible heuristics might include:



Detect important facts: e.g., if a user message contains a number, date, or name (potentially important factual info), keep it or at least its summary. If it’s small talk or a repeated acknowledgment ("thanks, got it"), we might toss it.

Length-based rule: Very long messages might be summarized to capture their essence, whereas very short messages could be stored as-is because they don’t cost much.

Keyword triggers: If certain topics or markers appear (e.g., "let’s summarize" or system instructions), decide accordingly. Or if the assistant just provided a summary, maybe we don’t store both the long form and summary.

Turn role: Maybe always store user questions verbatim (since user’s exact wording might matter), but summarize the assistant’s verbose answers (since we can regenerate them if we have the summary of what was answered).

Recency and frequency: Perhaps maintain a sliding window for verbatim memory and summarize older content beyond a threshold, etc.

These initial rules will be manually designed and fine-tuned on a small set of examples. The goal is to have a baseline policy that is reasonable. The RA will assist by labeling about 1,000 turns with what the ideal action should be (store/summarize/toss) based on their judgment. This labeled dataset will serve two purposes: (a) to evaluate how well our rules perform (we can check precision/recall of the rule-based decisions against this “ground truth”), and (b) to train the learned model.

In parallel, we will prepare a cost model for operations. For instance, storing a full message might cost X tokens in a vector DB or Y in a graph DB plus some embedding cost; summarizing has the cost of an OpenAI API call or local model run to generate the summary; tossing has essentially zero cost. These costs, along with expected future costs (e.g., if you store everything, retrieving might be slower due to larger DB; if you summarize, retrieval might be slightly less accurate, etc.), inform the dispatcher. The rule-based system will use simplified assumptions (like “always cheaper to toss than store unless X, Y, Z”), whereas the learned model can implicitly learn the trade-offs from the training data.

Once the rule-based dispatcher is in place and we have collected enough labeled examples of its successes/failures, we move to the learned dispatcher. We plan to train a lightweight classifier (e.g., DistilBERT, a small transformer) that takes as input the current turn (and perhaps some recent dialogue context or meta-features) and outputs one of {store, summarize, toss}. We call this the triage-BERT model. We’ll use LoRA (Low-Rank Adapters) fine-tuning on DistilBERT, which is parameter-efficient and allows fast training even on a single GPU. The labeled 1000 turns serve as training data. We might split it 80/20 for train/validation to tune for F1-score on this classification.

Our hypothesis is that the ML model can learn subtler patterns than our hard-coded rules. For example, it might learn that when a user asks a question that is likely to recur, we should store it, but if it's a follow-up on something already summarized, maybe not, etc., in a way that a few if-else rules might miss. The performance of this model will be measured by classification metrics (accuracy, F1) against the labeled set, and more importantly by the end-to-end effect on recall/cost in the benchmark. Our target (Milestone M3) is that this triage-BERT policy beats the heuristic policy by at least ~0.05 F1 (5 percentage points) on the held-out labeled data, indicating it’s making better decisions overall.

One major advantage of using a small local model for this decision is zero per-turn cost at runtime (after training). We avoid having to query a large model (like GPT-4) to ask “should I remember this?” each time – that would defeat the purpose by adding cost. Prior work suggests an LLM could guide memory writing, but it’s too expensive to be frugal. Instead, our approach aligns with the notion of cost-efficient agency: use cheap computations to save expensive ones. DistilBERT on a GPU is extremely fast (millisecond-scale per inference) and can handle a large number of decisions per second, so it won’t bottleneck conversation throughput. This design is motivated by the need for real-time memory management in agents, as noted in some agent frameworks and research on memory systems that emphasize low-latency decisions.

Finally, the dispatcher will have a mechanism to actually execute the chosen action. For "store", it will call the selected MemoryAdapter’s store function (which could be writing to Zep or adding to FAISS, etc.). For "summarize", we need a summarization function – likely using an LLM (maybe GPT-3.5 or Llama-2 local model) to compress the turn into a few sentences. We will integrate an option to use OpenAI API for summarization initially (with cost accounted for) and possibly a local model for offline use. Summaries themselves could be stored via an adapter as well (e.g., stored in vector DB or as special nodes in the graph). For "toss", the dispatcher simply does nothing (maybe logs that it skipped storing that turn).

To evaluate the dispatcher’s effect, we will run end-to-end simulations in the harness: feed entire sessions through an LLM that uses the memory (with or without dispatcher) and then ask final questions. This is how we measure recall and cost in practice. By comparing runs with different policies (all-store vs all-summary vs our frugal policy), we can quantify improvements.



LangChain Integration and Demo

While the benchmark and dispatcher can be used stand-alone, integration into existing ecosystems greatly increases impact. We plan a LangChain memory wrapper (LangChainFrugalMemory in frugal_memory.py) with minimal code (≤ 50 lines). LangChain is a popular framework for chaining LLMs with tools and memory; it defines a Memory interface that typically has methods like save_context(chat_history), load_memory_variables(), etc. Our wrapper will implement these by delegating to our Frugal Memory Dispatcher under the hood. This means any LangChain agent can swap its default memory (say, a simple buffer or vector store) with our FrugalMemory and immediately gain the benefits of cost-aware memory management. Keeping this shim lightweight and simple ensures maintainability and that it stays up-to-date with LangChain’s API.

For demonstration and testing, we will create a scripted interactive demo (for example, a “trip planner agent” scenario). In this scenario, an LLM-based agent plans a 3-day trip with a user through many back-and-forth interactions. This type of conversation can generate a lot of details (places to visit, user preferences, schedules). We’ll instrument two versions of the agent: one using a naive memory (store everything) and one using our frugal memory. By the end, we might ask the agent a question that requires recalling something from Day 1 of planning. We can then illustrate how both agents respond – ideally, both recall correctly, but our frugal agent will have done so with less cost. This will be captured in a short video (≤3 min) via Loom, which is useful for the project README or blog post.

The demo serves both as a validation (proving in a real use-case that the system works end-to-end) and as a communication tool (for others to quickly grasp what we built). By recording it, we can show the dynamic behavior: for instance, we might log or visually mark when the agent decides to summarize or drop information, to show the audience how it prunes the conversation live.



Documentation and Launch Preparation

Throughout the project, we will maintain documentation, but in the final phase we’ll polish it: including a comprehensive README, usage guides for the package, and the technical report (approximately 6 pages, structured like an academic paper with Introduction, Related Work, Methodology, Experiments, Results, Conclusion). The related work will cover some of the backlog reading we’ve been tracking – for example, summarizing the approaches of MemGPT, HyenaDNA (if relevant), or surveys of memory in LLMs (there’s a 2025 survey of memory mechanisms we noted). This grounds our work in context for researchers.

We’ll also prepare a blog post aimed at a broader developer audience, tentatively titled “Cut LLM Memory Cost by 2× While Keeping Recall High”. It will focus on the problem statement and our solution’s results, with an inviting tone to try the library or contribute to the benchmark.

When everything is in place (code, benchmark data, dashboard, docs), we target a launch: announcing the project on Hacker News, Twitter (or X), perhaps the LangChain community, etc. Our success criteria for launch (as per plan) is somewhat tongue-in-cheek: aiming for ≥100 GitHub stars in the first 48 hours and getting participants to submit their memory systems to the leaderboard. The real goal is to seed an open community benchmark similar to how GLUE or other leaderboards spur progress – we’d love for others to test their ideas (maybe someone fine-tunes a GPT-4 to be a better dispatcher, or integrates a new memory store) and use our evaluation harness to report results.



Work Plan and Timeline

We have structured the project into parallel workstreams and a calendar timeline with specific milestones (M0–M5). The workstreams (S1–S8) cover everything from core coding to data curation and outreach. The table below outlines the main streams, their scope, and whether they are on the critical path (i.e., essential for timely completion):



S1 – Core repo scaffolding, CI, testing harness: Set up the repository structure, packaging (using Poetry), and continuous integration (GitHub Actions for running tests, linting with black/ruff, etc.). Also, define the abstract interfaces (MemoryAdapter base class, etc.) and get a dummy adapter working. (Lead dev) – Critical path, as it underpins all other coding.

S2 – Dataset curation & trace logging: Gather and preprocess the datasets (PersonaChat, EpiMemBench stories, Slack/Ticket logs). Design the session format and conversion scripts. Ensure trace logging mechanism is ready. (RA) – Critical path because without data, we can’t evaluate anything.

S3 – Memory adapters (Zep, FAISS, R³Mem): Implement the three adapters and make sure they conform to the interface. Test each independently (e.g., can we store and retrieve something). May involve setting up a local FAISS index and a running instance of Zep or a stub. (Lead dev) – Critical path, needed for the benchmark to run.

S4 – Metrics & leaderboard (Streamlit + W&B): Develop the metrics computation functions and set up a W&B project. Create the Streamlit app skeleton that can read results and display the charts. (RA) – Critical path in the sense that results need to be visualized, but could lag slightly behind data and adapter availability.

S5 – Frugal-Dispatcher (rule-based → triage-BERT): Build the heuristic rules and integrate into a FrugalDispatcher class. Then collect labeled data and train the DistilBERT model with LoRA, integrate that into the dispatcher (as an option or as an improved version that overrides the heuristics). (Lead dev) – Critical path, as this is the core novel component.

S6 – LangChain shim + interactive demo: Write the LangChain memory integration and design the demo scenario. Possibly create a notebook or script to simulate the trip planner chat, and test it thoroughly. (Lead dev) – This depends on S3 and S5 being functional.

S7 – Documentation, blog, launch & comms: Write documentation, prepare the tech report, blog post, and coordinate the launch announcement. (RA) – This happens towards the end, after results from S4–S6 are ready, so not on the critical path earlier.

S8 – (Stretch) RL fine-tune dispatcher: (If time permits) Explore reinforcement learning fine-tuning for the dispatcher policy, using a reward that combines cost and recall (e.g., train a policy network via RLHF or simulation to directly optimize the $$ vs recall trade-off). (Either) – This is not required for launch.

Next, we break down tasks week-by-week with milestones:



Days 0–2: Bootstrap (Setup and M0) – Lead: Initialize the open_memory_suite repository with Poetry, basic project structure as shown above. Set up GitHub Actions for CI (so that any push runs tests and linter). Create a trivial MemoryAdapter implementation (maybe just an in-memory list) and a dummy test case. The RA is not heavily involved in this initial bootstrapping. Milestone M0: The repository installs and pytest -q passes with a dummy adapter, confirming our dev environment is ready.

Days 3–7: Datasets & Adapters (S2 & S3) – Lead: Begin implementing the ZepGraphAdapter and VectorStoreAdapter. Possibly stub the R3Mem adapter (might leave complex parts for later). Make sure you can connect to Zep (or use a local version if available). RA: Focus on data collection – download PersonaChat, the synthetic episodic memory stories, and the Slack/ticket data dump. Normalize all into the sessions.jsonl format (or possibly separate files per dataset to start). Also, design how to log each turn: possibly write a small function that takes a MemoryAdapter and a session and runs through it turn by turn, logging to a JSONL. By the end of week 1, we aim to run a test harness on a small sample (like one PersonaChat convo of 100 turns) across the 3 adapters, producing logs of how each adapter handled it (for now, the dispatcher can just default to “store all” since we haven’t done S5 yet). Milestone M1: The harness can run with all three adapters on a 100-turn sample, and logs are produced including latency, cost, and perhaps a dummy recall metric (we might not have queries to measure recall yet; recall@k could be simulated or just set up for future).

Days 8–12: Metrics & Rule-based Dispatcher (S4 & part of S5) – Lead: Develop the initial FrugalDispatcher v0 with heuristic rules (in heuristics.py). This involves writing regex or condition checks and integrating with the adapters. Also, incorporate a cost table that the dispatcher or metrics can use to compute cost. At the same time, implement an initial version of the metrics.py – functions to calculate recall given a trace log and ground-truth queries, and maybe to aggregate costs. Possibly prepare a Jupyter notebook to visualize a cost vs recall calculation for a simple scenario. RA: Start a Weights & Biases project and create a dashboard. Also, label about 100 turns (out of the 1000 planned) manually to test the rule-based decisions (this can guide the fine-tuning of the heuristics). Build a template for the W&B dashboard (charts for cost/recall) and set up the Streamlit app structure (perhaps static for now, just showing a placeholder or reading a local CSV). Milestone M2: We should have evidence (e.g., in a notebook or simple plot) that the rule-based dispatcher can save at least ~40% of cost with ~90% recall on a toy test. For instance, if storing everything costs $1.00 and our strategy costs $0.60 for the same session, and if out of 10 questions it can answer 9 (90%) as well as the full memory, then we hit the target. This milestone is about validating the concept before investing in the ML model.

Days 13–17: Tiny-BERT Upgrade (S5 continued) – Lead: Using the labeled data the RA has prepared (by now, hopefully ~1000 examples of turns with desired action), fine-tune DistilBERT with LoRA to create the triage classifier. This involves writing training code (possibly using the PEFT library) and then integrating the model into the FrugalDispatcher (perhaps as a second mode: rules-based vs ml-based). We’ll need to ensure that the model’s inference is fast – likely we’ll run it on the same GPU as the LLM or a separate one in larger setups. Also, during this week, implement the evaluate_cost() CLI or script that runs the harness with a given policy and outputs metrics (for nightly automation). RA: Set up a nightly GitHub Action that runs run_harness.py on perhaps a subset of data (to keep it quick) and logs the results to W&B and updates any badge (if we have a README badge). Also, start drafting the tech report: especially the Related Work (surveying relevant literature) and Methods sections (which can describe our approach in a scholarly way, much of which this document covers in plainer language). Milestone M3: The trained triage-BERT model is showing improved decision quality over the heuristics (e.g., if heuristics had F1 of 0.80 on the RA’s labeled set, the model achieves 0.85). Additionally, the entire pipeline is running in an automated fashion (CI or nightly), meaning our toolchain is solid. We consider this a significant milestone because it proves the ML approach works and we have a reliable evaluation loop.

Days 18–23: Plug-in & Demo (S6) – Lead: Implement the LangChain memory class LangChainFrugalMemory. This likely involves subclassing ConversationBufferMemory or similar, but instead of storing to a buffer, call our dispatcher’s methods. Test it in a notebook with a simple LangChain agent to ensure compatibility. Next, create the trip-planner demo script. This might be a Python script or notebook where a conversation is simulated (either with a real LLM or possibly a stub for determinism) that goes through planning a trip. Instrument the script to print out or record what the dispatcher does at each turn (e.g., “[Dispatcher] summarized the assistant’s response about hotel options”). Once it’s working, use Loom (or another screen recording) to capture a 2-3 minute walkthrough: showing the conversation and maybe the cost meter vs a baseline. RA: Finalize the Streamlit leaderboard with real data. By now we should have some results from different configurations (all-store, all-summary, rule-based, ML-based). Add a toggle in the app to switch between datasets or to overlay multiple strategies on the $-Recall chart. Ensure it’s easy to read (with legend, axis labels, maybe a table of exact values below the chart). Also continue expanding the tech report, adding initial Results and Ablation sections (e.g., how much did each component help, what if we remove the cost optimization etc.). Milestone M4: At this point, our solution is practically usable by outsiders: one can pip install frugal_memory (we will have published it to PyPI by now), the LangChain integration works, and the README now includes a link to the demo video. Essentially, we have a working prototype ready for external eyes.

Days 24–28: Polish & Launch (S7) – Lead: Enter code freeze (no new features, just bug fixes). Tag a v1.0 release on GitHub. Actively promote the project: post on HN, Twitter, etc., and monitor/respond to any incoming issues or questions from the community (important for building trust and adoption). RA: Publish the blog post on a platform (Medium or our own site). Finalize the tech report (especially the Conclusion, and any last-minute results from late experiments). Possibly prepare a 2-slide summary that could be used in team meetings or to attach to resumes, highlighting the achievements (for personal/team credit). Milestone M5: Launch success is somewhat externally determined (stars, traffic), but the key outcome is that the benchmark is live and ready for submissions, and we have at least seeded it with our own baseline results. Getting 100+ GitHub stars in 48h is a nice vanity metric indicating community interest. By this time, memory benchmark submissions are open for anyone who wants to challenge our results.

The diagram below (in text form) summarizes the timeline in a Gantt-like view, where each stream is a bar and milestones are indicated:



WK1 WK2 WK3 WK4 (calendar weeks)

12345678901234567890123456789012345678

S1 ▓▓▓▓▓▓▓▓▓▓ (Core scaffolding)

S2 ▓▓▓▓▓▓▓▓▓▓ (Dataset curation)

S3 ▓▓▓▓▓▓▓▓▓▓▓▓ (Adapters dev)

S4 ▓▓▓▓▓▓▓▓ (Metrics & leaderboard)

S5 ░░▓▓▓▓▓▓▓▓▓ (Dispatcher rules→ML, ongoing)

S6 ▓▓▓▓▓▓ (LangChain + Demo)

S7 ▓▓▓▓▓▓▓ (Docs, launch prep)

S8 ░░░ (Stretch: RL, if time)

(In the above, solid blocks indicate primary focused effort, and ░ denotes background or polishing work that continues in parallel. For example, S5 starts with light work (░░) during weeks 2–3 when rules are being refined, then heavy work (▓▓▓▓) in weeks 3–4 for the ML model.)

This timeline shows how certain tasks overlap. Notably, dataset prep (S2) overlaps with starting adapter coding (S3), which is fine because the lead can start coding against dummy data while RA prepares real data. Similarly, the rule-based dispatcher starts before all metrics are final, so we use interim evaluation until the full pipeline is ready.

Regular checkpoints (M0–M5) ensure we’re on track:



M0 (Day 2): Base repo & CI ready.

M1 (Day 7): Basic adapters and data integrated; sample run possible.

M2 (Day 12): Heuristic dispatcher validated on toy test.

M3 (Day 17): ML dispatcher working and better than rules; pipeline automated.

M4 (Day 23): Package ready for use, demo recorded.

M5 (Day 28): Launch executed, all deliverables in place.

Key Design Decisions and Rationale

Throughout the design of this project, we made several important decisions. Here we explain each choice, the reasoning behind it, and any alternatives we considered but did not choose:



Using vLLM for Local Model Serving: When our system needs a local LLM (either for summarization or potentially as a part of R³Mem or other baseline), we plan to use the vLLM engine rather than HuggingFace’s Text Generation Inference (TGI) or others. vLLM offers an advanced memory management for the model’s KV cache via PagedAttention, enabling extremely high throughput and efficient GPU memory usage. In fact, vLLM reports up to 3.5× higher throughput than TGI and nearly full GPU memory utilization (only ~4% waste vs 60–80% in naive implementations). This means we can serve models (like a 7B Llama) with lower latency and possibly on smaller hardware. The alternative (TGI) was rejected because of its slower cold-start and additional infrastructure complexity – TGI is great for multi-model serving, but in our case, we likely just need one model running efficiently on one GPU for summarization. The faster response from vLLM also aligns with our low-latency goals for the dispatcher.

Trace Schema as JSONL (with unified stats): We decided that every memory operation (across all adapters) will log to a JSON Lines file with fields for cost_cents, latency_ms, tokens, etc. This uniform logging makes it trivial to aggregate and compare metrics across different memory systems. We considered relying on each system’s built-in logging or metrics (for example, Zep might have its own logs, OpenAI API returns usage tokens, etc.), but those are vendor-specific and hard to aggregate. By centralizing logging in our harness, we ensure we collect the same stats for a Zep lookup as for a FAISS query or a summarization call. This also simplifies writing the metric computations – they just stream through the trace records. An added benefit is easier debugging: one log gives a complete picture of what happened in a session.

Triage-BERT vs. GPT-4 for Dispatch Decisions: An alternative approach for deciding what to remember could be to ask a powerful LLM (like GPT-4) in real-time: “Does this piece of info seem important for later? Should I summarize it or forget it?” While GPT-4 might give a decent answer, this is prohibitively expensive and slow to do for every turn. Our ethos is frugality, so we opted for a small local model (DistilBERT) that, once trained, operates at essentially no cost per turn. This choice sacrifices some decision-making sophistication (GPT-4 might understand context better in some cases) but pays off in being scalable and zero-cost. Moreover, by training on domain-specific patterns (with our labeled data), the triage-BERT could actually outperform a general GPT-4 heuristic in our specific setting. This decision aligns with the goal of having zero added cost per turn for memory management, which is critical if the dispatcher is to actually save money overall, not spend more. It also keeps the system self-contained (no dependency on an external API for the core logic).

Streamlit for Leaderboard UI: We chose Streamlit to build the leaderboard interface because it allowed us to go from idea to functional UI quickly, with minimal front-end work. The alternative of building a custom web app (e.g., a Next.js frontend plus a backend API) was deemed overkill for essentially plotting some charts and tables of results. Streamlit apps can be deployed easily (we might use Streamlit’s cloud or a simple VM) and require only Python knowledge, which our team already has. This means we can iterate on the UI (adding toggles, adjusting plots) without context-switching to JavaScript or dealing with REST endpoints. The downside is less control over the UI/UX compared to a custom web app, but for our technical audience, the functionality (seeing and comparing numbers) is more important than fine-grained design. We also considered just sharing W&B links for the dashboard, but decided on Streamlit because it allows a bit more narration and user interaction in the app, and avoids requiring login.

Local vs. Cloud for Memory Stores: During development, we’ll use local or self-hosted versions of memory stores when possible (e.g., FAISS is local by nature; Zep can be run in a community edition connected to a local Neo4j). This avoids external dependencies and rate limits while testing. In production usage, one might use managed services (like a cloud vector DB or Zep cloud), but our design keeps that flexible. One risk identified was Zep API rate limits (if using the public service) – our mitigation, as noted, is to run a local instance for benchmark purposes. This ensures fairness (no internet latency variance) and reliability in our tests.

Explicit vs. Implicit Memory Balancing: The adapters and dispatcher reflect a fundamental design decision: we are not committing solely to one paradigm of memory. Explicit memory (external databases) is powerful but can grow unbounded and needs policies to prune/merge. Implicit memory (compressing into model weights or hidden state) is elegant but can be unreliable in retrieval. By having both (the R³Mem adapter vs others) in our benchmark, we are acknowledging this trade-off. Our dispatcher primarily operates in the explicit realm (deciding what to store externally), but one could imagine an extension where it also decides when to flush or refresh an implicit memory. The hierarchical compression idea from R³Mem (document-level to entity-level summarization) also influenced how we think about summarization in the dispatcher (e.g., summarizing whole clusters of messages into one summary node in the graph store). In design, whenever possible, we tried to keep options open so researchers can plug in different strategies – e.g., if someone develops a new memory type, they just make an adapter for it and can evaluate it with our benchmark.

In summary, each design choice was guided by either efficiency, simplicity, or flexibility. We leveraged community best practices where available (like vLLM for serving, Streamlit for quick UI) and avoided reinventing wheels that don’t further our research goals. We also explicitly planned fallbacks: for instance, if the DistilBERT model underperforms (a risk we identified), we will still have the rule-based dispatcher as a reasonable alternative; if the Streamlit app on a free tier is too slow or sleeps, we can deploy it on a small VM to keep it live.



Testing and Validation Strategy

Quality assurance is important since we’re building both a library and an evaluation suite. Our testing approach covers unit tests, integration tests, and some performance testing:



Unit Tests (Component-level): For each core module, we write targeted tests. For example, for Adapters, we will use mock stores or small in-memory instances to test that adapter.store() and adapter.retrieve() work as expected. In tests/test_adapters.py, we might simulate a simple memory (say, store a few items, then query them) without needing a real Neo4j or FAISS (we can monkeypatch or use a temporary FAISS index file). Similarly, for the dispatcher rules, we’ll craft specific chat snippets that should trigger each rule and assert the decision is correct (e.g., test that a turn containing "FYI" and no question is classified as toss by a particular regex). These tests will be run in CI for every pull request, along with static type checking (using mypy) and lint checks.

Integration Tests (End-to-end in harness): We will have a mode to run the entire run_harness.py on a small scale (e.g., one short conversation with each adapter) as an integration test. This ensures that all pieces work together: data flows from input through the dispatcher to the adapter and back to output, without exceptions or mis-ordered events. By running this on a 100-turn sample, as mentioned, we simulate a realistic use and verify the system produces the expected logs and metrics. Another integration test is to replace the dispatcher rules with the BERT model in the harness and confirm it still runs end-to-end (i.e., the model loading is fine, etc.). We’ll likely trigger these integration tests in a nightly build (since they might be too slow for every PR).

Performance/Load Tests: Since one of our selling points is low latency and scalability, we should test the system under heavier load. For example, run a 1,000-turn stress test where we feed a long conversation or many conversations sequentially and measure if any slowdown occurs or if memory usage grows unexpectedly. We expect the adapters like FAISS to handle this easily (FAISS can do many queries per second), and the graph (Neo4j) might need checking for memory leaks or slow queries over time. We also will measure throughput of the triage model: e.g., how many decisions per second can DistilBERT handle on a single GPU? This tells us if our approach scales to, say, real-time chat with dozens of messages per minute. If needed, we could simulate concurrent conversations to see if any race conditions in logging or memory updating occur.

Our CI pipeline, as mentioned, will run all unit tests and lint checks on every push, ensuring basic quality. The nightly scheduled run will execute the integration harness (maybe with the latest main branch code) and push results to W&B, so we can track if recall or cost metrics drift (regression testing for performance). We’ll also include mypy type checking to catch errors early, given Python’s dynamic nature.

By the time of release, we intend to have a 100% pass rate on all tests, and ideally some test coverage measurement to identify any untested parts. Additionally, the tech report will include a Results section that effectively is an extended test of our claims: it will show that our method meets the stated goals (cost reduction %, recall retention, etc.). Those experiments double as validation of correctness and effectiveness.



Risks and Mitigations

No ambitious project is without risks. We have compiled a risk register to track potential issues and how we plan to mitigate them:

RiskLikelihoodImpactMitigation StrategyZep API rate limits or dependencyMediumBlocks adapter runs or slows dev/testingUse a local Neo4j with Zep’s open-source Graphiti for testing. Have a caching layer for Zep queries in case we hit limits. In worst case, mock the Zep adapter for integration tests to not rely on external service.DistilBERT fine-tune under-fitsMediumDispatcher saves little cost (if ML policy not better than rules)Keep the rule-based fallback in the dispatcher (can always revert to rules if confidence in model is low). Improve training by increasing the labeled dataset (have RA label more than 1000 turns if needed). If DistilBERT is too weak, consider a slightly larger model (or use GPT-3.5 in a few-shot manner as a temporary policy). The cost goal can still be met by rules in worst case.Streamlit free tier “sleepy” appLowLeaderboard downtime or slow loading for usersHost the Streamlit app on an alternative (e.g., a lightweight Fly.io VM or HuggingFace Spaces) to have more control. Also make the raw results data available, so even if the app is down, users can fetch results. In a pinch, switch to a static website that loads a pre-generated chart (less interactive but reliable).Integration complexity (many parts)MediumUnexpected bugs in end-to-end usage or hard to debug issuesMitigation: Build incrementally and test at each milestone. For example, ensure that each adapter works standalone before adding dispatcher logic; ensure dispatcher + one adapter works before adding all. Use extensive logging (maybe a debug mode that prints decisions verbosely). Also write docs for each module to reduce developer error.User adoption riskLowBenchmark not used by others (impact reduced)Mitigation: Focus on outreach at launch, make it easy to use (pip install, good docs). Possibly reach out to a few friendly researchers to try it and provide feedback/test submissions early. Emphasize the unique value of having a common benchmark – something many people have called for in discussions about LLM long-term memory.Most of the above risks are manageable. The technical risks around model performance and integration will be mitigated by our testing plan and by keeping fallbacks. The project management risks (like timeline slip or lack of adoption) are mitigated by our careful planning (the Gantt chart) and outreach efforts.

One additional stretch idea listed (RL fine-tuning the dispatcher) is not on critical path, as noted. That’s a risk in the sense of scope creep – we will only pursue it if time permits after delivering the core objectives. Another potential risk is that the memory benchmark task itself is very hard to evaluate (what if recall is tricky to define across different systems?). We plan to mitigate that by focusing on very concrete recall questions (like known QA pairs in the dataset) so we have a clear success metric. If needed, we can simplify – e.g., measure how well the final answers match a ground truth when memory is needed, rather than something complicated.



Stretch Goals and Future Work

If we successfully reach milestone M5 early or have extra resources, there are a few stretch goals we are interested in exploring:



Reinforcement Learning for Dispatcher: So far, our dispatcher policy is trained with supervised labels (imitation of what we think is best). A more optimal approach might be to directly optimize a reward that combines cost and recall, by simulating many conversations. We could use Reinforcement Learning with Human Feedback (RLHF) or purely a simulated reward (e.g., +10 points for answering a question correctly, -1 point for each cent spent) to fine-tune the policy. This could potentially find strategies we didn’t think of (maybe it learns to sometimes summarize and sometimes store based on subtle context). We call this the RL-HF dispatcher idea. It’s high risk (may not work better than supervised due to sparse reward), which is why it’s a stretch goal.

Memory Graph Visualizer: When using structured memories like Zep’s knowledge graph, it would be great to visualize the memory after a conversation. A stretch feature is a small tool to generate a force-directed graph showing entities and connections that were formed during the conversation. This could be either in the Streamlit app or a separate script. It’s not core to our evaluation, but it helps humans qualitatively see what the memory looks like (and is a cool factor for presentations).

Additional Memory Baselines: To enrich the benchmark, we can add implementations or integrations for other memory systems beyond our initial three. Candidates include:

MemGPT (if an implementation is available) – a system that also tackled agent memory and had its own Deep Memory Retrieval benchmark.

Hyena-based long memory or Perplexica Memory – some new architectures that claim very long context via alternative attention mechanisms. Or perhaps GPT-4 with a retrieval plugin as a baseline.

Vanilla GPT-4 with extended context – just to compare if throwing a 100k-token context window at the problem (with no external memory) does any better or worse in terms of cost vs recall.

These are more for the academic completeness – even if we don’t implement them, we’d design the benchmark such that others could contribute these.

Multi-agent or Cross-session Memory: Another direction is testing how memory works not just in one conversation thread, but across sessions or across agents. For instance, if one agent learns something and stores it, can another agent retrieve it later (shared memory)? This is beyond our main scope, but our benchmark design could be extended to test multi-session memory (the Slack data might already have threads that simulate this). Not a priority, but a future expansion idea.

By keeping these in mind, we ensure the project is extensible and leaves room for follow-up work, which is important for a research-oriented project. However, we will only pursue these once the main deliverables are secured.



Conclusion

In this plan, we outlined a comprehensive approach to building an Open Memory Benchmark suite and a Frugal Memory Dispatcher to improve long-term memory efficiency in LLM-based agents. We began by identifying the need for such a system: LLMs are powerful but forgetful and costly when it comes to extended interactions. Our solution combines a rigorous evaluation framework with a practical memory-management policy that is both data-driven and cost-aware.

The methodology centers on clear, modular components – from adapters for different memory stores (graph databases, vector stores, compressed contexts) to a two-stage dispatcher (rules then learned model) that decides how to allocate memory budget. We stressed explainability and intuition at each step: the rules are interpretable, and the learned model’s decisions can be analyzed with the same metrics. By logging everything and defining standardized metrics (like recall and cost), we ensure that improvements are measurable and grounded in evidence.

Each design decision, from using vLLM for serving local models (for its efficiency) to choosing a small BERT for decision-making (for zero marginal cost per turn), was made to maximize performance while minimizing complexity and expense. Our timeline demonstrates a realistic path to execution, with incremental milestones that de-risk the project by validating assumptions early (e.g., checking the heuristic policy’s effectiveness before committing to training a model).

Ultimately, if we execute this plan, we expect to deliver:



A benchmark dataset and suite that others in the community can use to test their own memory strategies, filling a gap similar to how GLUE did for NLP tasks.

A working Frugal Memory system that developers can plug into chatbots or agents to halve their memory costs (or better) while maintaining strong performance on recalling necessary information.

Insights and a reference implementation documented in our report and code, helping advance understanding of how to manage LLM memories. For example, the outcome might show that a combination of knowledge graph for structured info and vector store for semantic info, filtered by a smart policy, gives the best of both worlds – something that can influence future designs of AI systems.

The success of this project will be measured not just in technical metrics, but in adoption and community engagement. By aiming for an open-source release and a leaderboard, we signal that we want collaborators and even “competitors” to test against our approach. This will drive the field forward in a transparent way.

In closing, this plan provides a clear roadmap for our team to follow. It balances methodical engineering (with testing, CI, and careful breakdown of tasks) and innovative research (trying a new angle on the memory problem). With everyone’s expertise – the lead’s focus on core implementation and the RA’s support in data and evaluation – we are well-equipped to execute it. The result, we hope, will be a tangible step towards LLMs that are not just large and smart, but also remember effectively and efficiently.



Q1: What is the precise difference between "Memory" and "Context"?



Context refers to the information fed directly into the LLM's prompt at inference time (i.e., what fits in the context window). This is volatile and ephemeral.

Memory refers to the external, persistent storage systems (Zep, FAISS, etc.) that hold information over the long term.

The Goal of the System: The memory system's job is to retrieve relevant information from the long-term Memory and place it into the immediate Context so the LLM can use it to generate a response. Our dispatcher optimizes what gets written to Memory in the first place.

Q2: How exactly is "Recall" being measured?



Recall is not a simple keyword search. It is measured via functional correctness on a question-answering task. At the end of a conversational session, the benchmark will pose a question whose answer was stated earlier. The complete system (LLM + memory) "succeeds" if the final generated answer correctly answers the question.

Evaluation Method: We will use an LLM-as-a-judge approach for scalable evaluation. A powerful model (e.g., GPT-4o) will be prompted to compare the agent's final answer against a ground-truth answer and score its correctness (e.g., a simple binary pass/fail or a score from 1-5). This is more robust than string matching and accounts for semantic equivalence.

Q3: What is the specific difference between the R³Mem adapter's function and the dispatcher's "summarize" action?



This is a crucial distinction between two different philosophies of memory compression.

Dispatcher Summarization is a per-turn triage action. The dispatcher analyzes a single conversational turn and decides to create a new, condensed memory artifact from it, which is then stored externally (e.g., the summary text is embedded and saved to FAISS). It's an explicit, discrete write operation.



The R³Mem Adapter simulates a holistic context compression. It doesn't act on a single turn in isolation. Instead, it continuously maintains a compressed representation of the entire recent conversation history within the prompt itself.



Analogy: The dispatcher is like a secretary who listens to a meeting, writes a summary of one agenda item on an index card, and files it. The R³Mem approach is like having a shorthand expert continuously rewriting the notes for the entire meeting to be as dense as possible on a single page.

Q4: What is the system's default behavior or fail-safe?



The system is designed with a clear fallback hierarchy. If the triage-BERT model fails to load or produce a valid decision, the FrugalDispatcher will automatically revert to using the rule-based heuristics.py. If the heuristic rules also fail (which is highly unlikely), the system will default to the safest, highest-recall action: store the turn verbatim. This ensures system availability and prevents data loss at the cost of temporary frugality.

Q5: How is the cost_table.yaml populated? Is it a live billing tool?



The cost table contains static estimates, not live data. It is not a billing system. Costs for API calls (e.g., OpenAI) are based on their publicly listed pricing as of July 2025. Costs for local operations (e.g., running the triage-BERT) are estimated based on GPU time and amortized hardware cost, but are often set to near-zero to emphasize the savings over external API calls. The purpose of the table is to provide a consistent, relative cost model for the dispatcher's policy to optimize against, not to track exact expenditure.



Clarifications & Methodological Commitments

This section codifies our approach to the most ambiguous and high-risk aspects of the project. It serves as a pact between collaborators to ensure we navigate these challenges with a clear, shared strategy.

1. On the Primacy of the Heuristic Dispatcher & the ML "Off-Ramp"

This addresses the primary timeline risk: the Week 2-3 development bottleneck and the possibility of the ML model failing.



The Commitment: The FrugalDispatcher v0 (the rule-based heuristic model in heuristics.py) is the primary, shippable deliverable. The fine-tuned triage-BERT is a performance enhancement and a stretch goal. We will build the entire system around the heuristic model first.

The "Off-Ramp" Procedure: We will formally assess the triage-BERT on Day 17. If it does not demonstrate a statistically significant and meaningful improvement over the heuristic model (e.g., +0.05 F1-score on the labeled dataset), we will freeze its development. All subsequent work for the v1.0 launch—including the LangChain shim, demo, and final benchmarks—will proceed using only the heuristic model.

Justification: This strategy guarantees we have a valuable, working product to ship even if the most complex R&D component underperforms. It transforms the ML model from a critical-path dependency into a progressive enhancement, making our timeline far more resilient.

2. On Defining and Measuring "Recall" with Rigor

This addresses the concern that our core benefit metric could be weak or gamed.



The Commitment: "Recall" will be measured by end-to-end task success, evaluated by a strong LLM judge (GPT-4o), and validated by manual spot-checks.

The Procedure:

Task: At the end of a session, a hold-out question is posed to the agent.

Generation: The agent, using its memory system, generates a final, natural-language answer.

LLM-as-Judge: GPT-4o will be given the generated answer, the ground-truth answer, and the original question. It will output a JSON object with a {"correct": boolean, "reasoning": "..."} schema.

Manual Audit (Critical Step): We will manually review a random sample of at least 50 of the judge's evaluations (especially on borderline cases) to validate its accuracy and check for biases (e.g., is it overly lenient or strict?). If the judge is found to be unreliable, we will refine its prompt or, if necessary, rely on a smaller, fully manual evaluation for the final report.

Justification: This multi-layered approach ensures our central claim ("maintaining ~90% recall") is robust and credible. It avoids relying on simplistic metrics (like keyword matching) and includes a crucial human-in-the-loop step to guard against automated evaluation flaws.

3. On Distinguishing Memory Philosophies for Clear Evaluation

This addresses the potential confusion between different forms of memory compression (R³Mem vs. summarization), which could muddy our benchmark's conclusions.



The Commitment: We will treat and document different memory strategies as distinct "philosophies." The benchmark's goal is to compare the performance of these philosophies, not just their implementations.

The Framework:

PhilosophyMechanismLocus of ActionCost ProfileRole in Our BenchmarkExplicit TriageThe Frugal DispatcherPer-turn, before writing to memoryVariable (compute for decision + storage/API for action)Our core novel contribution.Vector CacheVectorStoreAdapter (FAISS)On-write (embedding) & on-read (search)Storage + compute (embedding/search)The standard industry baseline.Structured MemoryZepGraphAdapterOn-write (entity extraction) & on-readAPI calls / DB operations (more complex)The advanced structured baseline.Implicit CompressionR3MemWrapperAdapterHolistic, continuously on the entire contextLLM tokens (adds to prompt length)A proxy for "in-context only" memory strategies.

Justification: This clear taxonomy prevents us from making "apples-to-oranges" comparisons. It elevates the project's intellectual contribution from merely testing tools to evaluating fundamental strategies for agent memory.

