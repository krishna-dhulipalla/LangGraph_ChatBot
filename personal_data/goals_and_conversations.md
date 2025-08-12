# 🌟 Personal and Professional Goals

## ✅ Short-Term Goals (0–6 months)

1. **Deploy Multi-Agent Personal Chatbot**

   - Integrate RAG-based retrieval, tool calling, and Open Source LLMs
   - Use LangChain, FAISS, BM25, and Gradio UI

2. **Publish Second Bioinformatics Paper**

   - Focus: TF Binding prediction using HyenaDNA and plant genomics data
   - Venue: Submitted to MLCB

3. **Transition Toward Production Roles**

   - Shift from academic research to applied roles in data engineering or ML infrastructure
   - Focus on backend, pipeline, and deployment readiness

4. **Accelerate Job Search**

   - Apply to 3+ targeted roles per week (platform/data engineering preferred)
   - Tailor applications for visa-friendly, high-impact companies

5. **R Shiny App Enhancement**

   - Debug gene co-expression heatmap issues and add new annotation features

6. **Learning & Certifications**
   - Deepen knowledge in Kubernetes for ML Ops
   - Follow NVIDIA’s RAG Agent curriculum weekly

---

## ⏳ Mid-Term Goals (6–12 months)

1. **Launch Open-Source Project**

   - Create or contribute to ML/data tools (e.g., genomic toolkit, chatbot agent framework)

2. **Scale Personal Bot Capabilities**

   - Add calendar integration, document-based Q&A, semantic memory

3. **Advance CI/CD and Observability Skills**

   - Implement cloud-native monitoring and testing workflows

4. **Secure Full-Time Role**
   - Land a production-facing role with a U.S. company offering sponsorship support

---

## 🚀 Long-Term Goals (1–3 years)

1. **Become a Senior Data/ML Infrastructure Engineer**

   - Work on LLM orchestration, agent systems, scalable infrastructure

2. **Continue Academic Contributions**

   - Publish in bioinformatics and AI (focus: genomics + transformers)

3. **Launch a Research-Centered Product/Framework**
   - Build an open-source or startup framework connecting genomics, LLMs, and real-time ML pipelines

---

# 💬 Example Conversations

## Q: _What interests you in data engineering?_

**A:** I enjoy architecting scalable data systems that generate real-world insights. From optimizing ETL pipelines to deploying real-time frameworks like the genomic systems at Virginia Tech, I thrive at the intersection of automation and impact.

---

## Q: _Describe a pipeline you've built._

**A:** One example is a real-time IoT pipeline I built at VT. It processed 10,000+ sensor readings using Kafka, Airflow, and Snowflake, feeding into GPT-4 for forecasting with 91% accuracy. This reduced energy costs by 15% and improved dashboard reporting by 30%.

---

## Q: _What was your most difficult debugging experience?_

**A:** Debugging duplicate ingestion in a Kafka/Spark pipeline at UJR. I isolated misconfigurations in consumer groups, optimized Spark executors, and applied idempotent logic to reduce latency by 30%.

---

## Q: _How do you handle data cleaning?_

**A:** I ensure schema consistency, identify missing values and outliers, and use Airflow + dbt for scalable automation. For larger datasets, I optimize transformations using batch jobs or parallel compute.

---

## Q: _Describe a strong collaboration experience._

**A:** While working on cross-domain NER at Virginia Tech, I collaborated with infrastructure engineers on EC2 deployment while handling model tuning. Together, we reduced latency by 30% and improved F1-scores by 8%.

---

## Q: _What tools do you use most often?_

**A:** Python, Spark, Airflow, dbt, Kafka, and SageMaker are daily drivers. I also rely on Docker, CloudWatch, and Looker for observability and visualizations.

---

## Q: _What’s a strength and weakness of yours?_

**A:**

- **Strength**: Turning complexity into clean, usable data flows.
- **Weakness**: Over-polishing outputs, though I’m learning to better balance speed with quality.

---

## Q: _What do you want to work on next?_

**A:** I want to deepen my skills in production ML workflows—especially building intelligent agents and scalable pipelines that serve live products and cross-functional teams.

## How did you automate preprocessing for 1M+ biological samples?

A: Sure! The goal was to streamline raw sequence processing at scale, so I used Biopython for parsing genomic formats and dbt to standardize and transform the data in a modular way. Everything was orchestrated through Apache Airflow, which let us automate the entire workflow end-to-end — from ingestion to feature extraction. We parallelized parts of the process and optimized SQL logic, which led to a 40% improvement in throughput.

---

## What kind of semantic search did you build using LangChain and Pinecone?

A: We built a vector search pipeline tailored to genomic research papers and sequence annotations. I used LangChain to create embeddings and chain logic, and stored those in Pinecone for fast similarity-based retrieval. It supported both question-answering over domain-specific documents and similarity search, helping researchers find related sequences or studies efficiently.

---

## Can you describe the deployment process using Docker and SageMaker?

A: Definitely. We started by containerizing our models using Docker — bundling dependencies and model weights — and then deployed them as SageMaker endpoints. It made model versioning and scaling super manageable. We monitored everything using CloudWatch for logs and metrics, and used MLflow for tracking experiments and deployments.

---

## Why did you migrate from batch to real-time ETL? What problems did that solve?

A: Our batch ETL jobs were lagging in freshness — not ideal for decision-making. So, we moved to a Kafka + Spark streaming setup, which helped us process data as it arrived. That shift reduced latency by around 30%, enabling near real-time dashboards and alerts for operational teams.

---

## How did you improve Snowflake performance with materialized views?

A: We had complex analytical queries hitting large datasets. To optimize that, I designed materialized views that pre-aggregated common query patterns, like user summaries or event groupings. We also revised schema layouts to reduce joins. Altogether, query performance improved by roughly 40%.

---

## What kind of monitoring and alerting did you set up in production?

A: We used CloudWatch extensively — custom metrics, alarms for failure thresholds, and real-time dashboards for service health. This helped us maintain 99.9% uptime by detecting and responding to issues early. I also integrated alerting into our CI/CD flow for rapid rollback if needed.

---

## Tell me more about your IoT-based forecasting project — what did you build, and how is it useful?

A: It was a real-time analytics pipeline simulating 10,000+ IoT sensor readings. I used Kafka for streaming, Airflow for orchestration, and S3 with lifecycle policies to manage cost — that alone reduced storage cost by 40%. We also trained time series models, including LLaMA 2, which outperformed ARIMA and provided more accurate forecasts. Everything was visualized through Looker dashboards, removing the need for manual reporting.

I stored raw and processed data in Amazon S3 buckets. Then I configured lifecycle policies to:
• Automatically move older data to Glacier (cheaper storage)
• Delete temporary/intermediate files after a certain period
This helped lower storage costs without compromising data access, especially since older raw data wasn’t queried often.
• Schema enforcement: I used tools like Kafka Schema Registry (via Avro) to define a fixed format for sensor data. This avoided issues with malformed or inconsistent data entering the system.
• Checksum verification: I added simple checksum validation at ingestion to verify that each message hadn’t been corrupted or tampered with. If the checksum didn’t match, the message was flagged and dropped/logged.

---

## IntelliMeet looks interesting — how did you ensure privacy and decentralization?

A: We designed it with federated learning so user data stayed local while models trained collaboratively. For privacy, we implemented end-to-end encryption across all video and audio streams. On top of that, we used real-time latency tuning (sub-200ms) and Transformer-based NLP for summarizing meetings — it made collaboration both private and smart.

---

💡 Other Likely Questions:

## Which tools or frameworks do you feel most comfortable with in production workflows?

A: I’m most confident with Python and SQL, and regularly use tools like Airflow, Kafka, dbt, Docker, and AWS/GCP for production-grade workflows. I’ve also used Spark, Pinecone, and LangChain depending on the use case.

---

## What’s one project you’re especially proud of, and why?

A: I’d say the real-time IoT forecasting project. It brought together multiple moving parts — streaming, predictive modeling, storage optimization, and automation. It felt really satisfying to see a full-stack data pipeline run smoothly, end-to-end, and make a real operational impact.

---

## Have you had to learn any tools quickly? How did you approach that?

A: Yes — quite a few! I had to pick up LangChain and Pinecone from scratch while building the semantic search pipeline, and even dove into R and Shiny for a gene co-expression app. I usually approach new tools by reverse-engineering examples, reading docs, and shipping small proofs-of-concept early to learn by doing.
