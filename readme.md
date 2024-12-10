# README.md

## From Lived Experiences to Legal Frameworks: Reconciling Public Perceptions with Formal Data

This repository contains all the code and data necessary for implementing the project described in the study: **From Lived Experiences to Legal Frameworks: Reconciling Public Perceptions with Formal Data**. The focus of this repository is to train and deploy a Large Language Model (LLM) system designed for context-aware, empathetic responses grounded in statistical and lived-experience data.

---
### **Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/SK-143381/is-597-hcds.git
   ```
2. Ensure you have the necessary credentials for accessing GPT-based models (if applicable).

---

### **Usage**

#### **Using the Large Language Model (LLM)**

1. Navigate to `llm/chat.ipynb`.
2. Open the notebook in a Jupyter-compatible environment.
3. Run the **first cell** to initialize the environment and load the dependencies.
4. Run the **fourth cell** to start interacting with the LLM.

The LLM supports queries related to:
- Accessibility recommendations (e.g., physical activity, workplace accommodations).
- Legal guidance based on accessibility frameworks.
- General empathetic, context-aware advice grounded in public data and sentiment analysis.

---
### Resources

You may find the legal data in the `legal-data` folder. Quantitative analysis of NHIS data is available at `publicly-sourced_data/filtering-disabilities` and Reddit data topic modelling can be found in `social-media_data`.

---
### **Feedback and Contributions**

We welcome contributions to improve accessibility, inclusivity, or legal compliance in the LLM system. Submit a pull request or raise an issue in the repository. For any questions, contact the contributors.

Happy exploring!
