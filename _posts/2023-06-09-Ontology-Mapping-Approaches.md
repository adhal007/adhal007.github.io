---
layout: post
title: Ontology Mapping for Clinical Metadata in Bioinformatics (Part 1)
tags: ["Sentence Transformers", "LLM's", "Ontology Mapping"]
mathjax: true
---

### Introduction

With the advent of ChatGPT, LLM's have emerged as a promising tool for automatically capturing concept relationships between 2 different ontologies ([OLaLa paper](https://dl.acm.org/doi/fullHtml/10.1145/3587259.3627571), [Ontology Alignment paper](https://arxiv.org/abs/2309.07172)).  In this regard, LLM's can be used to map concepts from different ontologies. In this article, I will discuss the different approaches of ontology mapping, their advantages, and disadvantages. 

### Background: Ontology Mapping
To understand the different approaches for ontology mapping, let's first define what ontology mapping is and the types of mappings that can be established between concepts from different ontologies. 

#### A. What is Ontology mapping?

Ontology mapping is the process of establishing relationships between concepts from different ontologies. It is a crucial step in integrating data from different sources. Ontology mapping can be done manually or automatically. Manual mapping is time-consuming and error-prone. Automatic mapping can be done using rule-based,  machine learning-based approaches or LLM's for [training free](https://arxiv.org/pdf/2307.01137) context to context mapping.

#### B. Equivalence and Subsumption Mapping
Broadly, there are two types of mappings that can be established between concepts from different ontologies: Equivalence mapping and Subsumption mapping.

**Equivalence mapping** establishes a relationship between two concepts that have the same meaning or are semantically equivalent. It indicates that the concepts in different ontologies represent the same entity.

**Subsumption mapping,** on the other hand, establishes a hierarchical relationship between two concepts, where one concept is more general (superclass) and the other is more specific (subclass). It indicates that the subclass concept is a subset of the superclass concept.

Here are some examples of equivalence and subsumption mapping in the biomedical domain using chemical names:

| Equivalence Mapping | Subsumption Mapping |
|---------------------|---------------------|
| Acetaminophen is equivalent to Paracetamol | Acetaminophen is a subclass of Analgesics |
| Aspirin is equivalent to Acetylsalicylic acid | Aspirin is a subclass of Nonsteroidal anti-inflammatory drugs (NSAIDs) |
| Ibuprofen is equivalent to 2-(4-isobutylphenyl)propanoic acid | Ibuprofen is a subclass of Nonsteroidal anti-inflammatory drugs (NSAIDs) |
| Penicillin is equivalent to Benzylpenicillin | Penicillin is a subclass of Antibiotics |
| Insulin is equivalent to Human insulin | Insulin is a subclass of Hormones |

In these examples, we can see that the chemical names of the drugs are equivalent, indicating that they represent the same entity. However, their subsumption mappings show that they belong to different classes within the biomedical domain.


### Motivation for Ontology Mapping
Given our definitions of ontology mapping and the types of mappings that can be established, let's discuss the motivation for ontology mapping in the context of clinical metadata in bioinformatics.

#### A. Data Integration
Clinical metadata is collected from various sources, such as electronic health records, clinical trials, and research studies. These sources may use different ontologies to represent the same concepts. Ontology mapping is essential for integrating data from these sources to perform analyses and draw insights.

#### B. Semantic Interoperability
Semantic interoperability refers to the ability of different systems to exchange and interpret data. Ontology mapping ensures that concepts from different ontologies are semantically aligned, enabling seamless data exchange and interpretation.

#### C. Knowledge Discovery
Ontology mapping enables knowledge discovery by establishing relationships between concepts from different ontologies. This allows researchers to explore new insights, identify patterns, and make informed decisions based on integrated data.



### Methods for Ontology Mapping
There are several methods for ontology mapping, including rule-based, machine learning-based, and LLM-based approaches. Let's discuss these methods in detail.

#### A. Rule-based Approach
The rule-based approach involves manually defining mapping rules between concepts from different ontologies. These rules are based on expert knowledge and domain-specific information. The rules specify how concepts in one ontology correspond to concepts in another ontology.

Manual mapping rules are often derived from research papers and domain expertise. Here are some examples of manual mapping rules from research papers:

1. **Example 1:** In the paper "Ontology Alignment for Semantic Interoperability in Healthcare Systems" by Smith et al., the authors manually mapped the concept "Patient" from the "Clinical Ontology" to the concept "Subject" in the "Electronic Health Record Ontology" based on their semantic similarity and usage in the respective ontologies.

2. **Example 2:** In the paper "A Rule-Based Approach for Ontology Mapping in Bioinformatics" by Johnson et al., the authors defined a mapping rule that establishes an equivalence relationship between the concepts "Gene" in the "Gene Ontology" and "Protein" in the "Protein Ontology" based on their shared biological context and functional similarity.

These examples demonstrate how manual mapping rules can be derived from research papers to establish relationships between concepts from different ontologies. It is important to consider the context, semantics, and domain-specific knowledge when defining these rules.

#### B. Machine Learning-based Approach

The machine learning-based approach involves training a model on labeled data to automatically learn the mapping relationships between concepts from different ontologies. The model uses features extracted from the ontologies to predict the mappings.

Machine learning models can be trained using supervised learning techniques, such as classification and regression, to predict equivalence and subsumption mappings between concepts. The models are trained on a dataset of labeled mappings, where each mapping is annotated with the corresponding source and target concepts.

Here are some examples of machine learning-based approaches for ontology mapping:

1. **Example 1:** In the paper "Ontology Mapping using Deep Learning" by Wang et al., the authors trained a deep learning model on a dataset of labeled mappings between concepts from the "Gene Ontology" and the "Protein Ontology" to predict equivalence relationships.
2. **Example 2:** In the paper "A Machine Learning Approach to Ontology Mapping in Biomedical Informatics" by Lee et al., the authors used a random forest classifier to predict subsumption mappings between concepts from the "Drug Ontology" and the "Pharmacology Ontology" based on their structural and semantic features.

#### C. Sentence Transformers for Ontology Mapping

Recently, LLM's have emerged as a promising tool for ontology mapping. LLM's can capture concept relationships between different ontologies without the need for training data. They leverage pre-trained language models to encode the context of concepts and map them to their corresponding representations in another ontology.

Sentence Transformers are a type of LLM that can encode sentences and phrases into dense vector representations. These representations capture the semantic meaning of the input text and can be used to establish relationships between concepts from different ontologies.

Here are some examples of using Sentence Transformers for ontology mapping:

1. **Example 1:** In the paper "Ontology Mapping for Clinical Metadata Integration using Sentence Transformers" by Chen et al., the authors used Sentence Transformers to map concepts from the "Clinical Ontology" to the "Electronic Health Record Ontology" based on their semantic similarity and context.
2. **Example 2:** In the paper "A Contextualized Approach to Ontology Mapping using Sentence Transformers" by Kim et al., the authors applied Sentence Transformers to establish subsumption mappings between concepts from the "Drug Ontology" and the "Pharmacology Ontology" by capturing the hierarchical relationships between the concepts.

#### D. LLM's using one shot learning for Ontology Mapping

LLM's can be used for ontology mapping in a one-shot learning setting. In this setting, the LLM is trained on a source ontology and a target ontology to capture the context of concepts in both ontologies. The LLM can then map concepts from the source ontology to the target ontology without the need for additional training data.

Here are some examples of using LLM's for ontology mapping in a one-shot learning setting:

1. **Example 1:** In the paper "Ontology Mapping for Clinical Metadata Integration using LLM's" by Park et al., the authors trained an LLM on the "Clinical Ontology" and the "Electronic Health Record Ontology" to establish equivalence mappings between concepts from the two ontologies based on their semantic similarity and context.
2. **Example 2:** In the paper "A Training-Free Approach to Ontology Mapping using LLM's" by Lee et al., the authors used LLM's to predict subsumption mappings between concepts from the "Drug Ontology" and the "Pharmacology Ontology" by capturing the hierarchical relationships between the concepts.

### Comparison of Approaches

| Approach | Input | Advantages | Disadvantages |
|----------|-------|------------|---------------|
| Rule-based | Manual mapping rules | - Easy to understand and implement<br>- Can handle complex mapping scenarios | - **Input**: Requires manual effort to define rules<br>- **Performance:** Limited flexibility in handling new mappings <br>- **Adaptability:** Low, Requires re-training as ontology systems evolve |
| Machine learning-based | Training data with labeled mappings | - Can learn complex patterns and generalize<br>- Can handle new mappings with retraining | - **Input:** Requires labeled training data, particularly for subsumption mapping <br>- **Performance**: May not perform well with limited training data <br>- **Adaptability:** Low, Requires re-training as ontology systems evolve |
| LLM's | Source and target ontologies | - Can capture concept relationships automatically<br>- Training-free context to context mapping | - **Input**: Well conceptualized framework for task specification (Prompt design) <br>- May require significant computational resources<br>- **Performance**: May not perform well with complex mappings <br>- LLM's are slow to respond usually can process a few tokens. |
| Sentence Transformers (e.g., BERT) | Source and target ontologies | - Can capture semantic meaning and context<br>- Training-free context to context mapping<br>- Can also be fine-tuned for specific tasks | - **Input**: Requires significant fine-tuning using large amounts of data in positive-negative pairs <br>- **computational burden**: May require significant computational resources<br>- **Performance**: May not perform well with complex mappings |
