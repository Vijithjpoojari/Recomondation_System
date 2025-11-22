# 🎯 Cross-Domain Meta-Learning Recommendation System  
### Deep Learning · Meta-Learning · MAML · Reptile · NCF · VAE · GNN  
### Domains: Movies · Books · Music  

---

## 📌 Project Overview
This project develops a **deep learning–based cross-domain recommendation system** that adapts seamlessly across multiple content categories (Movies, Books, Music) using **meta-learning techniques**.  
The goal is to create a model that can quickly personalize recommendations for new users or new domains using only **minimal interaction data**, providing high-quality and diverse recommendations.

The system utilizes:
- **Model-Agnostic Meta-Learning (MAML)**
- **Reptile**
- **Neural Collaborative Filtering (NCF)**
- **Variational Autoencoder (VAE)**
- **Graph Neural Network (GNN)**

Based on experiments and performance analysis, **MAML** was identified as the most effective for few-shot cross-domain personalization.

---

# 📊 Dataset Metadata

## 🎬 1. Movies (MovieLens)
- **Source:** MovieLens  
- **Attributes:** User ID, Movie ID, Rating, Timestamp, Genre  
- **Sample Size:** ~28,000 ratings, 1,000 users, 1,500 movies  

## 📚 2. Books (Goodreads)
- **Source:** Goodreads  
- **Attributes:** Book ID, Title, Authors, Average Rating, Ratings Count  
- **Sample Size:** ~12,000 books with strong engagement  

## 🎵 3. Music (Last.fm)
- **Source:** Last.fm  
- **Attributes:** User ID, Track ID, Play Count, Timestamp, Genre  
- **Sample Size:** ~170,000 plays across 3,000 users and 10,000 tracks  

---

# 🔍 Exploratory Data Analysis (EDA)

### ✔️ Descriptive Statistics
- Mean, median, variance of ratings  
- Play count distribution  
- Genre frequency analysis  

### ✔️ User & Item Interaction Distribution
- Activity distribution per user  
- Popularity distribution per item  
- Long-tail patterns observed  

### ✔️ Sparsity Check
- High sparsity across all domains  
- Especially sparse interactions in books & music  

### ✔️ Cross-Domain Preference Correlation
- Weak-to-moderate similarity of tastes across domains  
- Cross-domain user embeddings required  

---

# 🛠️ Preprocessing Pipeline

### 1. Data Cleaning
- Remove duplicates  
- Fix inconsistent formats  
- Handle missing entries  

### 2. Normalization
- Min-Max normalization  
- Log scaling for play counts  
- Standard scaling for rating counts  

### 3. Feature Engineering
- User & item embeddings  
- Genre encoding (multi-hot / embeddings)  
- Unified feature vector length for all domains  

### 4. Data Splitting
- Train / validation / test  
- Support / query split for meta-learning tasks  

### 5. Domain Alignment
To unify multi-domain data:
- Common latent embedding space  
- Genre → universal vector mapping  
- Unified metadata representations  

---

# 🔎 Performance Metrics

### Ranking Metrics  
- **Precision@K**  
- **Recall@K**  
- **F1 Score**  
- **NDCG@K**

### Error Metrics  
- **MSE**  
- **MAE**

### System Metrics  
- **Coverage**  
- **Cold-start performance**  
- **Domain generalization score**  

---

# 🎯 Project Objectives

1. **Cross-Domain Adaptability**  
   Build a unified recommendation model capable of learning from multiple domains.

2. **High Personalization**  
   Provide accurate and meaningful recommendations with minimal input data.

3. **Efficiency**  
   Quick adaptation and low training time for new users.

4. **User Satisfaction**  
   Improve engagement with diverse, personalized recommendations.

---

# 🧠 Literature Review Summary

## 📘 Paper 1: Personalized Federated Meta-Learning (MAML-Based)
- Uses MAML to personalize federated recommenders  
- Improves efficiency with sparse attention mechanisms  

## 📘 Paper 2: Meta-Learning Research Review
- Covers 17 major meta-learning contributions across top conferences  
- Shows versatility of meta-learning beyond recommendation systems  

## 📘 Paper 3: LiMAML (LinkedIn)
- Industry-scale meta-learning recommender  
- Converts sub-networks into meta-embeddings  
- Works with huge commercial datasets  

---

# 🤖 Model Pros & Cons

| Model | Pros | Cons |
|-------|------|------|
| **MAML** | Excellent personalization, few-shot, best cross-domain | High compute |
| **Reptile** | Fast, easy, lightweight meta-learning | Slightly less accurate |
| **NCF** | Learns nonlinear interactions | Requires more data |
| **VAE** | Strong latent representations | Complex to train |
| **GNN** | Captures graph relations | Heavy computation |

---

# 🏆 Selected Models for Implementation
1. **Model-Agnostic Meta-Learning (MAML)**  
2. **Reptile**  
3. **Neural Collaborative Filtering (NCF)**  
4. **Variational Autoencoders (VAE)**  
5. **Graph Neural Networks (GNN)**  

---

# 🧩 Baseline Model: FCNN

### Input Features:
- User ID  
- Item ID  
- Genre / Category  
- Play count / Rating  
- Author (books)  

### Architecture:
- Embedding layers  
- Dense layers  
- Final regression node  

---

# 🔁 End-to-End Working Pipeline

### ✔️ Step 1: Data Loading and Preprocessing
Load, clean, normalize, encode, and align multi-domain data.

### ✔️ Step 2: Task Construction for Meta-Learning
Each user-domain dataset → creates a meta-learning task.

### ✔️ Step 3: Meta-Learning Training
- Train with MAML or Reptile  
- Evaluate inner-loop and outer-loop loss  

### ✔️ Step 4: Multi-Domain Evaluation
- Precision@K  
- Recall@K  
- MSE  
- Coverage  

---

# 🧪 Sample Code (Simplified MAML)

```python
meta_model = NCF(num_users, num_items)
meta_opt = Adam(meta_model.parameters(), lr=1e-3)

for episode in range(EPISODES):
    tasks = sample_tasks(batch_size)
    meta_opt.zero_grad()

    for task in tasks:
        support, query = task.support, task.query

        fast_model = clone(meta_model)
        fast_opt = SGD(fast_model.parameters(), lr=0.01)

        for _ in range(5):
            pred = fast_model(support.users, support.items)
            loss = mse(pred, support.labels)
            fast_opt.zero_grad()
            loss.backward()
            fast_opt.step()

        q_pred = fast_model(query.users, query.items)
        q_loss = mse(q_pred, query.labels)
        q_loss.backward()

    meta_opt.step()
