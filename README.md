# 🗳️ US Election 2024 Sentiment Analysis and Topic Modeling with Reddit Posts  

Welcome to our project repository for analyzing the US Presidential Election 2024 through Reddit data! This project combines data collection, sentiment classification, and topic modeling to explore the polarization and prevailing themes in the political discourse surrounding the upcoming election.

---

## 📖 Project Overview  

The 2024 US Presidential Election is shaping up to be a pivotal moment in modern political history. Understanding the sentiment and topics driving public opinion can provide valuable insights into voter behavior and societal trends.  

This project aims to:  
1. **Collect Reddit posts**: Scrape posts from various subreddits (e.g., `r/politics`, `r/PoliticalDiscussion`, etc.) that discuss the election and filter them for relevance to key topics like the candidates, major policies, and public sentiment.  
2. **Sentiment Analysis**: Train and deploy machine learning models to classify posts into three categories:  
   - **Pro-Trump**  
   - **Pro-Harris**  
   - **Neutral**  
3. **Topic Modeling**: Identify the dominant themes and issues that shape the political discourse leading up to the election.  
4. **Polarization Analysis**: Use the insights from sentiment and topic modeling to assess the degree of polarization in the 2024 election.  

---

## 🔍 Features  

- **Data Collection**:  
  - Scrape thousands of Reddit posts using the `praw` library.  
  - Focus on political subreddits for diverse perspectives on the election.  
  - Apply time-based filters to analyze posts during key election milestones.  

- **Machine Learning Models**:  
  - Fine-tune transformer-based models (e.g., BERT) using Hugging Face's Trainer API for sentiment classification.  
  - Leverage zero-shot classification for quick analysis of unseen data.  

- **Topic Modeling**:  
  - Use NLP libraries like `spaCy` and `gensim` for extracting topics.  
  - Visualize relationships between terms using word embeddings (e.g., GloVe).  

- **Data Analysis**:  
  - Explore sentiment trends over time.  
  - Investigate the most influential words and phrases related to each candidate.  

---

## 🚀 Getting Started  

### Prerequisites  

- Python 3.8+  
- Recommended packages:  
  - `transformers`, `datasets`, `praw`, `gensim`, `spaCy`, `scikit-learn`, `matplotlib`  

### Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/us-election-2024-analysis.git
   cd us-election-2024-analysis

