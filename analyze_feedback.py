#!/usr/bin/env python3
"""
Feedback Analysis Tool
Performs NLP analysis on survey feedback including sentiment analysis, topic modeling, and clustering.
"""

import argparse
import logging
import os
import re
import sys
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from textblob import TextBlob
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
REQUIRED_COLS = ["Went_Well", "Did_Not_Go_Well", "Improvement_Suggestions", "Other_Comments"]
DEFAULT_NUM_TOPICS = 3
DEFAULT_NUM_CLUSTERS = 3

def check_dependencies():
    """Ensure required packages are installed."""
    required_packages = {
        'xlsxwriter': 'xlsxwriter',
        'textblob': 'textblob',
        'wordcloud': 'wordcloud'
    }
    
    missing = []
    for pkg, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.info("Installing missing packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

def preprocess_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text.strip()

def analyze_feedback(input_path, output_path, num_topics=DEFAULT_NUM_TOPICS, num_clusters=DEFAULT_NUM_CLUSTERS):
    """
    Main analysis function that processes feedback data and generates insights.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save output Excel file
        num_topics (int): Number of topics for LDA analysis
        num_clusters (int): Number of clusters for KMeans
    """
    try:
        # Load and validate data
        logger.info(f"Loading data from {input_path}")
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='latin1')
            
        # Check required columns
        missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Preprocess data
        logger.info("Preprocessing feedback data...")
        df = df.fillna('')
        df["All_Feedback"] = df[REQUIRED_COLS].agg(" ".join, axis=1)
        df["Cleaned_Feedback"] = df["All_Feedback"].apply(preprocess_text)
        df = df[df['Cleaned_Feedback'].str.len() > 0]
        
        # Sentiment Analysis
        logger.info("Performing sentiment analysis...")
        df["Sentiment"] = df["Cleaned_Feedback"].apply(lambda x: TextBlob(x).sentiment.polarity)
        
        # Topic Modeling
        logger.info("Running topic modeling...")
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        dtm = vectorizer.fit_transform(df["Cleaned_Feedback"])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)
        
        # Get top words for each topic
        topics_df = pd.DataFrame()
        for i, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:]]
            topics_df[f'Topic {i+1}'] = pd.Series(top_words)
        
        # Clustering
        logger.info("Clustering feedback responses...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(dtm)
        
        # Generate output file
        logger.info(f"Saving results to {output_path}")
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Save raw data
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Save topics
            topics_df.to_excel(writer, sheet_name='Topics', index=False)
            
            # Save summary statistics
            summary_df = pd.DataFrame({
                'Metric': ['Total Responses', 'Average Sentiment', 
                          'Positive Responses', 'Neutral Responses', 
                          'Negative Responses'],
                'Value': [
                    len(df),
                    df['Sentiment'].mean(),
                    len(df[df['Sentiment'] > 0.1]),
                    len(df[(df['Sentiment'] >= -0.1) & (df['Sentiment'] <= 0.1)]),
                    len(df[df['Sentiment'] < -0.1])
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Save cluster details
            cluster_details = df.groupby('Cluster').agg({
                'Sentiment': 'mean',
                'Cleaned_Feedback': 'count'
            }).rename(columns={'Cleaned_Feedback': 'Count'})
            cluster_details.to_excel(writer, sheet_name='Cluster Details')
            
            # Add visualizations
            workbook = writer.book
            worksheet = workbook.add_worksheet('Visualizations')
            
            # Sentiment plot
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Sentiment'], bins=20, kde=True)
            plt.title('Sentiment Distribution')
            img_data = BytesIO()
            plt.savefig(img_data, format='png')
            plt.close()
            worksheet.insert_image('A1', '', {'image_data': img_data})
            
            # Cluster distribution
            plt.figure(figsize=(8, 5))
            df['Cluster'].value_counts().plot(kind='bar')
            plt.title('Response Clusters Distribution')
            img_data = BytesIO()
            plt.savefig(img_data, format='png')
            plt.close()
            worksheet.insert_image('A20', '', {'image_data': img_data})
            
            # Word Clouds for each cluster
            row_offset = 40
            for cluster in range(num_clusters):
                cluster_text = " ".join(df[df["Cluster"] == cluster]["Cleaned_Feedback"])
                if cluster_text:
                    plt.figure(figsize=(10, 5))
                    wordcloud = WordCloud(width=800, height=400, 
                                        background_color='white').generate(cluster_text)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.title(f"Cluster {cluster+1} Word Cloud")
                    img_data = BytesIO()
                    plt.savefig(img_data, format='png')
                    plt.close()
                    worksheet.insert_image(f'A{row_offset}', '', {'image_data': img_data})
                    row_offset += 20
        
        logger.info("Analysis completed successfully!")
        print("\nSummary Statistics:")
        print(f"Total feedback responses analyzed: {len(df)}")
        print(f"Average sentiment score: {df['Sentiment'].mean():.2f}")
        print("Cluster distribution:")
        print(df['Cluster'].value_counts())
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

def main():
    """Handle command-line arguments and execute analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze feedback data using NLP techniques',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True, 
                       help='Path to input CSV file')
    parser.add_argument('-o', '--output', required=True,
                       help='Path for output Excel file')
    parser.add_argument('-t', '--topics', type=int, default=DEFAULT_NUM_TOPICS,
                       help='Number of topics for LDA analysis')
    parser.add_argument('-c', '--clusters', type=int, default=DEFAULT_NUM_CLUSTERS,
                       help='Number of clusters for KMeans')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    check_dependencies()
    analyze_feedback(args.input, args.output, args.topics, args.clusters)

if __name__ == "__main__":
    main()
