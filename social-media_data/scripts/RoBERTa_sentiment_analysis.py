import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import urllib.request
import csv

def load_model_and_tokenizer(task='sentiment'):
    """
    Loads a pre-trained model and tokenizer from Hugging Face for a specific task.

    Args:
        task (str): The task for which the model was trained (e.g., 'sentiment').

    Returns:
        model: The loaded sequence classification model.
        tokenizer: The tokenizer corresponding to the model.
    """
    model_name = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def load_labels(task='sentiment'):
    """
    Downloads and loads the label mapping for the selected task.

    Args:
        task (str): The task for which the model was trained (e.g., 'sentiment').

    Returns:
        labels (list): List of classification labels.
    """
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
    return labels

def preprocess_text(text):
    """
    Preprocesses text by replacing usernames and links with placeholders.

    Args:
        text (str): The original text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(text, model, tokenizer, labels, max_length=512):
    """
    Analyze sentiment of a given text using a pre-trained model.
    
    Args:
    - text (str): The input text (comment) for sentiment analysis.
    - model: Pre-trained model for sequence classification.
    - tokenizer: Tokenizer corresponding to the pre-trained model.
    - labels (list): List of sentiment labels.
    - max_length (int): Maximum length of the tokenized sequence.
    
    Returns:
    - dict: Dictionary mapping sentiment labels to their respective scores.
    """
    # Tokenize the text with truncation and dynamic padding
    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,  # Truncate texts longer than `max_length`
        padding='max_length',  # Pad texts shorter than `max_length`
        max_length=max_length  # Max length of 512 tokens
    )
    
    # Run the model and get the output
    output = model(**encoded_input)
    
    # Extract the scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Create a dictionary with the results
    result = {labels[i]: scores[i] for i in range(len(scores))}
    
    return result

def process_comments(df, model, tokenizer, labels):
    """
    Iterates over the DataFrame and analyzes the sentiment of each comment.

    Args:
        df (pd.DataFrame): DataFrame containing the comments.
        model: Pre-trained sentiment analysis model.
        tokenizer: Tokenizer corresponding to the model.
        labels (list): List of sentiment classification labels.

    Returns:
        pd.DataFrame: Updated DataFrame with sentiment scores.
    """
    # Create new columns to store sentiment scores
    df['roBERTa_sentiment_positive'] = np.nan
    df['roBERTa_sentiment_negative'] = np.nan
    df['roBERTa_sentiment_neutral'] = np.nan
    df['preponderant_sentiment'] = np.nan
    
    # Iterate through the comments and analyze sentiment
    for index, row in df.iterrows():
        comment = row['comments']
        
        if pd.isna(comment):  # Skip null comments
            continue
        
        # Analyze the sentiment of the comment
        sentiment_scores = analyze_sentiment(comment, model, tokenizer, labels)
        
        # Store the scores in the DataFrame
        df.at[index, 'roBERTa_sentiment_positive'] = sentiment_scores.get('positive', 0)
        df.at[index, 'roBERTa_sentiment_negative'] = sentiment_scores.get('negative', 0)
        df.at[index, 'roBERTa_sentiment_neutral'] = sentiment_scores.get('neutral', 0)

        # Determine the preponderant sentiment (sentiment with highest score)
        preponderant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        
        # Store the preponderant sentiment label in the DataFrame
        df.at[index, 'preponderant_sentiment'] = preponderant_sentiment

    return df

def save_results(df, output_file):
    """
    Saves the sentiment analysis results to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the analysis results.
        output_file (str): Path to save the CSV file.
    """
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Main function to execute the sentiment analysis pipeline
def main(input_file, output_file):
    """
    Main function that executes the complete sentiment analysis pipeline.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file with sentiment scores.
    """
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load sentiment labels
    labels = load_labels()
    
    # Process the comments and get sentiment scores
    df = process_comments(df, model, tokenizer, labels)
    
    # Save the results
    save_results(df, output_file)

# Example usage
if __name__ == "__main__":
    input_file = "social-media_data/output_sentiment_analysis/vader_sentiment_analysis_results.csv"
    output_file = "social-media_data/output_sentiment_analysis/final_output_sentiment_RoBERTa_Vader.csv"
    main(input_file, output_file)
