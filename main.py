from transformers import pipeline
import pandas as pd

# Load the pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# Load the pre-trained zero-shot classification model
classification = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the categories for classification
categories = ["Compensation and Benefits", "Work Life Balance", "Job Security", "Culture", "Career Path"]

# Define input sentences
sentences = [
    """
    I also started in the industry as an underwriting trainee at 23.
    I felt like I was having a quarter life crisis and wanted to quit so many times -
    both during training and even afterwards. My friends were taking on fun recruiting jobs
    or joining sexy tech start-ups with other people in their early-20s,
    while I felt trapped at a desk.
    I also struggled to learn the material and was uncomfortable at being bad at my job.
    Now I'm 27, making $120k, and couldn't be more excited to continue growing in this industry.
    """,
    """
    I also started in the industry as an underwriting trainee at 23.
    I felt like I was having a quarter life crisis and wanted to quit so many times -
    both during training and even afterwards. My friends were taking on fun recruiting jobs
    or joining sexy tech start-ups with other people in their early-20s,
    while I felt trapped at a desk.
    I also struggled to learn the material and was uncomfortable at being bad at my job.
    Now I'm 27, making ends meet.
    """,
    """
    I get good bonus from the company.
    """,
    """
    For anyone new to the industry or struggling to find their place,
    I recommend seeking out a mentor. Be annoying, ask them a million questions
    (I liked to have a list prepared for every time I met with my mentor).
    I'd also encourage you to hang in there and give the industry at least a full year
    before you start considering going elsewhere. The subject matter is difficult,
    but I promise none of us in the industry understood insurance intuitively.
    Be patient with yourself and stay curious.
    """,
    """
    I went through similar concerns with my UW training program and voiced them to my manager.
    Learning insurance from the ground up is hard, and you likely won't know what
    you're doing for the first 1-2 years. I've even had brokers tell me,
    You don't know what you're doing, and it was the most embarrassing experience ever.
    As a perfectionist, I had so much angst about not excelling at everything
    I do, but if you stick in, the industry is extremely rewarding and a
    lifelong learning experience (you'll never know everything).
    """,
    """
    I absolutely hated it and found it to be so extremely monotonous.
    I've also been in claims and brokerage. Underwriting was my least fave of all 3.
    You feel like a used car salesman, hoping a broker picks you and your quote.
    """
]

# Initialize results list
results = []

# Process each sentence
for sentence in sentences:
    # Perform sentiment analysis
    sentiment_result = sentiment_analysis(sentence)
    sentiment = sentiment_result[0]["label"]
    sentiment_confidence = sentiment_result[0]["score"] * 100

    # Perform zero-shot classification
    classification_result = classification(sentence, candidate_labels=categories)

    # Store results
    classifications = {label: score * 100 for label, score in zip(classification_result["labels"], classification_result["scores"])}
    results.append([sentence.strip(), sentiment, sentiment_confidence] + [classifications[category] for category in categories])

# Prepare headers
headers = ["Sentence", "Sentiment", "Confidence"] + categories

# Create a DataFrame
df = pd.DataFrame(results, columns=headers)

# Display the DataFrame
print(df)