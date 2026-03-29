
<h2 align="center">Lab 01 - Introduction to Generative AI</h2>

## Objectives:  
* Learn about using different types of prompting styles
* Use OpenAI's API
* Calling Hugging Face and its APIs
* Build multi-turn conversational AI with prompt engineering
* Perform sentiment analysis using real-world datasets
* Additional Resources



## Exercise 1:

1. In this lab we will use OpenAI API to develop an application in python
2. Create an account at https://www.deeplearning.ai
3. Register to the free course at https://learn.deeplearning.ai/courses/chatgpt-building-system
4. Select the second activity
   

<img src="/resources/lab02.png" alt="Alt text" style="width: 60%;"/>

4. Execute the first code line in Jupiter notebook
5. Add the following code and in a codeblock and run

```
# Define system prompt (initial behavior) 
system_prompt = {"role": "system", "content": "You are a helpful assistant with deep knowledge in physics."} 
 
# Define user prompt (specific query) 
user_prompt = {"role": "user", "content": "Explain how black holes are formed."} 
 
# Send prompt to API 
response = openai.ChatCompletion.create( 
    model="gpt-3.5-turbo", 
    messages=[system_prompt, user_prompt] 
) 
print(response['choices'][0]['message']['content'])
```
Please see the figure below if you have issues running this</br>

<img src="/resources/lab03.png" alt="Alt text" style="width: 45%;"/>



6. Copy the codeblock given above again to the next cell and perform the following changes
7. Give a System Prompt for a chatbot to behave as an expert event planner
8. Give a User Prompt asking for advice how to plan for an event you have to do, maybe your friends Birthday Party, or a Batch Outing
9. Tryout a unique System prompt and User Prompt for a problem of your choice.
10. Share screen shots and upload the prompts and responses for the above activities 

## Exercise 2.
1. Create an account in https://huggingface.co
2. Select any AI project of your liking under Spaces
3. Tryout the AI project and generate the output
4. Share screenshots of your usage of the AI project
5. Under Models select a Model of your choice to find more details about it
6. Create an account in https://colab.research.google.com
7. Change Runtime to GPU

<img src="/resources/lab04.png" alt="Alt text" style="width: 30%;"/>

9. Run the summarization example in a codeblock, here the Hugging Face open model will be downloaded to Google Colab and then it will be exectuted. Because of this it will take some time to run the model for the first time.




```
from transformers import pipeline
summarizer = pipeline("summarization")
summary = summarizer("""Hugging Face is a pioneering open-source AI platform
  that empowers developers, researchers, and organizations to easily access and
  deploy cutting-edge machine learning models for natural language processing, computer vision,
  and beyond—democratizing AI and accelerating innovation through a collaborative ecosystem of tools,
  datasets, and community-driven contributions.""", max_length=20)
print(summary)

```
10. Have a look at a specific summarization model available in HuggingFace - https://huggingface.co/EbanLee/kobart-summary-v3
11. Run the summarization for this model given below, here the Hugging Face open model will be downloaded to Google Colab and then it will be exectuted. Because of this it will take some time to run the model for the first time.

```
from transformers import pipeline

# Use a known summarization model
summarizer = pipeline("summarization", model="EbanLee/kobart-summary-v3")

# Input text to summarize
text = """Hugging Face is a pioneering open-source AI platform
that empowers developers, researchers, and organizations to easily access and
deploy cutting-edge machine learning models for natural language processing, computer vision,
and beyond—democratizing AI and accelerating innovation through a collaborative ecosystem of tools,
datasets, and community-driven contributions."""

# Generate summary
summary = summarizer(text, max_length=20, min_length=5, do_sample=False)
print(summary)
```
12. Select a Natural Language Model of your choice from Hugging Face

<img src="/resources/lab05.png" alt="Alt text" style="width: 30%;"/>

13. Try your best to see if you can get your selected Model running in Google CoLab.  In some instances you will have to install additional python libraries.
Some models have on the Right Side called Use this Model, where you can get Google CoLab code that you can try to use.
<img src="/resources/lab06.png" alt="Alt text" style="width: 25%;"/>
15. Attach screen shots of your solution running for 9, 11, and 13
16. Submit the code and the output for 13

---

## Exercise 3: Prompt Engineering & Multi-Turn Conversations

> **Goal:** Go beyond single-turn prompts. Learn how to guide LLM behaviour with advanced prompting techniques and build a conversation that maintains history across multiple turns.

### Part A — Few-Shot Prompting

Few-shot prompting gives the model examples of the input/output pattern you want before asking your real question. This is one of the most effective ways to control output format and style.

1. In a new Google Colab notebook, install the OpenAI library and set your API key:

```python
!pip install openai
import openai
openai.api_key = "YOUR_API_KEY"
```

2. Run the following few-shot example. Notice how the model learns the classification format purely from the examples — no fine-tuning required:

```python
few_shot_prompt = [
    {"role": "system", "content": "You are a product review classifier. Classify reviews as POSITIVE, NEGATIVE, or NEUTRAL. Reply with only the label."},
    {"role": "user",   "content": "This laptop is absolutely amazing, best purchase I have ever made!"},
    {"role": "assistant", "content": "POSITIVE"},
    {"role": "user",   "content": "The battery died after 2 hours. Very disappointing."},
    {"role": "assistant", "content": "NEGATIVE"},
    {"role": "user",   "content": "It arrived on time and the packaging was fine."},
    {"role": "assistant", "content": "NEUTRAL"},
    {"role": "user",   "content": "I love the keyboard but the screen is a bit too dim for my taste."},
]

response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=few_shot_prompt)
print(response['choices'][0]['message']['content'])
```

3. Change the final user message to three reviews of your own choice and observe how the model responds.

### Part B — Chain-of-Thought (CoT) Prompting

Chain-of-thought prompting instructs the model to reason step-by-step before giving a final answer. This dramatically improves accuracy on tasks that require logic or reasoning.

4. Run the following example and compare the outputs with and without CoT:

```python
# Without Chain-of-Thought
simple_prompt = [
    {"role": "system", "content": "Answer the question directly."},
    {"role": "user",   "content": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have?"}
]

# With Chain-of-Thought
cot_prompt = [
    {"role": "system", "content": "Think through this step by step before giving your final answer."},
    {"role": "user",   "content": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have?"}
]

for label, prompt in [("Without CoT", simple_prompt), ("With CoT", cot_prompt)]:
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt)
    print(f"\n--- {label} ---")
    print(response['choices'][0]['message']['content'])
```

### Part C — Multi-Turn Conversation with Memory

A key feature of chat-based LLMs is memory within a conversation. The model does not automatically remember previous turns — you must pass the full conversation history in each API call.

5. Run the following interactive loop. Type `exit` to end the conversation:

```python
import openai
openai.api_key = "YOUR_API_KEY"

# The conversation history is maintained as a list
conversation_history = [
    {"role": "system", "content": "You are a friendly tutor helping a student learn about machine learning. Keep answers concise and use simple analogies."}
]

print("Chat with your ML tutor! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Append the new user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Send the full history to the API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    assistant_reply = response['choices'][0]['message']['content']

    # Append assistant reply so the next turn remembers it
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    print(f"\nTutor: {assistant_reply}\n")
```

6. Have at least a **5-turn conversation** with the tutor. Ask follow-up questions that reference earlier parts of the conversation (e.g., "Can you give me an example of what you just described?") to observe how the model uses conversation history.

### ✅ Deliverables for Exercise 3
- Screenshot of your few-shot classifier output for your 3 custom reviews
- Screenshot comparing CoT vs non-CoT responses
- Screenshot of your 5-turn multi-turn conversation
- In your own words (2–3 sentences): **why does passing the full history matter, and what would happen if you only sent the latest message?**

---

## Exercise 4: Sentiment Analysis with a Real Dataset

> **Goal:** Use a pre-trained Hugging Face model to perform sentiment analysis on a real-world dataset, then evaluate its performance using standard classification metrics.

In Exercise 2 you ran a summarisation pipeline on text you typed yourself. Now you will load a real benchmark dataset, run inference across multiple samples, and measure how well the model performs.

### Part A — Load the Dataset

1. In a new Google Colab notebook (GPU runtime), install the required libraries:

```python
!pip install datasets transformers scikit-learn
```

2. Load the **Tweet Eval** sentiment dataset from Hugging Face. This dataset contains tweets labelled as `negative (0)`, `neutral (1)`, or `positive (2)`:

```python
from datasets import load_dataset

dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

# Inspect the dataset structure
print(dataset)
print("\nExample record:")
print(dataset['test'][0])
```

3. Take a random sample of 100 test examples to keep inference fast on the free Colab tier:

```python
import random
random.seed(42)

test_samples = dataset['test'].shuffle(seed=42).select(range(100))
texts  = test_samples['text']
labels = test_samples['label']   # Ground truth: 0=negative, 1=neutral, 2=positive

print(f"Loaded {len(texts)} test samples")
print("Label distribution:", {i: labels.count(i) for i in [0, 1, 2]})
```

### Part B — Run the Sentiment Pipeline

4. Load a pre-trained sentiment model fine-tuned on tweets and run it over all 100 samples:

```python
from transformers import pipeline

# This model was trained specifically on tweet sentiment
classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True
)

print("Running inference on 100 samples...")
predictions_raw = classifier(texts, batch_size=16)

# Map model output labels to numeric labels
label_map = {"negative": 0, "neutral": 1, "positive": 2}
predictions = [label_map[p['label'].lower()] for p in predictions_raw]

print("Done!")
print("First 5 predictions:", predictions[:5])
print("First 5 ground truth:", list(labels[:5]))
```

### Part C — Evaluate Performance

5. Calculate accuracy and a full classification report:

```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(labels, predictions)
report   = classification_report(labels, predictions,
                                  target_names=["Negative", "Neutral", "Positive"])

print(f"Accuracy: {accuracy:.2%}\n")
print("Classification Report:")
print(report)
```

6. Visualise the confusion matrix:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Negative", "Neutral", "Positive"])
disp.plot(cmap="Blues")
plt.title("Sentiment Analysis — Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
```

### Part D — Explore Errors

7. Identify cases where the model was wrong and inspect them:

```python
errors = [(texts[i], labels[i], predictions[i])
          for i in range(len(texts)) if labels[i] != predictions[i]]

print(f"Total errors: {len(errors)} out of {len(texts)}\n")

print("--- Sample Misclassifications ---")
for text, true, pred in errors[:5]:
    label_name = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"Text      : {text[:120]}")
    print(f"True label: {label_name[true]}  |  Predicted: {label_name[pred]}")
    print()
```

### ✅ Deliverables for Exercise 4
- Screenshot of the classification report showing accuracy, precision, recall, and F1
- The confusion matrix image saved from your notebook
- Choose **one misclassified example** and write 3–4 sentences explaining why you think the model got it wrong
- Reflection: What class did the model struggle with most, and why might that be?

---

## Additional Resources

| Resource | Link |
|---|---|
| OpenAI API Documentation | https://platform.openai.com/docs |
| Hugging Face `transformers` docs | https://huggingface.co/docs/transformers |
| Hugging Face `datasets` docs | https://huggingface.co/docs/datasets |
| Tweet Eval Dataset | https://huggingface.co/datasets/cardiffnlp/tweet_eval |
| Scikit-learn Metrics Guide | https://scikit-learn.org/stable/modules/model_evaluation.html |
| DeepLearning.AI Short Courses | https://learn.deeplearning.ai |
| Prompt Engineering Guide | https://www.promptingguide.ai |

   


