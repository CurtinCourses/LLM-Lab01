
<h2 align="center">Lab 01 - Introduction to Generative AI</h2>


## Objectives:  
* Learn about using different types of prompting styles
* Use OpenAI's API
* Calling Hugging Face and its APIs
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

   


