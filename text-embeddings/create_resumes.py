import openai
from sk import my_sk 
import time

import pandas as pd

def wait_for_assistant(thread, run):
    """
        Function to periodically check run status of AI assistant and print run time
    """

    # wait for assistant process prompt
    t0 = time.time()
    while run.status != 'completed':

        # retreive status of run (this might take a few seconds or more)
        run = client.beta.threads.runs.retrieve(
          thread_id=thread.id,
          run_id=run.id
        )

        # wait 0.5 seconds
        time.sleep(0.25)
    dt = time.time() - t0
    print("Elapsed time: " + str(dt) + " seconds")
    
    return run

# setup communication with API
client = openai.OpenAI(api_key=my_sk)

# define instruction string
intstructions_string = """ResumeGenerator is designed as an input-output system with minimal interaction. \
It focuses on creating fake resumes in a neutral and professional tone, covering specified sections: names, summary, professional experience, education, technical skills, certifications, awards, and honors. \ 
It creates fictional resumes based on the user's description. It never asks for more details and uses its best judgment to fill in any gaps in user requests. \
Providing straightforward, efficient service with little back-and-forth communication."""

# create ai assistant
assistant = client.beta.assistants.create(
    name="ResumeGenerator",
    instructions=intstructions_string,
    model="gpt-3.5-turbo"
)

# Generate resumes
def generateResume(user_message):
    """
        Function to generate fake resume based on user description.
    """
    
    # create thread (i.e. object that handles conversations between user and assistant)
    thread = client.beta.threads.create()
    
    # add a user message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )
    
    # send message to assistant to generate a response
    run = client.beta.threads.runs.create(
      thread_id=thread.id,
      assistant_id=assistant.id,
    )
    
    # wait for assistant process prompt
    run = wait_for_assistant(thread, run)
    
    # view messages added to thread
    messages = client.beta.threads.messages.list(
      thread_id=thread.id
    )
    
    return messages.data[0].content[0].text.value

# create fake resumes based on various data/AI roles

# define dataset names
dataset_name_list = ["train", "test"]

# define role descriptions to pass to ai assistant and number of resumes to generate for each
description_list = ["Data Scientist", "Data Engineer", "Machine Learning Engineer", "AI Consultant", "Data Entrepreneur", "Generate a random resume, you decide the roles and industry."]
count_list = [40,20,20,10,5,5]

for dataset_name in dataset_name_list:
    # initialize dict to store resume and role data
    resume_dict = {'resume':[], 'role':[]}
    
    if dataset_name == "test":
        count_list = [20,10,10,5,3,2]
    
    for i in range(len(description_list)):
        description = description_list[i]
        for j in range(count_list[i]):
            resume_dict['resume'].append(generateResume(description))
            if i==len(description_list):
                description = "Random"
            resume_dict['role'].append(description)


    # store resumes in dataframe
    df_resume = pd.DataFrame.from_dict(resume_dict)
    # save dataframe as csv
    df_resume.to_csv('resumes/resumes_'+dataset_name+'.csv', index=False)