# Statement  

Project LM2  
Use an LLM for few-shot learning. The task is as follows:  
In many instances, it is important to detect and remove irrelevant sentences from text. One
example is that of generating graphs from text, where the presence of irrelevant sentences
might complete distort the generated graphs. An application is that of generating business
process models (graphs, petrinets, BPMN) from textual descriptions. The generated model
should be a faithful reproduction of the textual description of the business process, without
any irrelevant sentences, such as those providing general information about a company.
However, there is a lack of training data, i.e. annotated with irrelevant sentences, to train
supervised learning models. One solution is that of few-shot learning.  
The aim is to use an LLM to learn about irrelevant sentences from sample texts. These
irrelevant sentences can be indicated by various manners, .e.g. prompts to the LLMs or by
annotating a handful of them.
Then, the LLM should be able to automatically detect new irrelevant sentences.
Project steps:

- Choose an LLM, preferably open-source of reasonable size, e.g. Falcon7B (available
from HuggingFace).
- Prompt your LLM for the task described above using example sentences that will be
provided to you.
- Test on new sentences: report precision, recall, and F1 scores.
- Fine-tune your LLM if needed, and test again.
