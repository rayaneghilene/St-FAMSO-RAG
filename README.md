# St-FAMSO-RAG

Personalized study aid chatbot designed specifically for medical students at **FAMSO** to enhance their learning experience and mastery of course material. 
This system utilizes the principles of the RAG (Reterieval Augmented Generation) to generate responses based on the course materials.
It leverages the model ```llava:7b``` for text generation and the model ```all-MiniLM-L12-V2``` for the embedding extraction.
This Chatbot runs locally using Ollama and Streamlit. 

![image](https://github.com/rayaneghilene/St-FAMSO-RAG/assets/100053511/17cac149-652f-443c-baf0-a35ccb24cd0d)
## Requirements 
You need to install **Ollama** locally on your machine to run this code. [Link to install ollama](https://ollama.com/)

Once installed you need to import the Llava:7b model. You can do so using the following command on the terminal:
```
ollama pull llava:7b
```
**You can install these requirements for this project via:**
```
!pip install -r requirements.txt
```
## Usage
The code provided in this repo is straightforward, clone the repo using the following:
```
git clone https://github.com/rayaneghilene/St-FAMSO-RAG.git
```
Use the following command to run the app:

```
streamlit run app.py
```
This will open a new window in your browser, and you'll be able to chat with the model.


## References
**Llava model:**
```
@misc{liu2023visual,
      title={Visual Instruction Tuning}, 
      author={Haotian Liu and Chunyuan Li and Qingyang Wu and Yong Jae Lee},
      year={2023},
      eprint={2304.08485},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contributing
We welcome contributions from the community to enhance the chatbot and further support medical education. 
If you have ideas for features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## Support
For any questions, issues, or feedback, please contact me at rayane.ghilene@ensea.fr
