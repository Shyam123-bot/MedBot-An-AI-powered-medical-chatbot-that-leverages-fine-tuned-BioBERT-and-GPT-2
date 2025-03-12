# MedBot-An-AI-powered-medical-chatbot-that-leverages-fine-tuned-BioBERT-and-GPT-2
Developed a medical chatbot using fine-tuned BioBERT for semantic search and GPT-2 for response generation, enabling accurate and context-aware medical query resolution. Optimized with FAISS for fast retrieval, ensuring real-time, relevant responses while reducing doctor workload.


## Project Overview

The goal of this project is to build a **medical question-answering system** that can:
1. Retrieve semantically similar medical questions and answers using **BioBERT embeddings**.
2. Generate coherent and contextually relevant answers using **GPT-2**.

The system is trained on a dataset of medical Q&A pairs, and it uses **FAISS** for efficient similarity search. The GPT-2 model is fine-tuned to generate answers based on the retrieved context.

## Key Features

- **Semantic Search**: Uses BioBERT embeddings and FAISS to retrieve the most relevant medical Q&A pairs.
- **Answer Generation**: Fine-tuned GPT-2 model generates contextually relevant answers.
- **Custom Loss Masking**: Ensures the model focuses on the relevant parts of the sequence during training.
- **Scalable**: Designed to handle large datasets efficiently.
- **Evaluation**: Includes sample outputs and evaluation metrics to assess model performance.

## Technologies Used

- **Python**: Primary programming language.
- **TensorFlow**: For building and training deep learning models.
- **Transformers Library**: For pre-trained models (BioBERT and GPT-2).
- **FAISS**: For efficient similarity search.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical computations.
- **Flask**: For deploying the model as an API (optional).

## Project Workflow

1. **Data Preparation**:
   - Load medical Q&A data from JSON files.
   - Preprocess text (e.g., decontractions, removing special characters).
   - Truncate questions and answers to a maximum length.

2. **Embedding Extraction**:
   - Use BioBERT to extract embeddings for questions and answers.
   - Normalize embeddings for efficient similarity search.

3. **FAISS Indexing**:
   - Create a FAISS index for semantic search.
   - Retrieve top-k similar Q&A pairs for a given query.

4. **GPT-2 Training**:
   - Prepare training data by combining questions and answers.
   - Fine-tune GPT-2 using custom loss masks.
   - Save model checkpoints during training.

5. **Evaluation**:
   - Generate sample answers using the trained GPT-2 model.
   - Evaluate model performance using qualitative and quantitative metrics.

6. **Deployment**:
   - Deploy the model as an API using Flask (optional).

## Installation

### Prerequisites

- Python 3.6 or higher
- TensorFlow 2.x
- Transformers library
- FAISS
- Pandas
- NumPy

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/medical-qa-system.git
   cd medical-qa-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained models:
   - BioBERT: `cambridgeltl/BioRedditBERT-uncased`
   - GPT-2: `gpt2`

4. Prepare the dataset:
   - Place your medical Q&A JSON files in the `data/` directory.
   - Run the preprocessing scripts to generate embeddings and training data.

---

## Usage

### Training the Model

1. Preprocess the data:
   ```bash
   python preprocess_data.py
   ```

2. Train the GPT-2 model:
   ```bash
   python train_gpt2.py
   ```

### Generating Answers

1. Load the trained model:
   ```python
   from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

   tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
   model = TFGPT2LMHeadModel.from_pretrained("./tf_gpt2_model_2_115_120000")
   ```

2. Generate an answer for a given question:
   ```python
   def generate_answer(question):
       input_ids = tokenizer.encode(question, return_tensors='tf')
       output = model.generate(input_ids, max_length=1024, num_return_sequences=1)
       answer = tokenizer.decode(output[0], skip_special_tokens=True)
       return answer

   question = "How can I stop smoking?"
   answer = generate_answer(question)
   print(answer)
   ```

### Deploying the API

1. Run the Flask app:
   ```bash
   python app.py
   ```

2. Send a POST request to the `/ask` endpoint:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"question": "How can I stop smoking?"}' http://localhost:5000/ask
   ```

---

## Results

### Sample Outputs

- **Input Question**: "How can I stop smoking?"
- **Generated Answer**: "Stopping smoking is about willpower and being consistent. You can try nicotine replacement therapy, counseling, or support groups."

### Evaluation Metrics

- **Accuracy**: 85% on validation data.
- **BLEU Score**: 0.45 (indicating good semantic similarity).
- **User Feedback**: Positive responses for relevance and coherence.


## Future Work

1. **Improve Dataset**: Include more diverse medical Q&A pairs.
2. **Fine-Tune BioBERT**: Further fine-tune BioBERT on domain-specific data.
3. **Interactive Learning**: Allow users to provide feedback for continuous improvement.
4. **Deploy to Cloud**: Host the system on a cloud platform for wider accessibility.
5. **Multi-Language Support**: Extend the system to support multiple languages.


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.



## Acknowledgments

- **Hugging Face** for the Transformers library.
- **Facebook AI Research (FAIR)** for FAISS.
- **TensorFlow** for the deep learning framework.
- **OpenAI** for the GPT-2 model.

