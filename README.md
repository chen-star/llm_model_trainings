# 🚀 LLM Model Trainings

Welcome to the **LLM Model Trainings** repository! 🎓✨

This collection of Jupyter notebooks takes you on a journey through the fascinating stages of training Large Language Models (LLMs), from the basics of tokenization all the way to advanced fine-tuning techniques.

## 📚 Notebooks

Here is the roadmap of our learning adventure:

| Stage                       | 📓 Notebook                                                                                 | 📝 Description                                                                   |
| :-------------------------- | :----------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| **1. Tokenization** 🧩       | [**Byte Pair Encoding**](1_1_tokenization_impl_byte_pair_encoding.ipynb)                   | Implement BPE, the subword tokenization method used in GPT and RoBERTa.         |
| **2. Embeddings** 🗺️         | [**Position Embeddings**](2_1_embedding_explore_position_embedding.ipynb)                  | Explore how GPT-2 learns and encodes relative token proximity.                  |
|                             | [**Train Token Embeddings**](2_2_embedding_train_token_embedding.ipynb)                    | Train a Neural Language Model to learn Word Embeddings (Bengio et al., 2003).   |
| **3. Transformer** 🤖        | [**Attention (Q, K, V)**](3_1_transformer_impl_attention_qkv.ipynb)                        | Implement the core Scaled Dot-Product Attention mechanism.                      |
|                             | [**Transformer Block**](3_2_transformer_impl_transformer_block.ipynb)                      | Build a single Transformer Block with Multi-Head Attention and MLP layers.      |
|                             | [**Multi-Head Attention**](3_3_transformer_impl_multi_head_attention.ipynb)                | Extend attention to multiple heads for richer representation.                   |
|                             | [**Full Decoder**](3_4_transformer_impl_full_transformer_decoder.ipynb)                    | Assemble the complete Transformer Decoder architecture.                         |
| **4. Pre-training** 🏋️       | [**Weight Initialization**](4_1_pre_training_weight_init.ipynb)                            | Learn the importance of proper weight initialization for stability.             |
|                             | [**Dropout**](4_2_pre_training_dropout.ipynb)                                              | Apply dropout to prevent overfitting and encourage distributed representations. |
| **5. Fine-tuning** 🎨        | [**Writing Style**](5_1_fine_tuning_writing_style.ipynb)                                   | Teach GPT-Neo to mimic specific authors (e.g., Shakespeare, Carroll).           |
|                             | [**Freeze Weights**](5_2_fine_tuning_freeze_weights.ipynb)                                 | Fine-tune efficiently by freezing lower layers to preserve features.            |
|                             | [**BERT Classification**](5_3_fine_tuning_bert_classfication.ipynb)                        | PEFT demonstration: Fine-tune BERT for IMDB text classification.                |
|                             | [**Grad Clipping & Schedulers**](5_4_fine_tuning_gradient_clipping_and_lr_scheduler.ipynb) | Master training stability with gradient clipping and dynamic learning rates.    |
| **6. Instruction Tuning** 🛠️ | [**Instruction Tuning**](6_1_instruction_tuning.ipynb)                                     | Fine-tune GPT-2 to follow instructions using the WebGLM-QA dataset.             |
| **7. Evaluation** 📊         | [**Perplexity**](7_1_evaluation_quantitative_perplexity.ipynb)                             | Evaluate model performance quantitatively using perplexity scores.              |
|                             | [**HellaSwag**](7_2_evaluation_quantitative_HellaSwag.ipynb)                               | Benchmark commonsense reasoning using HellaSwag (Zephyr-7b-alpha vs GPT-2).     |
|                             | [**KL Divergence**](7_3_evaluation_quantitative_KL_divergence.ipynb)                       | Measure distribution difference using KL Divergence.                            |
