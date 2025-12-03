# LLM Model Trainings

This repository contains a collection of Jupyter notebooks demonstrating various stages of Large Language Model (LLM) training, from tokenization to fine-tuning.

## Notebooks

### 1. Tokenization
*   **[1_1_tokenization_impl_byte_pair_encoding.ipynb](1_1_tokenization_impl_byte_pair_encoding.ipynb)**: Implementation of Byte Pair Encoding (BPE), a subword tokenization method used in models like GPT and RoBERTa.

### 2. Embeddings
*   **[2_1_embedding_explore_position_embedding.ipynb](2_1_embedding_explore_position_embedding.ipynb)**: Explores learned position embeddings in GPT-2, analyzing how they encode relative token proximity.
*   **[2_2_embedding_train_token_embedding.ipynb](2_2_embedding_train_token_embedding.ipynb)**: Demonstrates training a Neural Language Model to learn Word Embeddings, based on Bengio et al. (2003).

### 3. Transformer Implementation
*   **[3_1_transformer_impl_attention_qkv.ipynb](3_1_transformer_impl_attention_qkv.ipynb)**: Implementation of the Scaled Dot-Product Attention mechanism (Query, Key, Value).
*   **[3_2_transformer_impl_transformer_block.ipynb](3_2_transformer_impl_transformer_block.ipynb)**: Builds a single Transformer Block using PyTorch, including Multi-Head Attention and MLP layers.
*   **[3_3_transformer_impl_multi_head_attention.ipynb](3_3_transformer_impl_multi_head_attention.ipynb)**: Implementation of Multi-Head Attention, extending the single-head attention mechanism.
*   **[3_4_transformer_impl_full_transformer_decoder.ipynb](3_4_transformer_impl_full_transformer_decoder.ipynb)**: Assembles the full Transformer Decoder architecture.

### 4. Pre-training
*   **[4_1_pre_training_weight_init.ipynb](4_1_pre_training_weight_init.ipynb)**: Covers pre-training the Transformer model, focusing on the importance of proper weight initialization.
*   **[4_2_pre_training_dropout.ipynb](4_2_pre_training_dropout.ipynb)**: Explores the use of Dropout during pre-training to prevent overfitting and encourage distributed representations.

### 5. Fine-tuning
*   **[5_1_fine_tuning_writing_style.ipynb](5_1_fine_tuning_writing_style.ipynb)**: Fine-tunes a pre-trained GPT-Neo model to learn specific writing styles from books (e.g., Romeo and Juliet, Alice's Adventures in Wonderland).
*   **[5_2_fine_tuning_freeze_weights.ipynb](5_2_fine_tuning_freeze_weights.ipynb)**: Demonstrates fine-tuning with frozen weights to reduce overfitting and computational cost, preserving low-level features while training higher-level ones.
*   **[5_3_fine_tuning_bert_classfication.ipynb](5_3_fine_tuning_bert_classfication.ipynb)**: Fine-tunes a BERT model for text classification on the IMDB dataset, demonstrating Parameter-Efficient Fine-Tuning (PEFT) by freezing the base model and training only the classifier.
*   **[5_4_fine_tuning_gradient_clipping_and_lr_scheduler.ipynb](5_4_fine_tuning_gradient_clipping_and_lr_scheduler.ipynb)**: Explores Gradient Clipping to prevent exploding gradients and Learning Rate Schedulers to dynamically adjust the learning rate during fine-tuning.

### 6. Instruction Tuning
*   **[6_1_instruction_tuning.ipynb](6_1_instruction_tuning.ipynb)**: Fine-tunes a GPT-2 model on the WebGLM-QA dataset to follow instructions and answer questions.