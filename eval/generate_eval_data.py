"""Generate sample documents and labeled evaluation dataset."""

import json
from pathlib import Path

# Sample documents for evaluation
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "filename": "machine_learning_intro.txt",
        "content": """Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed. It focuses on developing
algorithms that can identify patterns and make predictions based on input data.

Types of Machine Learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through interaction and rewards

Key Applications:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Predictive analytics
- Autonomous vehicles

The machine learning workflow typically involves:
1. Data collection and preparation
2. Feature engineering
3. Model selection
4. Training and validation
5. Testing and evaluation
6. Deployment and monitoring
"""
    },
    {
        "id": "doc2",
        "filename": "deep_learning_guide.txt",
        "content": """Deep Learning: Neural Networks at Scale

Deep learning is a specialized branch of machine learning that uses artificial
neural networks with multiple layers (hence "deep") to process data.

Neural Network Architecture:
- Input layer: Receives raw data
- Hidden layers: Process information through weighted connections
- Output layer: Produces predictions

Deep Learning Frameworks:
- TensorFlow and Keras for production systems
- PyTorch for research and experimentation
- JAX for high-performance computing

Popular Deep Learning Models:
1. Convolutional Neural Networks (CNNs) for image processing
2. Recurrent Neural Networks (RNNs) for sequential data
3. Transformers for natural language understanding
4. Autoencoders for dimensionality reduction

Deep learning has revolutionized:
- Computer vision with state-of-the-art accuracy
- Natural language processing with language models
- Speech recognition and synthesis
- Generative AI applications

Training deep networks requires:
- Large labeled datasets
- Significant computational resources (GPUs/TPUs)
- Careful hyperparameter tuning
- Regularization techniques to prevent overfitting
"""
    },
    {
        "id": "doc3",
        "filename": "nlp_essentials.txt",
        "content": """Natural Language Processing Essentials

Natural Language Processing (NLP) is the field of artificial intelligence that
focuses on enabling computers to understand, interpret, and generate human language.

Core NLP Tasks:
1. Tokenization - Breaking text into words or sentences
2. Part-of-speech tagging - Identifying word types
3. Named entity recognition - Extracting entities like names and locations
4. Sentiment analysis - Determining emotional tone
5. Machine translation - Converting between languages

NLP Techniques:
- Bag-of-words models for text representation
- Word embeddings (Word2Vec, GloVe, FastText)
- Sequence models with attention mechanisms
- Transformer-based models like BERT and GPT

Popular NLP Libraries:
- NLTK for classical NLP tasks
- spaCy for production NLP pipelines
- Transformers library for pre-trained models
- Hugging Face for accessing state-of-the-art models

Modern NLP Applications:
- Chatbots and conversational AI
- Question answering systems
- Text summarization
- Information extraction
- Machine translation
- Semantic search

Large Language Models:
Recent advances in transformer models have enabled:
- Few-shot learning with minimal examples
- Zero-shot task transfer
- In-context learning capabilities
"""
    },
    {
        "id": "doc4",
        "filename": "computer_vision.txt",
        "content": """Computer Vision: Processing Visual Information

Computer Vision is the field of artificial intelligence that deals with enabling
machines to interpret and understand the visual world using digital images and videos.

Computer Vision Tasks:
1. Image classification - Assigning labels to images
2. Object detection - Locating and identifying objects
3. Semantic segmentation - Pixel-level classification
4. Instance segmentation - Identifying individual objects
5. Image recognition - Identifying content in images
6. Face recognition - Identifying people in images

Key Computer Vision Techniques:
- Convolutional Neural Networks for feature extraction
- Feature pyramids for multi-scale analysis
- Region-based methods for object detection
- Attention mechanisms for focusing on relevant regions

Popular Computer Vision Models:
- ResNet for image classification
- YOLO for real-time object detection
- Mask R-CNN for instance segmentation
- Vision Transformers for image understanding

Applications:
- Autonomous driving and vehicle perception
- Medical imaging and diagnosis
- Security and surveillance systems
- Augmented reality and virtual reality
- Industrial inspection and quality control
- Retail and inventory management

Challenges in Computer Vision:
- Handling variations in lighting and scale
- Occlusion and partial visibility
- Domain shift between training and deployment
- Computational efficiency requirements
- Privacy concerns with facial recognition
"""
    },
    {
        "id": "doc5",
        "filename": "recommendation_systems.txt",
        "content": """Recommendation Systems: Personalized Content Delivery

Recommendation systems are algorithms designed to suggest relevant items to users
based on their preferences, behavior, and similarity to other users.

Types of Recommendation Approaches:
1. Collaborative filtering - Based on user behavior similarity
2. Content-based filtering - Based on item similarity
3. Hybrid approaches - Combining multiple methods
4. Knowledge-based recommendations - Using domain knowledge
5. Context-aware recommendations - Considering temporal/spatial context

Collaborative Filtering Methods:
- User-based: Find similar users and recommend their favorites
- Item-based: Recommend items similar to user preferences
- Matrix factorization: Decompose user-item interaction matrix
- Deep learning approaches: Neural collaborative filtering

Content-Based Filtering:
- Extract features from items
- Build user preference profiles
- Match profiles with new items
- Advantages: No cold start problem, Explainable
- Disadvantages: Limited novelty, Requires good features

Hybrid Systems:
- Combine strengths of multiple approaches
- Netflix model: Combines collaborative and content-based
- Amazon recommendations: Personalization + context

Evaluation Metrics:
- Precision and Recall at k (Prec@k, Rec@k)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Coverage and diversity
- User satisfaction metrics

Challenges:
- Cold start problem for new users/items
- Sparsity of user-item interaction data
- Computational scalability
- Filter bubbles and echo chambers
- Privacy preservation
- Real-time recommendation requirements
"""
    },
    {
        "id": "doc6",
        "filename": "data_science_workflow.txt",
        "content": """Data Science Workflow and Best Practices

Data science is an interdisciplinary field that combines statistics, computer science,
and domain expertise to extract insights from data.

Data Science Lifecycle:
1. Problem Definition - Understanding business goals
2. Data Collection - Gathering relevant data
3. Data Exploration - Understanding data characteristics
4. Data Cleaning - Handling missing values and outliers
5. Feature Engineering - Creating meaningful features
6. Model Selection - Choosing appropriate algorithms
7. Model Training - Fitting models to training data
8. Model Evaluation - Assessing performance
9. Hyperparameter Tuning - Optimizing model parameters
10. Model Deployment - Moving to production
11. Monitoring - Tracking performance over time

Data Preparation Techniques:
- Handling missing data: Imputation or removal
- Outlier detection and treatment
- Normalization and standardization
- Categorical encoding: One-hot, label encoding
- Feature scaling and transformation

Model Evaluation Metrics:
Classification:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix

Regression:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Cross-validation:
- K-fold cross-validation
- Stratified cross-validation
- Time series cross-validation

Tools and Technologies:
- Python with pandas, scikit-learn, numpy
- R for statistical analysis
- SQL for data querying
- Jupyter notebooks for exploration
- Git for version control
- Docker for reproducibility

Best Practices:
- Document your process and assumptions
- Validate data quality before analysis
- Use version control for code and data
- Implement proper train/test splits
- Avoid data leakage
- Monitor model performance in production
- A/B testing for online experiments
"""
    },
    {
        "id": "doc7",
        "filename": "feature_engineering.txt",
        "content": """Feature Engineering: The Art of Data Representation

Feature engineering is the process of selecting, transforming, and creating features
from raw data to improve machine learning model performance.

Types of Features:
1. Numerical features - Continuous or discrete values
2. Categorical features - Discrete categories
3. Temporal features - Time-based information
4. Textual features - Derived from text data
5. Spatial features - Location-based information

Feature Transformation Techniques:
- Scaling: Standardization, Normalization, MinMax scaling
- Log transformation: Handling skewed distributions
- Box-Cox transformation: Stabilizing variance
- Polynomial features: Creating non-linear relationships
- Interaction features: Combining multiple features

Handling Categorical Variables:
- One-hot encoding: Binary representation of categories
- Label encoding: Numeric representation of categories
- Target encoding: Using target statistics
- Binary encoding: Compact representation
- Frequency encoding: Based on occurrence frequency

Feature Selection Methods:
- Filter methods: Based on statistical properties
- Wrapper methods: Using model performance
- Embedded methods: Built into algorithms (L1, L2)
- Correlation analysis: Removing highly correlated features

Temporal Features:
- Lagged features: Previous time steps
- Rolling statistics: Mean, variance over windows
- Seasonal decomposition: Extracting seasonal patterns
- Time-based features: Day, month, year, hour

Domain-Specific Engineering:
For images:
- Edge detection, SIFT features
- Texture descriptors

For text:
- TF-IDF, Word embeddings
- N-grams, POS tags

For time series:
- Trend, seasonality, autocorrelation
- Frequency domain features

Common Pitfalls:
- Data leakage: Information from test set
- Overfitting to training data
- Ignoring feature distributions
- Not handling missing values consistently
- Creating correlated features
"""
    },
    {
        "id": "doc8",
        "filename": "model_deployment.txt",
        "content": """Model Deployment and Production Systems

Deploying machine learning models to production requires careful consideration of
scalability, reliability, and maintainability.

Deployment Architectures:
1. Batch processing - Process data periodically
2. Real-time serving - Immediate predictions
3. Stream processing - Handle continuous data
4. Edge deployment - Run on edge devices

Deployment Platforms:
- Cloud platforms: AWS SageMaker, Google Cloud AI, Azure ML
- On-premise servers: Kubernetes, Docker
- Edge devices: TensorFlow Lite, ONNX Runtime
- Serverless: AWS Lambda, Google Cloud Functions

Model Serving Frameworks:
- TensorFlow Serving for TensorFlow models
- TorchServe for PyTorch models
- MLflow for multi-framework serving
- KServe for Kubernetes-native serving
- BentoML for general model serving

Containerization:
- Docker for packaging models and dependencies
- Kubernetes for orchestration and scaling
- Docker Compose for local development

Monitoring and Maintenance:
- Track prediction latency
- Monitor model accuracy over time
- Detect data drift
- Detect model drift
- Set up alerts for anomalies

MLOps Best Practices:
- Version control for models and data
- Automated testing and validation
- Continuous integration/deployment (CI/CD)
- A/B testing for new models
- Rollback strategies
- Documentation and lineage tracking

Challenges:
- Model size and inference speed
- Handling concept drift
- Managing dependencies
- Ensuring reproducibility
- Scaling to handle traffic
- Model governance and compliance
- Resource efficiency
"""
    },
    {
        "id": "doc9",
        "filename": "reinforcement_learning.txt",
        "content": """Reinforcement Learning: Learning Through Interaction

Reinforcement learning is a machine learning paradigm where agents learn to make
decisions by interacting with an environment and receiving rewards or penalties.

Key Concepts:
- Agent: The decision-making entity
- Environment: The world the agent interacts with
- State: Current situation of the agent
- Action: Decision or move by the agent
- Reward: Feedback signal for the action
- Policy: Strategy for choosing actions

RL Algorithms:
1. Q-Learning - Learn value of state-action pairs
2. Policy Gradient - Directly optimize policy
3. Actor-Critic - Combine Q-learning and policy gradient
4. Deep Q-Network (DQN) - Q-learning with neural networks
5. Proximal Policy Optimization (PPO) - Stable policy optimization

Applications:
- Autonomous vehicles and robotics
- Game playing (AlphaGo, Chess)
- Resource optimization
- Portfolio management
- Recommendation systems with interactions
- Dialog systems and conversational AI

Exploration vs Exploitation:
- Exploration: Trying new actions to learn
- Exploitation: Using known best actions
- Epsilon-greedy strategy
- Upper Confidence Bound (UCB)
- Thompson sampling

Challenges:
- Sparse rewards: Hard to learn from infrequent feedback
- Long credit assignment: Delayed consequences
- Sample efficiency: Requiring many interactions
- Non-stationary environments
- Safe exploration: Avoiding dangerous states
- Scalability to complex environments

Popular RL Frameworks:
- OpenAI Gym for environment simulation
- Stable-Baselines3 for algorithm implementations
- RLlib for distributed RL training
- Dopamine for research
"""
    },
    {
        "id": "doc10",
        "filename": "ethics_ai.txt",
        "content": """Ethics in Artificial Intelligence

As AI systems become more prevalent, ethical considerations become increasingly important
for responsible development and deployment.

Key Ethical Principles:
1. Fairness - Treating all groups equally
2. Transparency - Understanding model decisions
3. Accountability - Responsibility for outcomes
4. Privacy - Protecting personal data
5. Safety - Ensuring systems behave as intended
6. Security - Protecting against adversarial attacks

Bias and Fairness:
- Algorithmic bias: Unfair outcomes for certain groups
- Data bias: Unrepresentative training data
- Evaluation bias: Unfair performance metrics
- Mitigation strategies: Balanced datasets, fairness metrics
- Intersectionality: Multiple overlapping identities

Privacy Considerations:
- Data collection and consent
- Differential privacy for statistical analysis
- Federated learning for decentralized training
- GDPR and privacy regulations
- Right to explanation
- Data minimization

Transparency and Explainability:
- Black-box vs interpretable models
- Feature importance analysis
- SHAP values and LIME
- Model cards and documentation
- Explaining predictions to users

Safety and Robustness:
- Adversarial examples: Fooling models with small perturbations
- Out-of-distribution detection
- Robust training techniques
- Uncertainty quantification
- Fail-safe mechanisms

Societal Impact:
- Displacement of workers
- Perpetuation of stereotypes
- Digital divide and access
- Environmental impact of training
- Power concentration
- Autonomous weapons

Responsible AI Development:
- Diverse and inclusive teams
- Ethics review processes
- Stakeholder engagement
- Continuous monitoring
- Regular audits
- Ethical guidelines and frameworks
"""
    }
]

# Labeled evaluation dataset: (query, relevant_doc_ids)
LABELED_QUERIES = [
    {
        "query": "machine learning algorithms and supervised learning",
        "relevant_docs": ["doc1", "doc6"],
        "irrelevant_docs": ["doc3", "doc4"]
    },
    {
        "query": "neural networks and deep learning training",
        "relevant_docs": ["doc2", "doc1"],
        "irrelevant_docs": ["doc3", "doc5"]
    },
    {
        "query": "natural language processing and text understanding",
        "relevant_docs": ["doc3", "doc2"],
        "irrelevant_docs": ["doc4", "doc5"]
    },
    {
        "query": "computer vision image recognition and object detection",
        "relevant_docs": ["doc4", "doc2"],
        "irrelevant_docs": ["doc3", "doc5"]
    },
    {
        "query": "recommendation systems and collaborative filtering",
        "relevant_docs": ["doc5"],
        "irrelevant_docs": ["doc1", "doc2"]
    },
    {
        "query": "data preparation and feature engineering",
        "relevant_docs": ["doc6", "doc7"],
        "irrelevant_docs": ["doc2", "doc4"]
    },
    {
        "query": "model evaluation metrics and validation",
        "relevant_docs": ["doc6", "doc1"],
        "irrelevant_docs": ["doc5", "doc10"]
    },
    {
        "query": "production deployment and model serving",
        "relevant_docs": ["doc8"],
        "irrelevant_docs": ["doc1", "doc3"]
    },
    {
        "query": "reinforcement learning and agent training",
        "relevant_docs": ["doc9"],
        "irrelevant_docs": ["doc3", "doc4"]
    },
    {
        "query": "AI ethics fairness and bias mitigation",
        "relevant_docs": ["doc10"],
        "irrelevant_docs": ["doc1", "doc2"]
    },
    {
        "query": "transformers BERT language models embeddings",
        "relevant_docs": ["doc3", "doc2"],
        "irrelevant_docs": ["doc4", "doc5"]
    },
    {
        "query": "CNN ResNet image classification",
        "relevant_docs": ["doc4", "doc2"],
        "irrelevant_docs": ["doc3", "doc5"]
    },
    {
        "query": "data science workflow exploratory analysis",
        "relevant_docs": ["doc6", "doc7"],
        "irrelevant_docs": ["doc2", "doc8"]
    },
    {
        "query": "PyTorch TensorFlow frameworks training",
        "relevant_docs": ["doc2", "doc1"],
        "irrelevant_docs": ["doc3", "doc5"]
    },
    {
        "query": "privacy differential privacy federated learning",
        "relevant_docs": ["doc10", "doc8"],
        "irrelevant_docs": ["doc1", "doc2"]
    },
]


def save_eval_data(output_dir: Path):
    """Save evaluation data to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save documents
    with open(output_dir / "documents.json", "w") as f:
        json.dump(SAMPLE_DOCUMENTS, f, indent=2)

    # Save queries
    with open(output_dir / "queries.json", "w") as f:
        json.dump(LABELED_QUERIES, f, indent=2)

    print(f"Saved {len(SAMPLE_DOCUMENTS)} documents to {output_dir / 'documents.json'}")
    print(f"Saved {len(LABELED_QUERIES)} labeled queries to {output_dir / 'queries.json'}")


if __name__ == "__main__":
    eval_dir = Path("eval")
    save_eval_data(eval_dir)
