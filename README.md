<h1>Breast Cancer Classification Using Neural Networks</h1>

<h2>📌 Objective</h2>
<p>Build a deep learning model using Neural Networks to classify whether a breast tumor is malignant or benign based on medical features.</p>

<hr>

<h2>🛑 Problem Statement</h2>
<p>Breast cancer is one of the most common cancers affecting women worldwide. Early and accurate diagnosis can significantly increase survival rates.<br>
Traditional diagnosis methods are time-consuming and prone to human error. A machine learning solution can assist doctors in making faster and more reliable predictions.</p>

<hr>

<h2>📊 Dataset</h2>
<ul>
  <li><b>Source</b>: <a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data" target="_blank">Kaggle - Breast Cancer Wisconsin (Diagnostic) Dataset</a></li>
  <li><b>Details</b>:
    <ul>
      <li>569 samples with 30 features each.</li>
      <li>Attributes include radius, texture, smoothness, compactness, symmetry, and fractal dimension.</li>
      <li>Target:
        <ul>
          <li><code>0</code> → Malignant (cancerous)</li>
          <li><code>1</code> → Benign (non-cancerous)</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>

<hr>

<h2>⚙️ Methodology</h2>

<h3>1. Data Preprocessing</h3>
<ul>
  <li>Load and explore the dataset.</li>
  <li>Handle missing values (if any).</li>
  <li>Normalize/standardize the feature values (important for Neural Networks).</li>
  <li>Encode the target variable if needed.</li>
  <li>Train-test split (typically 80%-20%).</li>
</ul>

<h3>2. Exploratory Data Analysis (EDA)</h3>
<ul>
  <li>Visualize feature distributions.</li>
  <li>Analyze correlation between features and target variable.</li>
  <li>Identify and remove redundant features if necessary.</li>
</ul>

<h3>3. Model Building - Neural Network</h3>
<ul>
  <li>Design a simple feedforward neural network:
    <ul>
      <li>Input Layer: 30 neurons (one for each feature).</li>
      <li>Hidden Layers: 1 layers with ReLU activation.</li>
      <li>Output Layer: 2 neuron with Sigmoid activation (binary classification).</li>
    </ul>
  </li>
  <li>Compile the model using:
    <ul>
      <li>Loss function: Sparse Categorical Crossentropy</li>
      <li>Optimizer: Adam</li>
      <li>Metrics: Accuracy </li>
    </ul>
  </li>
  <li>Train the model with early stopping to prevent overfitting.</li>
</ul>

<h3>4. Model Evaluation</h3>
<ul>
  <li>Evaluate model on the test set using:
    <ul>
      <li>Accuracy</li>
      <li>Precision</li>
      <li>Recall</li>
      <li>F1-Score</li>
      <li>ROC-AUC Score</li>
      <li>Confusion Matrix</li>
    </ul>
  </li>
</ul>

<h3>5. Optimization</h3>
<ul>
  <li>Hyperparameter tuning:
    <ul>
      <li>Number of hidden layers and neurons.</li>
      <li>Learning rate and batch size.</li>
      <li>Activation functions.</li>
    </ul>
  </li>
  <li>Use Dropout regularization if overfitting occurs.</li>
</ul>

<h3>6. Deployment (optional)</h3>
<ul>
  <li>Create a simple Streamlit or Flask app for real-time predictions.</li>
  <li>Allow users to input tumor characteristics and predict malignancy.</li>
</ul>

<hr>

<h2>🛠️ Technologies Used</h2>
<ul>
  <li>Python</li>
  <li>Libraries:
    <ul>
      <li>TensorFlow / Keras</li>
      <li>NumPy, Pandas</li>
      <li>Matplotlib, Seaborn</li>
      <li>Scikit-learn (for preprocessing and evaluation)</li>
    </ul>
  </li>
  <li>Jupyter Notebook / Google Colab</li>
  <li>(Optional) Streamlit / Flask for deployment</li>
</ul>

<hr>

<h2>🚧 Challenges</h2>
<ul>
  <li>Preventing overfitting due to small dataset size.</li>
  <li>Ensuring high recall to minimize false negatives (missing cancer cases).</li>
  <li>Choosing optimal network architecture and hyperparameters.</li>
</ul>

<hr>

<h2>✅ Results</h2>
<ul>
  <li>Neural network achieved:
    <ul>
      <li>Accuracy above 95.8% on test data.</li>
      <li>Loss 12.25%, crucial for detecting malignant tumors.</li>
    </ul>
  </li>
</ul>

<hr>

<h2>📝 Conclusion</h2>
<p>This project demonstrates that a properly designed Neural Network can classify breast cancer cases with very high accuracy and recall.<br>
Such models can be integrated into diagnostic systems to support early and accurate breast cancer detection, saving lives through faster interventions.</p>

<hr>

<h2>🔮 Future Work</h2>
<ul>
  <li>Experiment with deeper and more complex neural network architectures.</li>
  <li>Test with larger, real-world clinical datasets.</li>
  <li>Apply Transfer Learning techniques for improved performance.</li>
  <li>Deploy the model for real-world clinical decision support.</li>
</ul>
