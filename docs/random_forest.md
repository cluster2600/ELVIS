# ELVIS Random Forest Documentation

## Overview

The Random Forest component of ELVIS (Enhanced Leveraged Virtual Investment System) is a powerful machine learning model used for cryptocurrency trading predictions. It leverages TensorFlow Decision Forests (TFDF) to implement an ensemble of decision trees that can capture complex patterns in market data and make robust trading decisions.

## Architecture

The Random Forest implementation in ELVIS consists of several key components:

1. **RandomForestModel Class**: A concrete implementation of the BaseModel interface that provides the core functionality
2. **TensorFlow Decision Forests**: The underlying library that implements the Random Forest algorithm
3. **Ensemble Integration**: Integration with other models (Neural Networks and MLX) for ensemble predictions
4. **Feature Engineering**: Preprocessing of market data to create features for the model

## Model Implementation

### Core Components

The Random Forest model is implemented in `core/models/random_forest_model.py` and inherits from the `BaseModel` abstract class defined in `core/models/base_model.py`. This ensures that the Random Forest model follows the same interface as other models in the system.

```python
class RandomForestModel(BaseModel):
    """
    Random Forest model for trading.
    Uses TensorFlow Decision Forests for implementation.
    """
```

### Key Features

1. **TensorFlow Decision Forests Integration**: Uses the TFDF library for efficient implementation of Random Forest algorithms
2. **Feature Importance Analysis**: Provides insights into which market features are most important for predictions
3. **Model Persistence**: Saves and loads models to/from disk for deployment
4. **Error Handling**: Robust error handling to ensure the system continues to function even if the model fails
5. **Hyperparameter Configuration**: Configurable parameters for model tuning

## Training Process

The Random Forest model training process follows these steps:

1. **Data Preparation**: Features and labels are prepared from market data
2. **Model Initialization**: The Random Forest model is initialized with specified hyperparameters
3. **Training**: The model is trained on the prepared data
4. **Evaluation**: The model's performance is evaluated on a test set
5. **Model Saving**: The trained model is saved for later use

### Training Code Example

```python
# Initialize the model
model = RandomForestModel(
    logger=logger,
    num_trees=100,
    max_depth=20,
    min_examples=5
)

# Train the model
model.train(X_train, y_train)

# Evaluate the model
metrics = model.evaluate(X_test, y_test)

# Get feature importance
importance = model.get_feature_importance()
```

## Hyperparameters

The Random Forest model can be configured with the following hyperparameters:

- **num_trees**: The number of decision trees in the forest (default: 100)
- **max_depth**: The maximum depth of each decision tree (default: 20)
- **min_examples**: The minimum number of examples required to split a node (default: 5)

These parameters can be adjusted to optimize the model's performance for different market conditions.

## Feature Engineering

The Random Forest model uses a set of predefined features for prediction:

```python
REQUIRED_FEATURES = [
    "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
    "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb", "sma_bb",
    "upper_bb", "news_sentiment", "social_feature", "adx", "rsi", "order_book_depth", "volume"
]
```

These features include:
- Price data (current price, future price)
- Technical indicators (SMA, MACD, Bollinger Bands, ATR, ADX, RSI)
- Volume metrics (volume, volume moving average)
- Order book data (depth)
- Sentiment data (news sentiment, social features)
- Trading metrics (order amount, filled orders, total)

## Ensemble Integration

The Random Forest model is part of an ensemble that includes:

1. **Random Forest (YDF)**: For capturing complex non-linear relationships
2. **Neural Network**: For learning deep patterns in the data
3. **MLX Model**: For incorporating natural language understanding

The ensemble combines predictions from all three models to make a final trading decision:

```python
def ensemble_predict(ydf_model, nn_model, features: dict, mlx_url: str = "http://localhost:1234/v1/completions"):
    """Predict trading decision using ensemble of models."""
    # YDF (Random Forest) prediction
    ydf_input = pd.DataFrame([features])
    ydf_pred = ydf_model.predict(ydf_input)
    
    # Neural Network prediction
    nn_input = np.array([[features[col] for col in REQUIRED_FEATURES]], dtype=np.float32)
    nn_pred = nn_model.predict({"input": nn_input})
    
    # MLX prediction
    prompt = f"Market features: {', '.join(f'{k}: {v}' for k, v in features.items())}. Recommend: BUY, SELL, or HOLD."
    mlx_output = mlx_generate(prompt, mlx_url)
    mlx_decision = parse_mlx_decision(mlx_output)
    
    # Ensemble averaging
    avg_probs = np.mean([ydf_probs, nn_probs, mlx_probs], axis=0)
    decision_idx = np.argmax(avg_probs)
    decision = CLASSES[decision_idx]
    confidence = avg_probs[decision_idx]
    
    return decision, confidence
```

## Model Deployment

The trained Random Forest model can be deployed in several ways:

1. **Direct Integration**: The model can be loaded and used directly in the trading system
2. **CoreML Conversion**: The model can be converted to CoreML format for deployment on Apple devices
3. **Ensemble Deployment**: The model can be deployed as part of the ensemble for more robust predictions

### Loading a Trained Model

```python
# Load the model
model = RandomForestModel.load("path/to/model.ydf")

# Make predictions
predictions = model.predict(X)
```

## Performance Evaluation

The Random Forest model's performance is evaluated using several metrics:

- **Accuracy**: The proportion of correct predictions
- **Loss**: The model's loss function value
- **Feature Importance**: The importance of each feature in making predictions

These metrics help in understanding the model's performance and identifying areas for improvement.

## Advantages and Limitations

### Advantages

1. **Robustness**: Random Forests are less prone to overfitting than individual decision trees
2. **Feature Importance**: Provides insights into which features are most important for predictions
3. **Handles Non-linearity**: Can capture complex non-linear relationships in the data
4. **Handles Missing Values**: Can handle missing values in the data
5. **Parallelizable**: Training can be parallelized for faster execution

### Limitations

1. **Computational Cost**: Training and inference can be computationally expensive for large datasets
2. **Memory Usage**: Requires more memory than simpler models
3. **Interpretability**: While individual trees are interpretable, the ensemble is less so
4. **Hyperparameter Tuning**: Requires careful tuning of hyperparameters for optimal performance

## Best Practices

1. **Feature Selection**: Use feature importance analysis to select the most relevant features
2. **Hyperparameter Tuning**: Experiment with different hyperparameter values to find the optimal configuration
3. **Cross-Validation**: Use cross-validation to ensure the model generalizes well to unseen data
4. **Ensemble Integration**: Use the Random Forest model as part of an ensemble for more robust predictions
5. **Regular Retraining**: Retrain the model periodically to adapt to changing market conditions

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce the number of trees or the maximum depth
   - Use a subset of features

2. **Poor Performance**
   - Check feature engineering
   - Adjust hyperparameters
   - Consider using a different algorithm

3. **Slow Training**
   - Reduce the number of trees
   - Use a subset of the data for training
   - Use parallel processing

### Debugging Tips

1. **Check Feature Importance**
   - Use the `get_feature_importance()` method to identify important features
   - Remove or adjust less important features

2. **Validate Data**
   - Check for missing values
   - Ensure features are properly scaled
   - Verify that the target variable is correctly defined

3. **Monitor Training Progress**
   - Log training metrics
   - Use early stopping to prevent overfitting

## Future Improvements

1. **Advanced Feature Engineering**
   - Implement more sophisticated feature engineering techniques
   - Add more market-specific features

2. **Hyperparameter Optimization**
   - Implement automated hyperparameter optimization
   - Use Bayesian optimization for more efficient tuning

3. **Online Learning**
   - Implement online learning to update the model with new data
   - Use incremental learning techniques

4. **Explainability**
   - Implement SHAP values for better model interpretability
   - Add visualization tools for model explanations

5. **Integration with Other Models**
   - Explore integration with other machine learning models
   - Implement more sophisticated ensemble techniques 