
import emlearn_linreg
import array

# Create model: 4 features, alpha=0.1, l1_ratio=0.5, lr=0.01
model = emlearn_linreg.new(4, 0.1, 0.5, 0.01)

# Training data (float32 arrays)
X = array.array('f', [1,2,3,4, 2,3,4,5])  # flattened
y = array.array('f', [10, 15])

# Train
model.train(X, y, 1000, 1e-6)

# Predict
prediction = model.predict(array.array('f', [1,2,3,4]))
print(prediction)
