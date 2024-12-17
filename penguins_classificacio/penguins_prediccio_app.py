import pickle
from flask import Flask, jsonify, request
from penguins_service import predict_single

# Inicialitzar s'aplicació Flask
app = Flask('penguins-predict')

# Carrega models
models = {}
for name in ['logistic_regression', 'svm', 'decision_tree', 'knn']:
    with open(f'notebooks/{name}_model.pkl', 'rb') as f:
        dv, scaler, model = pickle.load(f)
        models[name] = (dv, scaler, model)

@app.route('/predict', methods=['POST'])
def predict():
    penguin = request.get_json()
    model_name = request.args.get('model')
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 400
    
    dv, scaler, model = models[model_name]
    predicted_species = predict_single(penguin, dv, scaler, model)
    result = {
        'predicted_species': predicted_species
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
