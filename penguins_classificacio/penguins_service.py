def predict_single(penguin, dv, scaler, model):
    data_dict = dv.transform([penguin])
    data_scaled = scaler.transform(data_dict)
    prediction = model.predict(data_scaled)
    return prediction[0]
