import sys
import json
import ydf
import pandas as pd

def main():
    model_path = "model_rf.ydf"
    try:
        model = ydf.from_tensorflow_decision_forests(model_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load YDF model: {e}"}))
        sys.exit(1)

    try:
        input_json = sys.stdin.read()
        features = json.loads(input_json)
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        probabilities = list(prediction.get('probabilities', {}).values())
        output = {"probabilities": probabilities}
        print(json.dumps(output))
    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {e}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
