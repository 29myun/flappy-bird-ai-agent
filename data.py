import json
import os

def save(model_type, record, total_games_played, model_folder):
    file_name = model_type + "_model.json"
    file_path = os.path.join(model_folder, file_name)
    
    data = {"model_type": model_type, "record": record, "total_games_played": total_games_played}
    
    if os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(data, f)
    else:
        raise RuntimeWarning("File path not found")

def load(model_type, model_folder="./saved_models_data"):
    file_name = model_type + "_model.json"
    file_path = os.path.join(model_folder, file_name)
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise RuntimeError("Model not found")

        