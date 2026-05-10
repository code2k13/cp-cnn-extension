import array
import gc
import math
import cnn_helper

INPUT_SIZE = 900 
inp = array.array("f", [0.0] * INPUT_SIZE)

def simple_softmax(scores):
    shift_scores = [math.exp(s - max(scores)) for s in scores]
    sum_exp = sum(shift_scores)
    return [s / sum_exp for s in shift_scores]

def process_sample(file_path):
    try:
        with open(file_path, "rb") as f:
            f.readinto(inp)
            
        cnn_helper.load_model("model.bin")
        out = cnn_helper.perform_inference(inp)
        cnn_helper.unload_model()
        gc.collect()
        
        return out
    except Exception as e:
        print("Error:", e)
        return None

for i in range(20):
    filename = "mnist_samples/sample_{:02d}.bin".format(i)
    raw_output = process_sample(filename)
    
    if raw_output:
        probs = simple_softmax(raw_output)
        prediction = probs.index(max(probs))
        confidence = max(probs) * 100
        
        print("File: {} -> Pred: {} ({:.1f}%)".format(filename, prediction, confidence))
    
    gc.collect()
