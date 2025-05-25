from neuronpedia.sample_data import GEMMA2_2B_IT_DINOSAURS_VECTOR
from neuronpedia.np_vector import NPVector
import os, json
import os
from dotenv import load_dotenv

load_dotenv()  # Looks for .env file in current directory

api_key = os.getenv("NEURONPEDIA_API_KEY")


# upload the custom vector
np_vector = NPVector.new(
    label="dinosaurs",
    model_id="gemma-2-2b-it",
    layer_num=20,
    hook_type="hook_resid_pre",
    vector=GEMMA2_2B_IT_DINOSAURS_VECTOR,
    default_steer_strength=44,
)

# steer with it
responseJson = np_vector.steer_chat(
    steered_chat_messages=[{"role": "user", "content": "Write a one sentence story."}]
)

print(json.dumps(responseJson, indent=2))
print("UI Steering at: " + responseJson["shareUrl"])
