import json
from step1.serialization import sender_user_content

path = "data/step1/episodes/episode_inputs.json"
episode_id = "<你的 episode_id>"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

info = data[episode_id]
prompt = sender_user_content(
    info["task_description"],
    info["initial_observation"],
)
print(prompt)
