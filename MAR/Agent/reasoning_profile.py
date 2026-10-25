reasoning_profile = [
    {
        "Name": "IO",
        "Description": "Call the model once and output the final answer directly. Use 1 round and 1 agent only.",
    },
    {
        "Name": "CoT",
        "Description": "Use 1 agent, called repeatedly, rounds controls reasoning steps.",
    },
    {
        "Name": "Reflection",
        "Description": "Use 1 agent for rounds >= 2; each later round decides whether to retain or revise its previous answer.",
    },
    {
        "Name": "Chain",
        "Description": "Multiple agents run sequentially; rounds controls each agent’s iterations before its final result passes downstream.",
    },
    {
        "Name": "FullConnected",
        "Description": "Multiple agents collaborate in a fully connected manner, with rounds >= 2.",
    },
    {
        "Name": "Debate",
        "Description": "Multiple agents collaborate by proposing viewpoints, reviewing, and revising viewpoints, with rounds >= 2.",
    },
]
