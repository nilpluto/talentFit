from ollama import chat

response = chat(
    model="qwen3:1.7b",
    messages=[
        {
            "role": "user",
            "content": "Who is Modi? Answer in 2 lines."
        }
    ]
)

print(response["message"]["content"])
