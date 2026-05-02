import ollama 

response = ollama.chat(

    model='llama3.1',

    messages=[

        {"role": "user", "content": "What is the capital of France?"}

    ]

)

print(response['message']['content'])