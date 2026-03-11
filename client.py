import requests
import json
import sys

# The address of YOUR local proxy server
PROXY_URL = "http://127.0.0.1:8000/proxy/chat"

def start_chat():
    # This list stores the whole convo history
    messages = []
    
    print("--- Local Proxy Chat Interface ---")
    print("Type 'exit' or 'quit' to stop.")
    print("----------------------------------\n")

    while True:
        # 1. Get user input
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Catch ya later!")
            break

        # 2. Add your message to the history
        messages.append({"role": "user", "content": user_input})

        # 3. Prepare the payload exactly how the proxy expects it
        payload = {
            "messages": messages,
            "model": "dolphinserver:24B", # Keeping it consistent with your images
            "template": "creative"
        }

        try:
            # 4. Send the request to your LOCAL proxy
            response = requests.post(PROXY_URL, json=payload)
            
            if response.status_code == 200:
                # Expect structured JSON: {"content": "..."}
                assistant_message = None
                try:
                    data = response.json()
                    if isinstance(data, dict) and "content" in data:
                        assistant_message = data.get("content")
                    else:
                        # Fallbacks: full message formats
                        if isinstance(data, dict) and "choices" in data:
                            try:
                                assistant_message = data["choices"][0]["message"]["content"]
                            except Exception:
                                assistant_message = None
                        else:
                            assistant_message = str(data)
                except ValueError:
                    assistant_message = response.text

                if not assistant_message:
                    assistant_message = "No content found in response."

                # Print only the structured assistant content
                print(f"\nDolphin: {assistant_message}\n")
                messages.append({"role": "assistant", "content": assistant_message})
            else:
                print(f"Error from Proxy: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the proxy. Is 'proxy-server.py' running?")
            break

if __name__ == "__main__":
    start_chat()