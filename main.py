import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def run_llama_command(prompt_text):

    # Define the command
    command = [
        './llama.cpp/llama-cli',
        '-m', 'llama.cpp/models/finance-chat.Q4_K_M.gguf',
        '--color',
        '-c', '512',
        '--temp', '0.8',            # Slightly higher temperature for more diversity
        '--top_p', '0.9',           # Nucleus sampling parameter
        '--top_k', '40',            # Top-k sampling parameter
        '--repeat_penalty', '1.1',
        '-n', '100',                # Increase the number of tokens to predict for more detailed responses
        '-p', prompt_text,  # Add a directive to the prompt
        '--log-disable'
    ]
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Get the generated output
    generated_output = result.stdout.strip()

    # Post-process the output to ensure sentence completion
    if '.' in generated_output:
        # Split the text by periods and get the last complete sentence
        sentences = generated_output.split('.')
        trimmed_output = '.'.join(sentences[:-1]) + '.'
    else:
        # If no period found, return the original generated output
        trimmed_output = generated_output

    # Print the output
    print("Generated Output:", trimmed_output)

    # Return the output
    return trimmed_output


@app.route('/talk', methods=['POST'])
def talk():
    # Get the JSON data from the request
    data = request.json

    # Ensure 'text' field is present in the JSON data
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" field in JSON data'}), 400

    # Get the prompt text
    prompt_text = data['text']

    # Run the llama command with the provided text
    try:
        output = run_llama_command(prompt_text)
        return jsonify({'output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
