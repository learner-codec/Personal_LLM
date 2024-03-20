import json
import random
import sys

# Check if the correct number of arguments are passed
if len(sys.argv) != 3:
    print("Usage: python convert_dataset.py <input_json_file> <output_txt_file>")
    sys.exit(1)

# Filenames from arguments
input_file = sys.argv[1]
output_txt_file = sys.argv[2]

# Load the JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

# Shuffle data
random.shuffle(data)

# Concatenate the records into a single string in the desired format
text = ''
dtext = ''
q = 0
basic_text = ''
for record in data:
    print(q, end='\r')
    q = q + 1
    if 'instruction' not in record:
        continue
    
    period = "." if random.randint(0, 1) == 0 else ""
    instruction = record['instruction']
    
    # Random modifications
    if random.randint(0, 1) == 0:
        instruction = instruction[:-1]
    if random.randint(0, 1) == 0:
        instruction = instruction[0].strip().lower() + instruction[1:]
    
    # Decide the format randomly
    if random.randint(0, 1) == 0:
        text = f"user: {instruction} {record['input']}{period}\nbot: {record['output']}\n".replace('\n\n', '\n').replace(" . ", '').replace(" .\n", '\n')
    else:
        text = f"user: {record['input']}. {instruction}\nbot: {record['output']}\n".replace('\n\n', '\n').replace(" . ", '').replace(" .\n", '\n')
    
    d = {}
    d['text'] = text
    dtext = dtext + '\n' + json.dumps(d)
    basic_text = basic_text + text

# Write to the output file
with open(output_txt_file, 'w') as f:
    f.write(basic_text)
