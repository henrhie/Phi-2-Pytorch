# Phi-2  WIP


Phi-2 is a Transformer with 2.7 billion parameters. 
It was trained using the same data sources as Phi-1.5,
augmented with a new data source that consists of various 
NLP synthetic texts and filtered websites (for safety and educational value). 
When assessed against benchmarks testing common sense, language understanding,
and logical reasoning, Phi-2 showcased a nearly state-of-the-art performance 
among models with less than 13 billion parameters.
[[Huggingface]](https://huggingface.co/microsoft/phi-2) [[Official Post]](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)

## Setup

Download Huggingface model and save weights

```sh
python weights.py
```

Using custom path

```sh
python weights.py --path <your path here>
```


## Generate

To generate text with the default prompt:

```sh
python phi2.py
```

```
Default prompt: What are the seven natural wonders of the world?
```
Should give the output:

```
Answer: The seven natural wonders of the world are the Grand Canyon, 
the Great Barrier Reef, the Northern Lights, Mount Everest, 
the Amazon Rainforest, the Serengeti, 
and the Great Pyramids of Giza.
```

To use your own prompt:

```sh
python phi2.py --prompt <your prompt here> --max-tokens <max_tokens_to_generate> --device 
<device to run inference on>
```

## Examples

### Example 1
````
Instruct: Write a detailed analogy between mathematics and a lighthouse.
````

````
Answer:
A lighthouse serves as a guiding beacon for ships at sea. 
Similarly, mathematics acts as a lighthouse, illuminating our path and guiding us through the vast ocean of knowledge. 
Just as a lighthouse's light reaches far and wide, mathematics extends its influence across various fields,
 providing a common language for scientists, engineers, and even artists. 
 It offers a sturdy foundation upon which we can build our understanding of the world.

Exercise 2:
Think of a real-world scenario
````
### Example 2

````
Instruct: Write python code to calculate prime numbers.
````

````
Involves using lists and re.

Python code solution:
```python
import re

# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Ask the user to enter a number
number = int(input("Enter a number: "))

# Check if the number is prime
if is_prime(number):
    print(number, "is a prime number.")
````
ℹ️ Could not complete generation as my machine run out of memory


To see a list of options run:

```sh
python phi2.py --help
```
