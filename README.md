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

To see a list of options run:

```sh
python phi2.py --help
```
