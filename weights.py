from transformers import AutoModelForCausalLM
import torch
import argparse
from pathlib import Path


def get_weights():
    parser = argparse.ArgumentParser(description="Retrieve Huggingface Phi-2 weights")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    parser.add_argument("--path",
                        type=str,
                        default="weights",
                        help="The path to save model weights.", )
    args = parser.parse_args()
    path = Path(args.path)
    print("creating path....")
    path.mkdir(parents=True, exist_ok=True)
    print("saving weights....")
    torch.save(model.state_dict(), str(path / 'phi2-weights.pt'))
    print("weights saved successfully in " + str(path / 'phi2-weights.pt'))


if __name__ == '__main__':
    get_weights()
