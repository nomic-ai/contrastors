from contrastors.models.huggingface import NomicBertForPreTraining, NomicBertConfig 
from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--biencoder", action="store_true")
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    if args.biencoder:
        config = BiEncoderConfig.from_pretrained(args.ckpt_path)
        model = BiEncoder.from_pretrained(args.ckpt_path, config=config)
        model = model.trunk
    else:
        config = NomicBertConfig.from_pretrained(args.ckpt_path)
        model = NomicBertForPreTraining.from_pretrained(args.ckpt_path, config=config)
    model.push_to_hub(args.model_name, private=args.private)