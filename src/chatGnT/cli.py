# Write CLI for chatGnT predicttions
import argparse
from importlib.resources import files
from chatGnT.config import CFG
from chatGnT.models import predict
from chatGnT.models.transformer import TransformerModel_MultiTask, TransformerModel_SingleTask
import warnings

#TODO: add other flags (temperature, etc)
#TODO: add caching or model load only once per session

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
    category=UserWarning,
)


def main():
    parser = argparse.ArgumentParser(description="Generate recipes using trained models.")
    parser.add_argument(
        "--model-type", type=str, choices=["st", "mt"], default="st", help="Type of model to use (single-task or multi-task).")
    parser.add_argument(
        "--input", type=str, required=True, help="Input for the model (ingredient or cocktail name).")

    args = parser.parse_args()
    assets = files("chatGnT.assets")

    # Print recommended daily value alcohol warning
    print("chatG&T encourages you to drink responsibly!\n")
    print("The recommended daily distilled alcohol intake is 1.5 parts.\n")
    print("Your generated recipe is:\n")
    
    if args.model_type == "st":
        # Load ST model and vocab
        # Generate recipe using predict.generate_st_from_input
        model_st, vocab_st, config_st = predict.load_model_st(
            assets / "model_st.pt", TransformerModel_SingleTask)
        result = predict.generate_st_from_input(
            model_st,
            vocab_st,
            args.input,
            recipe_prefix_len=2
        )
        print(predict.format_prediction_st(result["tokens"]))
    
    elif args.model_type == "mt":
        # Load MT model and vocabs
        # Generate recipe using predict.generate_mt_from_input
        model_mt, vocab_mt_amt, vocab_mt_ingred, config_mt = predict.load_model_mt(
            assets / "model_mt.pt", TransformerModel_MultiTask)
        result = predict.generate_mt_from_input(
            model_mt,
            vocab_mt_amt,
            vocab_mt_ingred,
            args.input,
            recipe_prefix_len=2
        )
        print(predict.format_prediction_mt(result["tokens"]))
    print("\n")

if __name__ == "__main__":
    main()
