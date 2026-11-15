
import argparse
from demo import demo
from training import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "demo"], default="demo")
    parser.add_argument("--policy", choices=["random", "heuristic", "ppo", "human"], default="random")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.steps, args.seed)
    else:
        demo(args.policy, args.seed)

if __name__ == "__main__":
    main()
