import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_eval_accuracy(log_dir, tag="Eval/Accuracy"):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        raise ValueError(f"Tag {tag} not found in {log_dir}")
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    return steps, vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run1', required=True, help="Path to first run (e.g., log/runs/grirl_0)")
    parser.add_argument('--run2', required=True, help="Path to second run (e.g., log/runs/grirl_1)")
    args = parser.parse_args()

    steps1, acc1 = load_eval_accuracy(args.run1)
    steps2, acc2 = load_eval_accuracy(args.run2)

    min_len = min(len(acc1), len(acc2))
    acc1 = acc1[:min_len]
    acc2 = acc2[:min_len]
    steps = steps1[:min_len]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, acc1, label=f'Run 1 ({os.path.basename(args.run1)})', linewidth=2)
    plt.plot(steps, acc2, label=f'Run 2 ({os.path.basename(args.run2)})', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Eval Accuracy')
    plt.title('Eval Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/eval_accuracy_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
