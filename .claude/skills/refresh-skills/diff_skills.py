"""Compare two current_player_skills*.json files.

Usage (from the repo root):
    python .claude/skills/refresh-skills/diff_skills.py OLD.json NEW.json [--top N]

Prints the players whose knobs moved the most (by total absolute change across
skill_level / crowd_following / confidence_following), plus players added or
dropped between the two files.
"""

import argparse
import json

KNOBS = ("skill_level", "crowd_following", "confidence_following")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    with open(args.old) as f:
        old = json.load(f)
    with open(args.new) as f:
        new = json.load(f)

    moves = []
    for name in old.keys() & new.keys():
        deltas = {k: new[name].get(k, 0) - old[name].get(k, 0) for k in KNOBS}
        weeks = new[name].get("weeks_played", 0) - old[name].get("weeks_played", 0)
        moves.append((sum(abs(d) for d in deltas.values()), name, deltas, weeks))
    moves.sort(reverse=True)

    print(f"{len(old.keys() & new.keys())} players in both files")
    print(f"\nBiggest movers (top {args.top}):")
    print(f"  {'Player':<30} {'skill':>8} {'crowd':>8} {'conf':>8} {'+weeks':>7}")
    for total, name, d, weeks in moves[: args.top]:
        if total == 0:
            break
        print(f"  {name[:30]:<30} {d['skill_level']:>+8.3f} {d['crowd_following']:>+8.3f} "
              f"{d['confidence_following']:>+8.3f} {weeks:>+7d}")

    for label, names in (("Added", new.keys() - old.keys()), ("Dropped", old.keys() - new.keys())):
        if names:
            print(f"\n{label}: {', '.join(sorted(names))}")


if __name__ == "__main__":
    main()
