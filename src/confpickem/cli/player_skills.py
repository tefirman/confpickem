#!/usr/bin/env python
"""
Unified Player Skills Management CLI

Analyzes historical player performance and applies realistic skill levels
to the simulator for more accurate predictions.

Usage:
  # Analyze historical performance and save skills
  player_skills.py analyze --weeks 3,4,5,6

  # Apply saved skills to current week
  player_skills.py apply --week 10

  # Do both: analyze and apply
  player_skills.py update --weeks 3,4,5,6 --week 10
"""

import sys
import argparse

# Import the player skills utilities from the package
try:
    from confpickem import analyze_player_skills, apply_realistic_skills
    analyze_main = analyze_player_skills.main
    apply_main = apply_realistic_skills.main
except ImportError as e:
    print(f"❌ Error importing player skills modules: {e}")
    print("   Make sure confpickem package is installed properly")
    sys.exit(1)


def main():
    """Unified player skills CLI"""
    parser = argparse.ArgumentParser(
        description='NFL Player Skills Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  analyze  - Analyze historical performance from saved HTML files
  apply    - Apply saved skills to current week's simulator
  update   - Do both: analyze then apply

Examples:
  # Analyze weeks 3-6 and save skills
  %(prog)s analyze --weeks 3,4,5,6

  # Apply saved skills to week 10
  %(prog)s apply --week 10

  # Analyze weeks 3-6 and apply to week 10
  %(prog)s update --weeks 3,4,5,6 --week 10
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze historical performance')
    analyze_parser.add_argument('--weeks', '-w', type=str, required=True,
                               help='Comma-separated week numbers (e.g., "3,4,5,6")')
    analyze_parser.add_argument('--league-id', '-l', type=int, default=15435,
                               help='Yahoo league ID (default: 15435)')

    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply saved skills to simulator')
    apply_parser.add_argument('--week', '-w', type=int, required=True,
                             help='NFL week number')
    apply_parser.add_argument('--league-id', '-l', type=int, default=15435,
                             help='Yahoo league ID (default: 15435)')

    # Update command (analyze + apply)
    update_parser = subparsers.add_parser('update', help='Analyze and apply skills')
    update_parser.add_argument('--weeks', type=str, required=True,
                              help='Comma-separated week numbers for analysis (e.g., "3,4,5,6")')
    update_parser.add_argument('--week', '-w', type=int, required=True,
                              help='NFL week number to apply to')
    update_parser.add_argument('--league-id', '-l', type=int, default=15435,
                              help='Yahoo league ID (default: 15435)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'analyze':
            print("🔍 ANALYZING HISTORICAL PLAYER PERFORMANCE")
            print("=" * 50)

            # Parse weeks
            weeks = [int(w.strip()) for w in args.weeks.split(',')]
            print(f"📊 Analyzing weeks: {weeks}")

            # Call analyze_player_skills main with appropriate args
            sys.argv = ['analyze_player_skills.py', '--weeks', args.weeks, '--league-id', str(args.league_id)]
            result = analyze_main()

            if result == 0:
                print("\n✅ Analysis complete!")
                print("💾 Skills saved to player_skills_analysis.json")
            return result

        elif args.command == 'apply':
            print(f"🎯 APPLYING PLAYER SKILLS TO WEEK {args.week}")
            print("=" * 50)

            # Call apply_realistic_skills main with appropriate args
            sys.argv = ['apply_realistic_skills.py', '--week', str(args.week), '--league-id', str(args.league_id)]
            result = apply_main()

            if result == 0:
                print("\n✅ Skills applied!")
                print("💾 Updated skills saved to current_player_skills.json")
            return result

        elif args.command == 'update':
            print("🔄 UPDATING PLAYER SKILLS")
            print("=" * 50)

            # First analyze
            print("\n📊 Step 1: Analyzing historical performance...")
            sys.argv = ['analyze_player_skills.py', '--weeks', args.weeks, '--league-id', str(args.league_id)]
            result = analyze_main()

            if result != 0:
                print("❌ Analysis failed")
                return result

            print("✅ Analysis complete!")

            # Then apply
            print(f"\n🎯 Step 2: Applying skills to week {args.week}...")
            sys.argv = ['apply_realistic_skills.py', '--week', str(args.week), '--league-id', str(args.league_id)]
            result = apply_main()

            if result != 0:
                print("❌ Apply failed")
                return result

            print("\n✅ Player skills updated successfully!")
            print("💾 Skills saved to current_player_skills.json")
            return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
