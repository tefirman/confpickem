#!/usr/bin/env python
"""
Unified Player Skills Management CLI

Analyzes historical player performance and applies realistic skill levels
to the simulator for more accurate predictions.

Usage:
  # Analyze historical performance and save skills
  player_skills.py analyze --year 2024

  # Apply saved skills to current players
  player_skills.py apply

  # Do both: analyze and apply
  player_skills.py update --year 2024
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
  # Analyze 2024 season and save skills
  %(prog)s analyze --year 2024

  # Apply saved skills to current players
  %(prog)s apply

  # Apply skills from specific year
  %(prog)s apply --year 2024

  # Analyze 2024 season and apply
  %(prog)s update --year 2024
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze historical performance')
    analyze_parser.add_argument('--year', '-y', type=int, default=2024,
                               help='Year to analyze (default: 2024)')
    analyze_parser.add_argument('--league-id', '-l', type=int, default=15435,
                               help='Yahoo league ID (default: 15435)')

    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply saved skills to simulator')
    apply_parser.add_argument('--year', '-y', type=int, default=None,
                             help='Year to use (default: combine all available years)')
    apply_parser.add_argument('--league-id', '-l', type=int, default=15435,
                             help='Yahoo league ID (default: 15435)')

    # Update command (analyze + apply)
    update_parser = subparsers.add_parser('update', help='Analyze and apply skills')
    update_parser.add_argument('--years', '-y', type=str, default=None,
                              help='Year(s) to analyze, comma-separated (e.g., --years 2024,2025). '
                                   'If not specified, skips analysis and applies all available years.')
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

            print(f"📊 Analyzing year: {args.year}")

            # Call analyze_player_skills main with appropriate args
            sys.argv = ['analyze_player_skills.py', '--year', str(args.year)]
            result = analyze_main()

            if result == 0:
                print("\n✅ Analysis complete!")
                print(f"💾 Skills saved to player_skills_{args.year}.json")
            return result

        elif args.command == 'apply':
            print("🎯 APPLYING PLAYER SKILLS TO CURRENT PLAYERS")
            print("=" * 50)

            # Call apply_realistic_skills main with appropriate args
            if args.year:
                sys.argv = ['apply_realistic_skills.py', '--year', str(args.year)]
                print(f"📊 Using skills from year: {args.year}")
            else:
                sys.argv = ['apply_realistic_skills.py']
                print("📊 Combining skills from all available years")

            result = apply_main()

            if result == 0:
                print("\n✅ Skills applied!")
                print("💾 Updated skills saved to current_player_skills.json")
            return result

        elif args.command == 'update':
            print("🔄 UPDATING PLAYER SKILLS")
            print("=" * 50)

            # Parse years if provided
            if args.years:
                years = [int(y.strip()) for y in args.years.split(',')]
                print(f"📊 Will analyze years: {years}")

                # Analyze each year
                for i, year in enumerate(years, 1):
                    print(f"\n📊 Step {i}/{len(years)}: Analyzing {year} season...")
                    sys.argv = ['analyze_player_skills.py', '--year', str(year)]
                    result = analyze_main()

                    if result != 0:
                        print(f"❌ Analysis failed for {year}")
                        return result

                    print(f"✅ Analysis complete for {year}!")

                # Apply combining all available years
                print(f"\n🎯 Step {len(years) + 1}: Applying combined skills from all years...")
            else:
                print("📊 No years specified, skipping analysis step")
                print("🎯 Applying combined skills from all available years...")

            # Apply without specifying year to combine all available
            sys.argv = ['apply_realistic_skills.py']
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
