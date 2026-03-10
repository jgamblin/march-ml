#!/usr/bin/env python3
"""
Live Tournament Scorer
Tracks actual tournament results vs. predictions in real-time
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class LiveTournamentScorer:
    """
    Real-time tournament tracking and validation
    """
    
    def __init__(self, predictions_file, output_dir='results'):
        """
        Initialize scorer with bracket predictions
        
        Args:
            predictions_file: Path to sim_5000_final.json or similar
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            self.predictions = json.load(f)
        
        # Initialize actual results tracking
        self.actual_results = {
            'first_round': [],
            'second_round': [],
            'sweet_16': [],
            'elite_8': [],
            'final_four': [],
            'championship': []
        }
        
        # Round mappings
        self.round_names = [
            'first_round', 'second_round', 'sweet_16', 
            'elite_8', 'final_four', 'championship'
        ]
        
        self.round_display_names = {
            'first_round': 'First Round (R64)',
            'second_round': 'Second Round (R32)',
            'sweet_16': 'Sweet 16',
            'elite_8': 'Elite 8',
            'final_four': 'Final Four',
            'championship': 'Championship'
        }
        
    def add_result(self, round_name, winner, loser=None, score=None):
        """
        Add an actual game result
        
        Args:
            round_name: Name of round (first_round, second_round, etc.)
            winner: Name of winning team
            loser: Name of losing team (optional)
            score: Game score tuple (winner_score, loser_score) (optional)
        """
        result = {
            'winner': winner,
            'loser': loser,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.actual_results[round_name].append(result)
    
    def import_results_from_json(self, results_file):
        """
        Import results from a JSON file
        
        Expected format:
        {
            "first_round": [
                {"winner": "Duke", "loser": "16-seed", "score": [80, 65]},
                ...
            ],
            ...
        }
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        for round_name in self.round_names:
            if round_name in results:
                self.actual_results[round_name] = results[round_name]
    
    def calculate_accuracy(self, round_name):
        """
        Calculate prediction accuracy for a specific round
        
        Returns:
            dict with accuracy, correct_picks, total_games
        """
        actual = self.actual_results[round_name]
        
        if len(actual) == 0:
            return {
                'accuracy': 0.0,
                'correct_picks': 0,
                'total_games': 0,
                'games': []
            }
        
        # Get predictions for this round
        pred_teams = self.predictions.get('regional_breakdown', {})
        champion_probs = self.predictions.get('championship_probabilities', [])
        
        correct = 0
        games_analysis = []
        
        for game in actual:
            winner = game['winner']
            loser = game.get('loser')
            
            # Find prediction for this matchup
            # For now, use simple champion probability lookup
            winner_prob = self._get_team_probability(winner, round_name)
            loser_prob = self._get_team_probability(loser, round_name) if loser else 0.0
            
            # Prediction is correct if predicted winner (higher prob) matches actual winner
            predicted_correct = winner_prob > loser_prob if loser else True
            
            if predicted_correct:
                correct += 1
            
            games_analysis.append({
                'winner': winner,
                'loser': loser,
                'winner_prob': winner_prob,
                'loser_prob': loser_prob,
                'predicted_correctly': predicted_correct
            })
        
        return {
            'accuracy': correct / len(actual),
            'correct_picks': correct,
            'total_games': len(actual),
            'games': games_analysis
        }
    
    def _get_team_probability(self, team_name, round_name):
        """
        Get team's probability of reaching a specific round
        """
        if not team_name:
            return 0.0
        
        # Look up in championship probabilities
        champ_probs = self.predictions.get('championship_probabilities', [])
        for team_data in champ_probs:
            if team_data.get('team') == team_name:
                # Return appropriate probability based on round
                round_key_map = {
                    'first_round': 'r64_prob',
                    'second_round': 'r32_prob',
                    'sweet_16': 'sweet16_prob',
                    'elite_8': 'elite8_prob',
                    'final_four': 'final_four_prob',
                    'championship': 'championship_prob'
                }
                
                prob_key = round_key_map.get(round_name, 'championship_prob')
                return team_data.get(prob_key, team_data.get('probability', 0.0))
        
        return 0.0
    
    def generate_report(self):
        """
        Generate comprehensive accuracy report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall_summary': {},
            'by_round': {},
            'insights': []
        }
        
        total_correct = 0
        total_games = 0
        
        for round_name in self.round_names:
            round_stats = self.calculate_accuracy(round_name)
            
            if round_stats['total_games'] > 0:
                report['by_round'][round_name] = {
                    'display_name': self.round_display_names[round_name],
                    'accuracy': round_stats['accuracy'],
                    'correct_picks': round_stats['correct_picks'],
                    'total_games': round_stats['total_games'],
                    'games': round_stats['games']
                }
                
                total_correct += round_stats['correct_picks']
                total_games += round_stats['total_games']
        
        # Overall accuracy
        report['overall_summary'] = {
            'total_games': total_games,
            'correct_picks': total_correct,
            'accuracy': total_correct / total_games if total_games > 0 else 0.0,
            'rounds_completed': sum(1 for r in self.round_names if len(self.actual_results[r]) > 0)
        }
        
        # Generate insights
        if total_games > 0:
            best_round = max(
                [(r, self.calculate_accuracy(r)['accuracy']) 
                 for r in self.round_names if len(self.actual_results[r]) > 0],
                key=lambda x: x[1],
                default=(None, 0)
            )
            
            report['insights'].append({
                'type': 'best_round',
                'message': f"Best prediction accuracy in {self.round_display_names.get(best_round[0], 'N/A')}: {best_round[1]:.1%}"
            })
            
            # Check for upsets (predicted lower prob team won)
            upsets = []
            for round_name in self.round_names:
                round_stats = self.calculate_accuracy(round_name)
                for game in round_stats['games']:
                    if not game['predicted_correctly']:
                        upsets.append({
                            'round': self.round_display_names[round_name],
                            'winner': game['winner'],
                            'loser': game['loser'],
                            'winner_prob': game['winner_prob']
                        })
            
            if len(upsets) > 0:
                report['insights'].append({
                    'type': 'upsets',
                    'count': len(upsets),
                    'message': f"Model missed {len(upsets)} upset(s)",
                    'upsets': upsets[:5]  # Top 5 upsets
                })
        
        return report
    
    def print_report(self):
        """
        Print formatted report to console
        """
        report = self.generate_report()
        
        print("=" * 60)
        print("LIVE TOURNAMENT SCORING REPORT")
        print("=" * 60)
        print(f"\nGenerated: {report['generated_at']}")
        
        summary = report['overall_summary']
        print(f"\n{'OVERALL ACCURACY':^60}")
        print("-" * 60)
        print(f"Total Games: {summary['total_games']}")
        print(f"Correct Predictions: {summary['correct_picks']}")
        print(f"Accuracy: {summary['accuracy']:.1%}")
        print(f"Rounds Completed: {summary['rounds_completed']}/6")
        
        if len(report['by_round']) > 0:
            print(f"\n{'ACCURACY BY ROUND':^60}")
            print("-" * 60)
            
            for round_name in self.round_names:
                if round_name in report['by_round']:
                    round_data = report['by_round'][round_name]
                    print(f"\n{round_data['display_name']:20s} "
                          f"{round_data['correct_picks']:2d}/{round_data['total_games']:2d} "
                          f"({round_data['accuracy']:6.1%})")
        
        if len(report['insights']) > 0:
            print(f"\n{'KEY INSIGHTS':^60}")
            print("-" * 60)
            
            for insight in report['insights']:
                print(f"\n• {insight['message']}")
                
                if insight['type'] == 'upsets' and 'upsets' in insight:
                    print("  Notable upsets:")
                    for upset in insight['upsets']:
                        print(f"    - {upset['winner']} over {upset['loser']} "
                              f"({upset['round']}, {upset['winner_prob']:.1%} predicted)")
        
        print("\n" + "=" * 60)
    
    def save_report(self, filename='live_tournament_report.json'):
        """
        Save report to JSON file
        """
        report = self.generate_report()
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Saved report to {output_file}")
        return output_file

def create_example_results_template():
    """
    Create an example results file template for manual entry
    """
    template = {
        "_comment": "Fill in actual tournament results as they happen",
        "_example": {
            "winner": "Duke",
            "loser": "North Dakota State",
            "score": [78, 65]
        },
        "first_round": [],
        "second_round": [],
        "sweet_16": [],
        "elite_8": [],
        "final_four": [],
        "championship": []
    }
    
    output_file = Path('data/actual_results_template.json')
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Created results template: {output_file}")
    print("Fill in this file with actual results as the tournament progresses")
    return output_file

def main():
    print("=== Live Tournament Scorer ===\n")
    
    # Check for predictions file
    predictions_file = Path('results/sim_5000_final.json')
    if not predictions_file.exists():
        print(f"Error: Predictions file not found: {predictions_file}")
        print("Run simulate_bracket.py first to generate predictions")
        return
    
    # Create example template
    print("Creating results template...")
    template_file = create_example_results_template()
    
    # Initialize scorer
    scorer = LiveTournamentScorer(predictions_file)
    
    # Check for actual results file
    actual_results_file = Path('data/actual_results.json')
    
    if actual_results_file.exists():
        print(f"\nLoading actual results from {actual_results_file}...")
        scorer.import_results_from_json(actual_results_file)
        
        # Generate and print report
        print("\nGenerating accuracy report...")
        scorer.print_report()
        
        # Save detailed report
        scorer.save_report('live_tournament_report.json')
    else:
        print(f"\nNo actual results found at {actual_results_file}")
        print("As the tournament progresses:")
        print("  1. Copy data/actual_results_template.json to data/actual_results.json")
        print("  2. Fill in game results as they happen")
        print("  3. Run this script again to see updated accuracy")
        print("\nExample result entry:")
        print('  {"winner": "Duke", "loser": "North Dakota State", "score": [78, 65]}')
    
    print("\n=== Usage Guide ===\n")
    print("To track the tournament in real-time:")
    print("  1. Fill data/actual_results.json with game results")
    print("  2. Run: python3 scripts/live_tournament_scorer.py")
    print("  3. View accuracy by round and overall")
    print("\nThe scorer will identify:")
    print("  • Overall prediction accuracy")
    print("  • Accuracy by tournament round")
    print("  • Upset games where model was wrong")
    print("  • Lessons learned for next year's model")

if __name__ == '__main__':
    main()
