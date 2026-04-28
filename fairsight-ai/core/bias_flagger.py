class BiasFlagger:
    LEVELS = {
        'CRITICAL': {
            'color': '#FF0000',
            'bg': '#fff1f2',
            'emoji': '🚨',
            'label': 'CRITICAL BIAS',
            'message': 'DO NOT DEPLOY — Immediate action required',
            'range': (0, 50)
        },
        'HIGH': {
            'color': '#FF4444',
            'bg': '#fff1f2',
            'emoji': '🔴',
            'label': 'HIGH BIAS',
            'message': 'Serious bias detected — Fix before deployment',
            'range': (50, 65)
        },
        'MODERATE': {
            'color': '#FF8800',
            'bg': '#fff7ed',
            'emoji': '🟡',
            'label': 'MODERATE BIAS',
            'message': 'Bias present — Monitor and improve',
            'range': (65, 80)
        },
        'FAIR': {
            'color': '#16a34a',
            'bg': '#f0fdf4',
            'emoji': '🟢',
            'label': 'FAIR MODEL',
            'message': 'Model meets fairness standards — Safe to deploy',
            'range': (80, 101)
        }
    }

    def get_flag(self, score):
        for level, info in self.LEVELS.items():
            lo, hi = info['range']
            if lo <= score < hi:
                return {**info, 'severity': level, 'score': score}
        return {**self.LEVELS['FAIR'], 'severity': 'FAIR', 'score': score}

    def flag_groups(self, group_rates):
        avg = sum(group_rates.values()) / len(group_rates)
        flags = []
        for group, rate in group_rates.items():
            dev = avg - rate
            if dev > 0.15:
                flags.append({
                    'group': group,
                    'gap': round(dev * 100, 1),
                    'severity': 'HIGH',
                    'message': (
                        f"'{group}' group receives "
                        f"{dev * 100:.1f}% fewer "
                        f"positive decisions than average"
                    )
                })
            elif dev > 0.08:
                flags.append({
                    'group': group,
                    'gap': round(dev * 100, 1),
                    'severity': 'MODERATE',
                    'message': (
                        f"'{group}' group is slightly "
                        f"disadvantaged by {dev * 100:.1f}%"
                    )
                })
        return flags

    def deploy_verdict(self, score, critical_flags):
        if score < 50 or len(critical_flags) > 0:
            return {
                'verdict': '🚫 DO NOT DEPLOY',
                'color': '#FF0000',
                'bg': '#fff1f2',
                'reason': (
                    'Critical bias detected. '
                    'Real people will be harmed.'
                )
            }
        elif score < 75:
            return {
                'verdict': '⚠️ DEPLOY WITH CAUTION',
                'color': '#FF8800',
                'bg': '#fff7ed',
                'reason': 'Moderate bias present. Monitor closely.'
            }
        return {
            'verdict': '✅ SAFE TO DEPLOY',
            'color': '#16a34a',
            'bg': '#f0fdf4',
            'reason': 'Model meets fairness standards.'
        }
