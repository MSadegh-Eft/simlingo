"""Serializable monitoring decisions; no optimizer or parameter changes here."""
import copy
import math


def stopping_defaults(stage):
    return {'patience': 3, 'score_relative_delta': .005, 'loss_relative_delta': .01,
            'minimum_updates': 500 if stage == 'align' else 581,
            'maximum_lr_fraction': .25}


def monitor_due(update, total):
    return update == 0 or update == total or update % 125 == 0


def driving_score(candidate, reference):
    values = []
    for name in ('route_ade', 'speed_ade'):
        a, b = candidate.get(name, {}), reference.get(name, {})
        if not a.get('samples') or not b.get('samples'):
            raise ValueError('Missing driving selection metric: ' + name)
        if any(a.get(k) != b.get(k) for k in ('samples', 'groups', 'population_sha256')):
            raise ValueError('Selection population differs: ' + name)
        if not math.isfinite(a['value']) or not math.isfinite(b['value']) or b['value'] <= 0:
            raise ValueError('Invalid driving selection metric: ' + name)
        values.append(a['value'] / b['value'])
    return sum(values) / len(values)


class Controller:
    def __init__(self, rules, state=None):
        self.rules = rules
        self.state = copy.deepcopy(state) if state is not None else {
            'schema': 1, 'history': [], 'signal_bests': {}, 'wait': 0,
            'candidate': None, 'stop_reason': None}

    def observe(self, update, total, score, language_loss, geometry_loss,
                retention, lr_fractions, binding):
        signals = {'score': score, 'language': language_loss, 'geometry': geometry_loss}
        if (type(update) is not int or not 0 <= update <= total or not lr_fractions
                or any(v is None or not math.isfinite(v) or v < 0 for v in signals.values())
                or any(not math.isfinite(v) or v < 0 for v in lr_fractions)):
            raise ValueError('Incomplete/non-finite monitor; no stopping decision')
        if not isinstance(retention.get('passed'), bool):
            raise ValueError('Missing retention screen')
        if self.state['history'] and update <= self.state['history'][-1]['update']:
            raise ValueError('Monitor decisions must be committed exactly once in increasing order')
        improved = []
        for name, value in signals.items():
            best = self.state['signal_bests'].get(name)
            delta = self.rules['score_relative_delta'] if name == 'score' else self.rules['loss_relative_delta']
            if best is None or value < best * (1 - delta):
                self.state['signal_bests'][name] = value
                improved.append(name)
        self.state['wait'] = 0 if improved else self.state['wait'] + 1
        row = {'update': update, 'signals': signals, 'improved': improved,
               'retention_passed': retention['passed'], 'lr_fractions': lr_fractions,
               'wait': self.state['wait'], 'binding': binding}
        previous = self.state['candidate']
        rank = (not retention['passed'], score)
        new_candidate = update > 0 and (previous is None or rank < (not previous['retention_passed'], previous['signals']['score']))
        if new_candidate:
            self.state['candidate'] = copy.deepcopy(row)
        plateau = (update >= self.rules['minimum_updates']
                   and max(lr_fractions) <= self.rules['maximum_lr_fraction']
                   and self.state['wait'] >= self.rules['patience'])
        reason = 'budget_exhausted' if update == total else 'validation_plateau' if plateau else None
        self.state['stop_reason'] = reason
        row.update(stop_reason=reason, new_candidate=new_candidate)
        self.state['history'].append(row)
        return row


def full_candidate_order(candidate_update, final_update):
    """The final checkpoint is evaluated only if a different candidate fails."""
    if candidate_update is None:
        raise ValueError('No trained monitoring candidate')
    return [candidate_update] + ([final_update] if final_update != candidate_update else [])
