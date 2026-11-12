"""One frozen forecast-error likelihood candidate on the unchanged trip law."""

from experiments.reconstruction_v34.forecast import MessageHistory


class FittedMessageHistory(MessageHistory):
    """Use training error bias/scale; retain all raw forecasts and message rules.

    Conditional independence across endpoint errors remains a working model.
    Updating a marginal forecast posterior is not used as an observation
    likelihood. The physical prior, chain transitions and support stay fixed.
    """
    def __init__(self, belief, parameters, *, fit, **kwargs):
        super().__init__(belief, parameters, **kwargs)
        self.fit = fit

    def emission_parameters(self, cid, kind):
        route = self.belief.graph.connections[cid].route
        return self.fit['route_kind'].get(route, {}).get(kind, self.fit['pooled'][kind])

    def forecast_log_emission(self, cid, kind, predicted, sd, times):
        fitted = self.emission_parameters(cid, kind)
        # Error was fitted as actual minus forecast, hence center=forecast+bias.
        return -.5*((times-predicted-fitted['bias_seconds'])/fitted['sd_seconds'])**2

    def public_state(self):
        rows = super().public_state()
        for row in rows:
            if row['evidence'] == 'forecast':
                fitted = self.emission_parameters(row['cid'], row['kind'])
                row.update(emission_bias_seconds=fitted['bias_seconds'],
                           emission_sd_seconds=fitted['sd_seconds'])
        return rows
