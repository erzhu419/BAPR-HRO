"""One latest numeric message per vehicle, after its publicly completed prefix."""

from experiments.reconstruction_v35.forecast import FittedMessageHistory


class SingleMessageHistory(FittedMessageHistory):
    """Keep all public records, but use one fixed-rule forecast likelihood.

    The selector reads only publication receipt and physical event order. An
    observed passage completes every preceding slot, including slots whose
    exact times were not observed. Schedule/prediction clocks do not determine
    completion, so a delayed vehicle's next event remains eligible.
    """
    def raw_forecast_records(self, trip):
        return super().forecast_records(trip)

    def selected_forecast(self, trip):
        observed_slots = [self.belief._positions[cid, kind][1]
                          for cid, kind, mode, value in self._records.get(trip, ())
                          if mode == 'observed']
        completed = max(observed_slots, default=-1)
        candidates = [record for record in self.raw_forecast_records(trip)
                      if self.belief._positions[record[0], record[1]][1] > completed]
        return min(candidates, key=lambda record: (
            -record[4], self.belief._positions[record[0], record[1]][1]), default=None)

    def forecast_records(self, trip):
        selected = self.selected_forecast(trip)
        return () if selected is None else (selected,)

    def public_state(self):
        rows = super().public_state()
        selected = {}
        for row in rows:
            if row['evidence'] != 'forecast':
                continue
            trip = row['trip']
            if trip not in selected:
                record = self.selected_forecast(trip)
                selected[trip] = None if record is None else record[:2]
            row['emission_selected'] = selected[trip] == (row['cid'], row['kind'])
        return rows
