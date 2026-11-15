"""Swiss service clocks use local civil seconds, including DST change days."""
from datetime import datetime
from zoneinfo import ZoneInfo


def civil_service_seconds(epoch, day):
    local = datetime.fromtimestamp(epoch, ZoneInfo('Europe/Zurich')).replace(tzinfo=None)
    return int((local-datetime.fromisoformat(day)).total_seconds())
