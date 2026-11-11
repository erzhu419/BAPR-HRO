"""Displacement bounds for an explicitly specified motion model."""
from math import cos, dist, floor, radians, sin, sqrt


def wgs84_reference_chord(first, second):
    """Distance between two lat/lon reference points on the WGS84 ellipsoid."""
    def cartesian(point):
        latitude, longitude = map(radians, point)
        eccentricity_sq = 0.0066943799901413165
        prime_vertical = 6_378_137. / sqrt(1 - eccentricity_sq * sin(latitude) ** 2)
        return (prime_vertical * cos(latitude) * cos(longitude),
                prime_vertical * cos(latitude) * sin(longitude),
                prime_vertical * (1 - eccentricity_sq) * sin(latitude))

    return dist(cartesian(first), cartesian(second))


def elapsed_seconds_floor(reference_distance, speed_cap, departure_radius, arrival_radius):
    """Conservative integer floor under bounded speed and endpoint uncertainty.

    Radii bound endpoint displacements in meters after nearest-point projection
    onto the solid ellipsoid. That projection is nonexpansive. In a model with
    common whole-second quantization, floor rounds the displacement bound down.
    Coordinates and fleet specifications do not establish the radii by themselves.
    """
    minimum_displacement = max(0., reference_distance - departure_radius - arrival_radius)
    return floor(minimum_displacement / speed_cap)
