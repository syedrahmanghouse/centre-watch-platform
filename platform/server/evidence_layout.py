"""Deterministic callouts that keep text off every detected person.

This module only arranges already-computed boxes. It does not infer identities,
alter detections, or need image pixels. If the image has no free space, labels
are placed on additional canvas below it instead of obscuring a person.
"""
from dataclasses import dataclass
import heapq
import math


Rect = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Placement:
    target_index: int
    rect: Rect
    connector: tuple[Point, ...]


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    placements: tuple[Placement, ...]


def _intersects(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _expanded(rect: Rect, amount: float) -> Rect:
    return (rect[0] - amount, rect[1] - amount, rect[2] + amount, rect[3] + amount)


def _center(rect: Rect) -> Point:
    return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


def _closest(rect: Rect, point: Point) -> Point:
    return (max(rect[0], min(rect[2], point[0])), max(rect[1], min(rect[3], point[1])))


def _segment_crosses(start: Point, end: Point, rect: Rect) -> bool:
    """Whether the open interior of a rectangle intersects a line segment."""
    # Shrinking by a subpixel permits a connector to travel along an obstacle
    # edge, which is useful for rectilinear routes between adjacent callouts.
    bounds = (rect[0] + .01, rect[1] + .01, rect[2] - .01, rect[3] - .01)
    low, high = 0.0, 1.0
    for origin, delta, minimum, maximum in (
        (start[0], end[0] - start[0], bounds[0], bounds[2]),
        (start[1], end[1] - start[1], bounds[1], bounds[3]),
    ):
        if abs(delta) < 1e-9:
            if not minimum < origin < maximum:
                return False
            continue
        enter, leave = (minimum - origin) / delta, (maximum - origin) / delta
        low, high = max(low, min(enter, leave)), min(high, max(enter, leave))
        if low > high:
            return False
    return low <= high and high > 0 and low < 1


def _clear_path(points: tuple[Point, ...], obstacles: list[Rect]) -> bool:
    return not any(_segment_crosses(a, b, rect)
                   for a, b in zip(points, points[1:]) for rect in obstacles)


def _length(points: tuple[Point, ...]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _anchors(label: Rect, target: Rect) -> tuple[Point, Point]:
    end = _closest(target, _center(label))
    return _closest(label, end), end


def _route(label: Rect, target: Rect, obstacles: list[Rect], width: int, height: int) -> tuple[Point, ...]:
    start, end = _anchors(label, target)
    direct = (start, end)
    if _clear_path(direct, obstacles):
        return direct
    # Try the two inexpensive right-angle routes before a visibility graph.
    elbows = ((start, (end[0], start[1]), end), (start, (start[0], end[1]), end))
    for path in sorted(elbows, key=_length):
        if _clear_path(path, obstacles):
            return tuple(p for index, p in enumerate(path) if not index or p != path[index - 1])

    # A shortest path through expanded obstacle corners avoids crossing faces
    # when a straight connector would run through a neighbouring student.
    nodes = [start, end]
    for rect in obstacles:
        for point in ((rect[0], rect[1]), (rect[2], rect[1]),
                      (rect[2], rect[3]), (rect[0], rect[3])):
            if (0 <= point[0] <= width and 0 <= point[1] <= height
                    and not any(r[0] < point[0] < r[2] and r[1] < point[1] < r[3] for r in obstacles)):
                nodes.append(point)
    nodes = list(dict.fromkeys(nodes))
    distances = {0: 0.0}
    previous = {}
    queue = [(0.0, 0)]
    while queue:
        distance, index = heapq.heappop(queue)
        if distance != distances.get(index):
            continue
        if index == 1:
            indices = [1]
            while indices[-1] != 0:
                indices.append(previous[indices[-1]])
            return tuple(nodes[i] for i in reversed(indices))
        for other, point in enumerate(nodes):
            if other == index:
                continue
            candidate_distance = distance + math.dist(nodes[index], point)
            if candidate_distance < distances.get(other, math.inf) and _clear_path((nodes[index], point), obstacles):
                distances[other] = candidate_distance
                previous[other] = index
                heapq.heappush(queue, (candidate_distance, other))
    # An overlapping detection can completely enclose the destination. The
    # association remains explicit even then; text is still always kept clear.
    return direct


def layout_labels(width: int, height: int, boxes: list[Rect], label_sizes: list[tuple[int, int]],
                  *, person_padding: int = 8, label_gap: int = 6, connector_gap: int = 2) -> Layout:
    """Return one callout per box, in input order, with optional extra height.

    ``boxes`` and ``label_sizes`` share indices. Coordinates use the rendered
    image scale. Labels never intersect any box expanded by ``person_padding``
    or another label; the returned height includes any fallback label rail.
    Connector endpoints belong to the associated box, not a nearby person.
    """
    if width <= 0 or height <= 0 or len(boxes) != len(label_sizes):
        raise ValueError("Positive canvas dimensions and one label size per box are required.")
    if min(person_padding, label_gap, connector_gap) < 0:
        raise ValueError("Callout spacing cannot be negative.")
    if any(w <= 0 or h <= 0 or w > width for w, h in label_sizes):
        raise ValueError("Labels must have positive dimensions and fit the canvas width.")
    if any(len(box) != 4 or not all(math.isfinite(v) for v in box)
           or box[0] >= box[2] or box[1] >= box[3] for box in boxes):
        raise ValueError("Boxes must be finite, nonempty rectangles.")
    if not boxes:
        return Layout(width, height, ())

    boxes = [tuple(float(v) for v in box) for box in boxes]
    obstacles = [_expanded(box, person_padding) for box in boxes]
    placed: dict[int, Rect] = {}
    canvas_height = height

    def candidates(index: int, rail_y: int | None = None):
        box, (w, h) = boxes[index], label_sizes[index]
        cx, cy = _center(box)
        if rail_y is not None:
            xs = {0, width - w, round(cx - w / 2)}
            xs.update(round(r[2] + label_gap) for r in placed.values())
            xs.update(round(r[0] - label_gap - w) for r in placed.values())
            return [(max(0, min(width - w, x)), rail_y) for x in sorted(xs)]
        xs = {0, width - w, round(cx - w / 2)}
        ys = {0, height - h, round(cy - h / 2)}
        for rect in obstacles + [_expanded(r, label_gap) for r in placed.values()]:
            xs.update((math.floor(rect[0] - w), math.ceil(rect[2])))
            ys.update((math.floor(rect[1] - h), math.ceil(rect[3])))
        # Grid positions fill gaps where obstacle-edge intersections alone
        # would push a callout far away from its associated student.
        step = max(10, min(24, h // 2))
        xs.update(range(0, width - w + 1, step))
        ys.update(range(0, height - h + 1, step))
        return ((x, y) for x in sorted(xs) if 0 <= x <= width - w
                for y in sorted(ys) if 0 <= y <= height - h)

    # The most crowded people have fewer nearby free areas. Place them first,
    # preserving the original index for numbering and deterministic ties.
    order = sorted(range(len(boxes)), key=lambda i: (
        -sum(_intersects(_expanded(boxes[i], max(label_sizes[i])), other) for j, other in enumerate(boxes) if i != j),
        boxes[i][1], boxes[i][0], i))
    for index in order:
        w, h = label_sizes[index]
        occupied = obstacles + [_expanded(rect, label_gap) for rect in placed.values()]
        options = []
        for x, y in candidates(index):
            rect = (x, y, x + w, y + h)
            if any(_intersects(rect, other) for other in occupied):
                continue
            start, end = _anchors(rect, boxes[index])
            distance = math.dist(start, end)
            # Proximity dominates; nearest-centre distance breaks ties along
            # large rectangles so a tag remains visually associated.
            score = distance + .15 * math.dist(_center(rect), _center(boxes[index]))
            options.append((score, y, x, rect))
        if options:
            # Prefer unobstructed leaders among the nearest candidates. This
            # avoids distant tags simply because a very long route is clear.
            nearest = sorted(options)[:32]
            def route_score(option):
                score, y, x, rect = option
                start, end = _anchors(rect, boxes[index])
                crossings = sum(_segment_crosses(start, end, other) for j, other in enumerate(boxes) if j != index)
                return score + crossings * 100, score, y, x
            placed[index] = min(nearest, key=route_score)[3]
            continue

        rail_y = height + person_padding + label_gap
        while True:
            valid = []
            for x, y in candidates(index, rail_y):
                rect = (x, y, x + w, y + h)
                if not any(_intersects(rect, other) for other in occupied):
                    valid.append(rect)
            if valid:
                rect = min(valid, key=lambda r: (math.dist(_center(r), _center(boxes[index])), r[0]))
                placed[index] = rect
                canvas_height = max(canvas_height, math.ceil(rect[3] + label_gap))
                break
            rail_y += h + label_gap

    placements = []
    for index, box in enumerate(boxes):
        # Do not expand a nearby box over the target endpoint: an endpoint
        # inside an obstacle has no possible route out of that obstacle.
        rect = placed[index]
        _, endpoint = _anchors(rect, box)
        other_boxes = [_expanded(other, connector_gap) for j, other in enumerate(boxes) if j != index]
        route_obstacles = [r for r in other_boxes
                           if not (r[0] < endpoint[0] < r[2] and r[1] < endpoint[1] < r[3])]
        route_obstacles += [_expanded(other, connector_gap) for j, other in placed.items() if j != index]
        route_obstacles.append(box)
        placements.append(Placement(index, rect, _route(rect, box, route_obstacles, width, canvas_height)))
    return Layout(width, canvas_height, tuple(placements))
