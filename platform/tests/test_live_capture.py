"""Live capture must expose current frames and fail honestly without real waits."""
from collections import deque
from types import SimpleNamespace
import threading

import pytest

from server import live_capture


class Clock:
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            self.value = value


class ScriptedCapture:
    def __init__(self, clock, steps):
        self.clock = clock
        self.steps = deque(steps)
        self.calls = 0

    def read(self):
        self.calls += 1
        step = self.steps.popleft() if self.steps else (self.clock.monotonic(), False, None)
        if isinstance(step, Exception):
            raise step
        timestamp, okay, frame = step
        self.clock.set(timestamp)
        return okay, frame


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    # Replace only this module's clock. Threading deadlines still use real time.
    monkeypatch.setattr(live_capture, 'time', SimpleNamespace(monotonic=fake.monotonic))
    return fake


def test_slow_consumer_receives_latest_frame_instead_of_buffered_intermediate(clock):
    first_consumed = threading.Event()
    burst_read = threading.Event()

    class GatedCapture:
        calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                clock.set(.1)
                return True, 'first'
            if self.calls == 2:
                if not first_consumed.wait(.5):
                    raise AssertionError('Consumer did not release the next decoder frames')
                clock.set(.2)
                return True, 'stale-intermediate'
            if self.calls == 3:
                clock.set(.3)
                return True, 'latest'
            burst_read.set()
            return False, None

    info = {'captureFlags': []}
    frames = live_capture.latest_frames(GatedCapture(), 10, .1, 10, info, read_timeout_ms=10)
    try:
        assert next(frames)[:2] == (.1, 'first')
        # Simulate inference consuming time: decoder continues while generator
        # is paused at its yield. No artificial multi-second sleep is needed.
        first_consumed.set()
        assert burst_read.wait(.5)
        assert next(frames)[:2] == (.3, 'latest')
        with pytest.raises(StopIteration):
            next(frames)
    finally:
        first_consumed.set()
        frames.close()
    assert 'incomplete_observation' in info['captureFlags']


def test_initial_decoder_failure_has_no_invented_frame(clock):
    info = {'captureFlags': []}
    capture = ScriptedCapture(clock, [(0., False, None)])
    with pytest.raises(ValueError, match='No camera frames'):
        list(live_capture.latest_frames(capture, 300, 1, 301, info, read_timeout_ms=10))
    assert info['captureFlags'] == ['incomplete_observation']
    assert capture.calls == 1


def test_decoder_failure_preserves_prior_frame_but_marks_window_incomplete(clock):
    info = {'captureFlags': []}
    capture = ScriptedCapture(clock, [(.1, True, 'last-good'), (.2, False, None)])
    frames = list(live_capture.latest_frames(capture, 300, 1, 301, info, read_timeout_ms=10))
    assert [frame for _, frame, _ in frames] == ['last-good']
    assert 'incomplete_observation' in info['captureFlags']


@pytest.mark.filterwarnings('error::pytest.PytestUnhandledThreadExceptionWarning')
def test_decoder_exception_is_sanitized_and_marks_window_incomplete(clock, capsys):
    info = {'captureFlags': []}
    capture = ScriptedCapture(clock, [(.1, True, 'last-good'), RuntimeError('rtsp://user:private-camera-password@camera failed')])
    frames = list(live_capture.latest_frames(capture, 300, 1, 301, info, read_timeout_ms=10))
    assert [frame for _, frame, _ in frames] == ['last-good']
    assert 'incomplete_observation' in info['captureFlags']
    output = capsys.readouterr()
    assert 'private-camera-password' not in output.out + output.err


def test_frame_returned_after_observation_deadline_is_not_published(clock):
    info = {'captureFlags': []}
    capture = ScriptedCapture(clock, [(.2, True, 'inside-window'), (1.2, True, 'late-frame')])
    frames = list(live_capture.latest_frames(capture, 1, .1, 10, info, read_timeout_ms=10))
    assert capture.calls == 2
    assert [frame for _, frame, _ in frames] == ['inside-window']
    assert all(timestamp <= 1 for timestamp, _, _ in frames)


def test_frame_budget_stops_consumer_at_requested_limit(clock):
    release = threading.Event()
    exhausted = threading.Event()

    class GatedCapture(ScriptedCapture):
        def read(self):
            if self.calls == 1 and not release.wait(.5):
                raise AssertionError('Consumer did not release the next frame')
            result = super().read()
            if not result[0]:
                exhausted.set()
            return result

    info = {'captureFlags': []}
    capture = GatedCapture(clock, [(.1, True, 'one'), (.2, True, 'two'), (.3, False, None)])
    frames = live_capture.latest_frames(capture, 300, .01, 1, info, read_timeout_ms=10)
    try:
        assert next(frames)[:2] == (.1, 'one')
        release.set()
        assert exhausted.wait(.5)
        # A second decoded frame exists, but the consumer's frame budget is one.
        with pytest.raises(StopIteration):
            next(frames)
    finally:
        release.set()
        frames.close()


def test_consumer_deadline_expires_even_when_reader_never_notifies(clock, monkeypatch):
    """Model a stuck decoder without leaving a real thread or sleeping."""
    waits, joins = [], []

    class FakeCondition:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, timeout):
            waits.append(timeout)
            assert len(waits) == 1, 'Consumer ignored its deadline and waited again'
            assert 0 < timeout <= .25
            clock.set(2.)  # Past a one-second window plus a 10 ms read timeout.

    class StalledReaderThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass  # The reader never publishes a frame or signals completion.

        def join(self, timeout):
            joins.append(timeout)

    monkeypatch.setattr(live_capture, 'threading', SimpleNamespace(
        Condition=FakeCondition, Thread=StalledReaderThread, Event=threading.Event,
    ))
    info = {'captureFlags': []}
    with pytest.raises(ValueError, match='No camera frames'):
        list(live_capture.latest_frames(object(), 1, .1, 10, info, read_timeout_ms=10))
    assert info['captureFlags'] == ['incomplete_observation']
    assert joins == [pytest.approx(1.01)]
