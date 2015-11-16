import time, numpy
import _thread
from . import _coreaudio
import queue

mainThreadQueue = queue.Queue()
sleepDuration = .05
numSamplesPlayed = 0
numSamplesRecorded = 0

class AudioInterface(object):
    def __init__(self, device=None, readyFn=None):
        self.playbackOffset = 0
        self.recordingOffset = 0
        self.recordingBuffer = None
        self.playbackBuffer = None
        self.shouldStop = False
        self.device = device
        self.readyFn = readyFn if readyFn else self.defaultReady
    def defaultReady(self):
        return getattr(self, 'recordingStarted', True) and \
               getattr(self, 'playbackStarted', True)
    def playbackCallback(self, buffer):
        global numSamplesPlayed
        numSamplesPlayed += buffer.shape[0]
        if not self.readyFn():
            buffer[:] = 0
        else:
            count = buffer.shape[0]
            y = self.playbackBuffer[self.playbackOffset:self.playbackOffset+count]
            if len(y.shape) == 1:
                y = y[:,numpy.newaxis]
            buffer[:y.shape[0]] = y
            buffer[y.shape[0]:] = 0
            self.playbackOffset += count
            if self.playbackOffset >= len(self.playbackBuffer) + self.outBufSize*10:
                return True
        if self.shouldStop:
            return True
        return False
    def recordingCallback(self, data):
        global numSamplesRecorded
        numSamplesRecorded += data.shape[0]
        if not self.readyFn():
            pass
        else:
            self.recordingBuffer.append(data.mean(1).copy())
            count = data.shape[0]
            self.recordingOffset += count
            if self.recordingOffset >= self.recordingLength:
                return True
        if self.shouldStop:
            return True
        return False
    def play(self, buffer, Fs, hostTime=None):
        self.playbackBuffer = buffer
        if hostTime is None:
            hostTime = _coreaudio.hostTimeNow()
        self.outBufSize = getattr(buffer, 'outBufSize', getattr(self, 'outBufSize', 2048))
        _coreaudio.startPlayback(self, Fs, self.device, hostTime)
    def record(self, count_or_stream, Fs, hostTime=None):
        if hostTime is None:
            hostTime = _coreaudio.hostTimeNow()
        if hasattr(count_or_stream, 'append'):
            self.recordingLength = len(count_or_stream)
            self.recordingBuffer = count_or_stream
        else:
            self.recordingLength = count_or_stream
            self.recordingBuffer = []
        self.inBufSize = getattr(count_or_stream, 'inBufSize', getattr(self, 'inBufSize', 2048))
        _coreaudio.startRecording(self, Fs, self.device, hostTime)
    def isPlaying(self):
        return hasattr(self, 'playbackDeviceID')
    def isRecording(self):
        return hasattr(self, 'recordingDeviceID')
    def idle(self):
        try:
            f = mainThreadQueue.get_nowait()
        except:
            pass
        else:
            f()
    def wait(self):
        try:
            while self.isPlaying() or self.isRecording():
                time.sleep(sleepDuration)
                self.idle()
        except KeyboardInterrupt:
            self.shouldStop = True
            while self.isPlaying() or self.isRecording():
                time.sleep(sleepDuration)
                self.idle()
        if hasattr(self, 'playbackException'):
            raise self.playbackException
        elif hasattr(self, 'recordingException'):
            raise self.recordingException
        if hasattr(self.recordingBuffer, 'stop'):
            self.recordingBuffer.stop()
        if isinstance(self.recordingBuffer, list):
            return numpy.concatenate(self.recordingBuffer)
        return None
    def stop(self):
        self.shouldStop = True
        self.wait()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()

## Have to initialize the threading mechanisms in order for PyGIL_Ensure to work
_thread.start_new_thread(lambda: None, ())

def play(buffer, Fs, device=None):
    ap = AudioInterface(device)
    try:
        ap.play(buffer, Fs)
    except KeyboardInterrupt:
        ap.stop()
    return ap.wait()

def record(count_or_stream, Fs, device=None):
    ap = AudioInterface(device)
    try:
        ap.record(count_or_stream, Fs)
    except KeyboardInterrupt:
        ap.stop()
    return ap.wait()

if 0:
    # use one AudioInterface for both play and record
    def play_and_record(buffer, Fs, device=None):
        ap = AudioInterface(device)
        try:
            ap.play(buffer, Fs)
            ap.record(buffer.shape[0] if hasattr(buffer, 'shape') else buffer, Fs)
        except KeyboardInterrupt:
            ap.stop()
        return ap.wait()
else:
    # use multiple AudioInterfaces
    def play_and_record(buffer, Fs, outDevice=None, inDevices=None):
        counter = 0
        devices = (outDevice,) + tuple(inDevices if inDevices else (None,))
        readyFn = lambda : True #counter == len(devices)
        aps = [AudioInterface(device, readyFn) for device in devices]
        startTime = _coreaudio.hostTimeNow() + 100e6 / _coreaudio.nanosecondsPerAbsoluteTick()
        try:
            aps[0].play(buffer, Fs, startTime)
            for ap in aps[1:]:
                ap.record(buffer.shape[0] if hasattr(buffer, 'shape') else buffer, Fs, startTime)
            while not readyFn():
                counter = sum(ap.defaultReady() for ap in aps)
        except KeyboardInterrupt:
            for ap in aps:
                ap.stop()
        return [ap.wait() for ap in aps]

def add_to_main_thread_queue(fn):
    mainThreadQueue.put_nowait(fn)

def devices():
    return _coreaudio.getDevices()
