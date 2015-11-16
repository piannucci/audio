#!/usr/bin/env python
import numpy as np
from . import coreaudio
import threading
import queue
import sys

defBuf = 2048

class StreamArray:
    def __init__(self, dtype=np.float32, inBufSize=defBuf, outBufSize=defBuf):
        self.dtype = dtype
        self.inBufSize = inBufSize
        self.outBufSize = outBufSize
    def __len__(self):
        return sys.maxsize
    def __getitem__(self, sl):
        if isinstance(sl, slice):
            count = sl.stop - sl.start
        elif isinstance(sl, int):
            count = 1
        else:
            raise TypeError('Unexpected slice arguments')
        return self.produce(count).astype(self.dtype)
    def append(self, sequence):
        return self.consume(sequence)

class ThreadedStream(StreamArray):
    def __init__(self, channels=1, in_thread=None, out_thread=None, out_queue_depth=64,
                 inBufSize=defBuf, outBufSize=defBuf):
        super().__init__(inBufSize=inBufSize, outBufSize=outBufSize)
        self.channels = channels
        self.in_queue = queue.Queue(64)
        self.out_queue = queue.Queue(out_queue_depth)
        self.out_fragment = None
        self.numSamplesRead = 0
        self.numSamplesWritten = 0
        self.numSamplesProduced = 0
        self.numSamplesConsumed = 0
        self.warnOnUnderrun = True
        if hasattr(self, 'thread_consume') or in_thread:
            self.in_thread = threading.Thread(target=self.in_thread_loop, daemon=True)
            self.in_thread.start()
        if hasattr(self, 'thread_produce') or out_thread:
            self.out_thread = threading.Thread(target=self.out_thread_loop, daemon=True)
            self.out_thread.start()
    def in_thread_loop(self):
        while True:
            work = self.in_queue.get()
            if work is None:
                break
            self.numSamplesConsumed += work.shape[0]
            self.thread_consume(work)
    def out_thread_loop(self):
        while True:
            work = self.thread_produce()
            if work is None:
                break
            self.numSamplesProduced += work.shape[0]
            self.out_queue.put(work)
    def consume(self, sequence):
        self.numSamplesWritten += sequence.shape[0]
        try:
            self.in_queue.put_nowait(sequence)
        except queue.Full:
            print('ThreadedStream overrun')
        except Exception as e:
            print('ThreadedStream exception %s' % repr(e))
    def produce(self, count):
        result = np.empty((count, self.channels), self.dtype)
        i = 0
        if self.out_fragment is not None:
            n = min(self.out_fragment.shape[0], count)
            result[:n] = self.out_fragment[:n]
            i += n
            if n < self.out_fragment.shape[0]:
                self.out_fragment = self.out_fragment[n:]
            else:
                self.out_fragment = None
        while i < count:
            try:
                fragment = self.immediate_produce()
                if len(fragment.shape) == 1:
                    fragment = fragment[:,np.newaxis]
                if fragment.shape[1] != self.channels and fragment.shape[1] != 1:
                    raise Exception('ThreadedStream produced a stream with the wrong number of channels.')
            except queue.Empty:
                result[i:] = 0
                if self.warnOnUnderrun:
                    print('ThreadedStream underrun')
                break
            except Exception as e:
                print('ThreadedStream exception %s' % repr(e))
            else:
                n = min(count-i, fragment.shape[0])
                result[i:i+n] = fragment[:n]
                i += n
                if fragment.shape[0] > n:
                    self.out_fragment = fragment[n:]
        self.numSamplesRead += result.shape[0]
        return result
    def immediate_produce(self):
        return self.out_queue.get_nowait()
    def onMainThread(self, fn):
        coreaudio.add_to_main_thread_queue(fn)
    def stop(self):
        self.in_queue.put(None)

class WhiteNoise(ThreadedStream):
    def thread_produce(self):
        return np.random.standard_normal(1024) * .2

class Sine(ThreadedStream):
    def __init__(self):
        super().__init__(channels=2)
        self.f = 880.
        self.i = 0
        self.Fs = 48000.
    def thread_produce(self):
        j = np.arange(1024) + self.i
        t = j/self.Fs
        omega = self.f*2*np.pi
        self.i += j.size
        l = .2*np.sin(omega*t)
        r = .2*np.sin(omega*t+.5*np.pi)
        return np.hstack((l[:,np.newaxis], r[:,np.newaxis]))

class VUMeter(ThreadedStream):
    def __init__(self, Fs=96e3):
        super().__init__()
        self.peak = 0.
        self.Fs = Fs
    def thread_consume(self, sequence):
        volume = 30 + int(round(10.*np.log10((np.abs(sequence).astype(float)**2).mean())))
        peak = 30 + int(round(10.*np.log10((np.abs(sequence).astype(float)**2).max())))
        self.peak = max(peak, self.peak-8*sequence.size/self.Fs)
        bar = '#'*volume
        n = int(np.ceil(self.peak))
        bar = bar + ' ' * (n-len(bar)) + '|'
        sys.stdout.write('\r\x1b[K' + bar)
        sys.stdout.flush()

class Visualizer(ThreadedStream):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import pylab as pl
        self.pl = pl
        dpi = pl.rcParams['figure.dpi']
        import AppKit
        screens = AppKit.NSScreen.screens()
        mainScreen = AppKit.NSScreen.mainScreen()
        secondaryScreens = set(screens).difference(set([mainScreen]))
        if len(secondaryScreens):
            screen = next(iter(secondaryScreens))
        else:
            screen = mainScreen
        origin = screen.frame().origin
        screensize = screen.frame().size
        scale = screen.backingScaleFactor()
        self.fig = pl.figure(figsize=(screensize.width/dpi, screensize.height/dpi), facecolor='k', edgecolor='k')
        mgr = self.fig.canvas.manager
        class ignore:
            def __init__(self):
                pass
            def __call__(self, *args, **kwargs):
                pass
            def __getattr__(self, attr):
                return self
        mgr.toolbar = ignore()
        #mgr.window.wm_geometry("=%dx%d%+d%+d" % (screensize.width, screensize.height, .75*origin.x, origin.y))
        mgr.resize(screensize.width, screensize.height)
        mgr.set_window_title('Blurt')
        #self.fig.set_figwidth(screensize.width/dpi)
        #self.fig.set_figheight(screensize.height/dpi)
        self.ax = pl.gca()
        #widget = self.fig.canvas.get_tk_widget()
        #widget.winfo_toplevel().lift()
        coreaudio.sleepDuration = .01
    def draw(self):
        self.onMainThread(self.pl.draw)

class Oscilloscope(Visualizer):
    def __init__(self):
        super().__init__()
        self.line = self.ax.plot(np.zeros(self.inBufSize))[0]
        self.ax.set_ylim(-.5,.5)
    def thread_consume(self, sequence):
        self.line.set_ydata(sequence)
        self.draw()

def set_foregroundcolor(ax, color):
     '''For the specified axes, sets the color of the frame, major ticks,
         tick labels, axis labels, title and legend
     '''
     for tl in ax.get_xticklines() + ax.get_yticklines():
         tl.set_color(color)
     for spine in ax.spines:
         ax.spines[spine].set_edgecolor(color)
     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_color(color)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_color(color)
     ax.axes.xaxis.label.set_color(color)
     ax.axes.yaxis.label.set_color(color)
     ax.axes.xaxis.get_offset_text().set_color(color)
     ax.axes.yaxis.get_offset_text().set_color(color)
     ax.axes.title.set_color(color)
     lh = ax.get_legend()
     if lh != None:
         lh.get_title().set_color(color)
         lh.legendPatch.set_edgecolor('none')
         labels = lh.get_texts()
         for lab in labels:
             lab.set_color(color)
     for tl in ax.get_xticklabels():
         tl.set_color(color)
     for tl in ax.get_yticklabels():
         tl.set_color(color)

def set_backgroundcolor(ax, color):
     '''Sets the background color of the current axes (and legend).
         Use 'None' (with quotes) for transparent. To get transparent
         background on saved figures, use:
         pp.savefig("fig1.svg", transparent=True)
     '''
     ax.patch.set_facecolor(color)
     lh = ax.get_legend()
     if lh != None:
         lh.legendPatch.set_facecolor(color)

class SpectrumAnalyzer(Visualizer):
    def __init__(self, Fs=96e3, Fc=0., NFFT=2048, noverlap=2048-256, window=None, duration=1., transpose=False, sides=1):
        super().__init__()
        self.Fs = Fs
        self.Fc = Fc
        self.NFFT = NFFT
        self.noverlap = noverlap
        if window is None:
            window = self.pl.mlab.window_hanning
        self.window = window(np.ones(self.NFFT))
        self.duration = duration
        self.transpose = transpose
        self.sides = sides
        self.advance = self.NFFT - self.noverlap
        self.columns = int(round(self.duration * self.Fs / self.advance))
        self.buffer = np.zeros((self.NFFT*self.sides/2, self.columns*2), np.float32)
        self.t = 0.
        self.window_phase = 0
        self.ax = self.pl.gca()
        box = self.ax.bbox.bounds
        self.fim = self.ax.imshow(self.buffer[:,:self.columns] if not self.transpose else self.buffer[:,:self.columns].T,
                                  vmin=0., vmax=1., extent=(self.t, self.t+self.duration, 0, self.Fs/2000.), aspect='auto', interpolation='nearest')
        self.ax.set_xlim(self.t, self.t+self.duration)
        self.ax.set_ylim(0, self.Fs/2000.)
        self.ax.patch.set_visible(False)
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Frequency (kHz)')
        self.ax.set_title('Time-Frequency Visualizer')
        self.fig.patch.set_facecolor('k')
        set_backgroundcolor(self.ax, 'k')
        set_foregroundcolor(self.ax, 'w')
        self.input_fragment = np.zeros(0, np.complex64)
    def thread_consume(self, input):
        stream = np.r_[self.input_fragment, input]
        n = max(1 + (stream.size-self.NFFT) // self.advance, 0)
        if n < 4: n = 0
        self.input_fragment = stream[n*self.advance:]
        if n > 0:
            idx = np.arange(self.NFFT)[:,np.newaxis] + self.advance * np.arange(n)[np.newaxis,:]
            input = stream[idx] * self.window[:,np.newaxis]
            input = np.fft.fftshift(np.fft.fft(input, axis=0), 0)
            input = (np.log10(np.abs(input)) / 5.) + 1
            k = np.array([-.1,-.3,1.2,-.3,-.1]) * 1.5
            for i in xrange(input.shape[1]):
                input[:,i] = np.convolve(input[:,i], k, 'same')
            input = input[:(2-self.sides)*self.NFFT/2]

            cols = self.columns
            phase = self.window_phase

            if phase+cols+n <= 2*cols:
                self.buffer[:,phase+cols:phase+cols+n] = input
            else:
                m = phase+n-cols
                self.buffer[:,m-n:] = input[:,:n-m]
                self.buffer[:,0:m] = input[:,-m:]
            self.buffer[:,phase:phase+n] = input

            phase = (phase + n) % cols
            self.window_phase = phase

            t = self.t
            b = self.buffer[:,phase:phase+cols]
            self.fim.set_data(b if not self.transpose else b.T)
            self.fim.set_extent((t, t+self.duration, 0, self.Fs/2000.))
            self.ax.set_xlim(t, t+self.duration)
            def onMainThread():
                self.fig.draw_artist(self.fig)
                self.fig.canvas.blit()
            self.onMainThread(onMainThread)
            self.t += float(n * self.advance) / self.Fs

def go():
    import matplotlib
    matplotlib.use('TkAgg')
    matplotlib.rcParams['toolbar'] = 'None'
    import coreaudio
    Fs = 48000
    sa = SpectrumAnalyzer(duration=3, Fs=Fs)
    coreaudio.record(sa, Fs)

if __name__ == '__main__':
    pass #go()
