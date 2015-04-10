import numpy as np
cimport numpy as cnp
from cpython cimport *

cdef extern from "stdlib.h":
    ctypedef int size_t
    ctypedef long intptr_t
    void *malloc(size_t size)
    void free(void* ptr)

cdef extern from "CoreFoundation/CoreFoundation.h":
    ctypedef unsigned int UInt8
    ctypedef unsigned int UInt16
    ctypedef unsigned int UInt32
    ctypedef const void * CFTypeRef
    ctypedef CFTypeRef CFDataRef
    ctypedef CFTypeRef CFStringRef
    ctypedef CFTypeRef CFAllocatorRef
    ctypedef UInt32 CFStringEncoding
    ctypedef signed long CFIndex
    CFDataRef CFStringCreateExternalRepresentation(CFAllocatorRef alloc, CFStringRef theString, CFStringEncoding encoding, UInt8 lossByte)
    CFStringEncoding kCFStringEncodingUTF8
    CFIndex CFDataGetLength(CFDataRef theData)
    const UInt8 *CFDataGetBytePtr(CFDataRef theData)
    void CFRelease(CFTypeRef cf)

cdef extern from "Python.h":
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)

cdef extern from "numpy/arrayobject.h":
    PyObject PyArray_Type
    object PyArray_NewFromDescr (PyObject *subtype, cnp.dtype newdtype, int nd,
                                 cnp.npy_intp *dims, cnp.npy_intp *strides, void
                                 *data, int flags, object parent)

def FOUR_CHAR_CODE(val):
    from struct import unpack
    return unpack('!I', val)[0]

def INV_FOUR_CHAR_CODE(val):
    from struct import pack
    return pack('!I', val)

kAudioHardwarePropertyProcessIsMaster = FOUR_CHAR_CODE('mast')
kAudioHardwarePropertyIsInitingOrExiting = FOUR_CHAR_CODE('inot')
kAudioHardwarePropertyUserIDChanged = FOUR_CHAR_CODE('euid')
kAudioHardwarePropertyDevices = FOUR_CHAR_CODE('dev#')
kAudioHardwarePropertyDefaultInputDevice = FOUR_CHAR_CODE('dIn ')
kAudioHardwarePropertyDefaultOutputDevice = FOUR_CHAR_CODE('dOut')
kAudioHardwarePropertyDefaultSystemOutputDevice = FOUR_CHAR_CODE('sOut')
kAudioHardwarePropertyDeviceForUID = FOUR_CHAR_CODE('duid')
kAudioHardwarePropertyProcessIsAudible = FOUR_CHAR_CODE('pmut')
kAudioHardwarePropertySleepingIsAllowed = FOUR_CHAR_CODE('slep')
kAudioHardwarePropertyUnloadingIsAllowed = FOUR_CHAR_CODE('unld')
kAudioHardwarePropertyHogModeIsAllowed = FOUR_CHAR_CODE('hogr')
kAudioHardwarePropertyRunLoop = FOUR_CHAR_CODE('rnlp')
kAudioHardwarePropertyPlugInForBundleID = FOUR_CHAR_CODE('pibi')
kAudioHardwarePropertyUserSessionIsActiveOrHeadless = FOUR_CHAR_CODE('user')
kAudioHardwarePropertyMixStereoToMono = FOUR_CHAR_CODE('stmo')

kAudioDevicePropertyDeviceName = FOUR_CHAR_CODE('name')
kAudioDevicePropertyDeviceIsRunning = FOUR_CHAR_CODE('goin')
kAudioDevicePropertyNominalSampleRate = FOUR_CHAR_CODE('nsrt')
kAudioDevicePropertyAvailableNominalSampleRates = FOUR_CHAR_CODE('nsr#')
kAudioDevicePropertyActualSampleRate = FOUR_CHAR_CODE('asrt')
kAudioDevicePropertyBufferFrameSize = FOUR_CHAR_CODE('fsiz')
kAudioDevicePropertyStreamFormat = FOUR_CHAR_CODE('sfmt')

kAudioObjectPropertyScopeGlobal = FOUR_CHAR_CODE('glob')
kAudioObjectPropertyScopeInput = FOUR_CHAR_CODE('inpt')
kAudioObjectPropertyScopeOutput = FOUR_CHAR_CODE('outp')
kAudioObjectPropertyElementMaster = 0
kAudioObjectClassID = FOUR_CHAR_CODE('aobj')
kAudioObjectClassIDWildcard = FOUR_CHAR_CODE('****')
kAudioObjectUnknown = 0

kAudioObjectSystemObject = 1

kAudioDevicePropertyScopeInput = FOUR_CHAR_CODE('inpt')
kAudioDevicePropertyScopeOutput = FOUR_CHAR_CODE('outp')
kAudioDevicePropertyScopePlayThrough = FOUR_CHAR_CODE('ptru')
kAudioDeviceClassID = FOUR_CHAR_CODE('adev')

kAudioFormatLinearPCM = FOUR_CHAR_CODE('lpcm')

kAudioFormatFlagIsFloat = (1 << 0)
kAudioFormatFlagIsSignedInteger = (1 << 2)
kAudioFormatFlagIsPacked = (1 << 3)

kAudioDevicePropertyDeviceUID = FOUR_CHAR_CODE('uid ')
kAudioObjectPropertyName = FOUR_CHAR_CODE('lnam')
kAudioObjectPropertyManufacturer = FOUR_CHAR_CODE('lmak')

kAudioTimeStampSampleTimeValid      = (1 << 0)
kAudioTimeStampHostTimeValid        = (1 << 1)
kAudioTimeStampRateScalarValid      = (1 << 2)
kAudioTimeStampWordClockTimeValid   = (1 << 3)
kAudioTimeStampSMPTETimeValid       = (1 << 4)

cdef extern from "CoreAudio/AudioHardware.h":
    ctypedef unsigned int OSStatus
    ctypedef unsigned char Boolean

    ctypedef UInt32 AudioObjectID
    ctypedef UInt32 AudioHardwarePropertyID
    ctypedef UInt32 AudioDeviceID
    ctypedef UInt32 AudioDevicePropertyID
    ctypedef UInt32 AudioStreamID

    ctypedef UInt32 AudioObjectPropertySelector
    ctypedef UInt32 AudioObjectPropertyScope
    ctypedef UInt32 AudioObjectPropertyElement

    ctypedef double Float64
    ctypedef unsigned long long UInt64
    ctypedef short int SInt16

    ctypedef struct AudioValueRange:
        Float64 mMinimum
        Float64 mMaximum

    ctypedef struct SMPTETime:
        UInt64  mCounter #;         //  total number of messages received
        UInt32  mType #;                //  the SMPTE type (see constants)
        UInt32  mFlags #;               //  flags indicating state (see constants
        SInt16  mHours #;               //  number of hours in the full message
        SInt16  mMinutes #;         //  number of minutes in the full message
        SInt16  mSeconds #;         //  number of seconds in the full message
        SInt16  mFrames #;          //  number of frames in the full message

    ctypedef struct AudioTimeStamp:
        Float64         mSampleTime #;  //  the absolute sample time
        UInt64          mHostTime #;        //  the host's root timebase's time
        Float64         mRateScalar #;  //  the system rate scalar
        UInt64          mWordClockTime #;   //  the word clock time
        SMPTETime       mSMPTETime #;       //  the SMPTE time
        UInt32          mFlags #;           //  the flags indicate which fields are valid
        UInt32          mReserved #;        //  reserved, pads the structure out to force 8 byte alignment

    ctypedef struct AudioStreamBasicDescription:
        Float64 mSampleRate #;      //  the native sample rate of the audio stream
        UInt32  mFormatID #;            //  the specific encoding type of audio stream
        UInt32  mFormatFlags #;     //  flags specific to each format
        UInt32  mBytesPerPacket #;  //  the number of bytes in a packet
        UInt32  mFramesPerPacket #; //  the number of frames in each packet
        UInt32  mBytesPerFrame #;       //  the number of bytes in a frame
        UInt32  mChannelsPerFrame #;    //  the number of channels in each frame
        UInt32  mBitsPerChannel #;  //  the number of bits in each channel
        UInt32  mReserved #;            //  reserved, pads the structure out to force 8 byte alignment

    ctypedef struct AudioObjectPropertyAddress:
        AudioObjectPropertySelector mSelector
        AudioObjectPropertyScope mScope
        AudioObjectPropertyElement mElement

    ctypedef OSStatus (*AudioObjectPropertyListenerProc)(AudioObjectID inObjectID, UInt32 inNumberAddresses, AudioObjectPropertyAddress *inAddresses, void *inClientData)
    void AudioObjectShow(AudioObjectID inObjectID)
    Boolean AudioObjectHasProperty(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress)
    OSStatus AudioObjectIsPropertySettable(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress, Boolean *outIsSettable)

    OSStatus AudioObjectGetPropertyDataSize(
        AudioObjectID inObjectID,
        AudioObjectPropertyAddress *inAddress,
        UInt32 inQualifierDataSize,
        void *inQualifierData,
        UInt32 *outDataSize)

    OSStatus AudioObjectGetPropertyData(
        AudioObjectID inObjectID,
        AudioObjectPropertyAddress *inAddress,
        UInt32 inQualifierDataSize,
        void *inQualifierData,
        UInt32 *ioDataSize,
        void *outData)

    OSStatus AudioObjectSetPropertyData(
        AudioObjectID inObjectID,
        AudioObjectPropertyAddress *inAddress,
        UInt32 inQualifierDataSize,
        void *inQualifierData,
        UInt32 inDataSize,
        void *inData)

    OSStatus AudioObjectAddPropertyListener(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress, AudioObjectPropertyListenerProc inListener, void *inClientData)
    OSStatus AudioObjectRemovePropertyListener(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress, AudioObjectPropertyListenerProc inListener, void *inClientData)

    ctypedef struct AudioBuffer:
        UInt32 mNumberChannels
        UInt32 mDataByteSize
        void* mData

    ctypedef struct AudioBufferList:
        UInt32 mNumberBuffers
        AudioBuffer mBuffers[1]

    ctypedef OSStatus (*AudioDeviceIOProc)(AudioDeviceID inDevice, AudioTimeStamp *inNow, AudioBufferList *inInputData, AudioTimeStamp *inInputTime, AudioBufferList *outOutputData, AudioTimeStamp *inOutputTime, void *inClientData)
    ctypedef AudioDeviceIOProc AudioDeviceIOProcID
    OSStatus AudioDeviceStart(AudioDeviceID inDevice, AudioDeviceIOProc inProc)
    OSStatus AudioDeviceStop(AudioDeviceID inDevice, AudioDeviceIOProc inProc)
    OSStatus AudioDeviceRemoveIOProc(AudioDeviceID inDevice, AudioDeviceIOProc inProc)
    OSStatus AudioDeviceCreateIOProcID(AudioDeviceID inDevice, AudioDeviceIOProc inProc, void *inClientData, AudioDeviceIOProcID *outIOProcID)
    OSStatus AudioDeviceDestroyIOProcID(AudioDeviceID inDevice, AudioDeviceIOProcID inIOProcID)


cdef extern from "mach/mach_time.h":
    UInt64 mach_absolute_time()

    ctypedef struct mach_timebase_info_data_t:
        UInt32 numer
        UInt32 denom

    int mach_timebase_info(mach_timebase_info_data_t *info)

cdef object arrayFromBuffer(AudioBuffer buffer, asbd):
    cdef UInt32 flags = asbd['mFormatFlags']

    if flags & kAudioFormatFlagIsFloat:
        format = 'f%d'
    elif flags & kAudioFormatFlagIsSignedInteger:
        format = 'i%d'
    else:
        format = 'u%d'

    cdef UInt32 bytesPerChannel = asbd['mBitsPerChannel'] // 8
    cdef cnp.dtype dt = np.dtype(format % bytesPerChannel)
    Py_INCREF(<object> dt)

    cdef UInt32 channelsPerFrame = asbd['mChannelsPerFrame']
    cdef UInt32 bytesPerFrame = asbd['mBytesPerFrame']

    cdef int ndims = 1 if channelsPerFrame == 1 else 2
    cdef cnp.npy_intp dims[2]
    cdef cnp.npy_intp strides[2]
    dims[0] = buffer.mDataByteSize // bytesPerFrame
    strides[0] = bytesPerFrame
    dims[1] = channelsPerFrame
    strides[1] = bytesPerChannel

    cdef UInt32 narrflags = cnp.NPY_WRITEABLE | cnp.NPY_C_CONTIGUOUS | cnp.NPY_F_CONTIGUOUS
    cdef cnp.ndarray narr = PyArray_NewFromDescr(&PyArray_Type, dt, ndims, dims, strides,
                                                 buffer.mData,
                                                 narrflags, <object>NULL)
    return narr


cdef OSStatus playbackCallback(
    AudioDeviceID inDevice, AudioTimeStamp *inNow, AudioBufferList *inInputData, AudioTimeStamp *inInputTime, AudioBufferList *outOutputData, AudioTimeStamp *inOutputTime,
    void *inClientData) with gil:

    cdef object cb = <object> inClientData

    try:
        sbd = cb.playbackASBD

        if not (inOutputTime.mFlags & kAudioTimeStampHostTimeValid):
            raise Exception('No host timestamps')
        startTime = cb.playbackStartHostTime
        outputTimeStart = inOutputTime.mHostTime
        framesDemanded = 0
        ticksPerFrame = 1e9 / (sbd['mSampleRate'] * cb.nanosecondsPerAbsoluteTick)
        bytesPerFrame = sbd['mBytesPerFrame']

        cb.playbackStarted = True
        zeroFill = False
        for i from 0 <= i < outOutputData.mNumberBuffers:
            framesDemanded += outOutputData.mBuffers[i].mDataByteSize / bytesPerFrame
            outputTime = outputTimeStart + framesDemanded * ticksPerFrame
            buffer = arrayFromBuffer(outOutputData.mBuffers[i], sbd)
            if outputTime < startTime:
                # zero pad front
                firstGoodSample = min((startTime - outputTime) / ticksPerFrame, buffer.shape[0])
                buffer[:firstGoodSample] = 0
                buffer = buffer[firstGoodSample:]
            if not zeroFill:
                if cb.playbackCallback(buffer):
                    stopPlayback(cb)
                    zeroFill = True
            else:
                buffer[:] = 0
    except Exception, e:
        stopPlayback(cb)
        cb.playbackException = e

    return 0

cdef OSStatus recordingCallback(
    AudioDeviceID inDevice, AudioTimeStamp *inNow, AudioBufferList *inInputData, AudioTimeStamp *inInputTime, AudioBufferList *outOutputData, AudioTimeStamp *inOutputTime,
    void *inClientData) with gil:

    cdef object cb = <object> inClientData
    
    try:
        sbd = cb.recordingASBD

        if not (inInputTime.mFlags & kAudioTimeStampHostTimeValid):
            raise Exception('No host timestamps')
        startTime = cb.recordingStartHostTime
        inputTimeStart = inInputTime.mHostTime
        framesProvided = 0
        ticksPerFrame = 1e9 / (sbd['mSampleRate'] * cb.nanosecondsPerAbsoluteTick)
        bytesPerFrame = sbd['mBytesPerFrame']

        cb.recordingStarted = True
        for i from 0 <= i < inInputData.mNumberBuffers:
            framesProvided += inInputData.mBuffers[i].mDataByteSize / bytesPerFrame
            inputTime = inputTimeStart + framesProvided * ticksPerFrame
            buffer = arrayFromBuffer(inInputData.mBuffers[i], sbd)
            if inputTime < startTime:
                # drop samples
                firstGoodSample = min((startTime - inputTime) / ticksPerFrame, buffer.shape[0])
                buffer = buffer[firstGoodSample:]
            if cb.recordingCallback(buffer):
                stopRecording(cb)
                break
    except Exception, e:
        stopRecording(cb)
        cb.recordingException = e

    return 0

cdef AudioObjectGetProperty(AudioObjectID obj, AudioObjectPropertySelector selector, AudioObjectPropertyScope scope, UInt32 propertySize, void *prop):
    cdef AudioObjectPropertyAddress address
    address.mSelector = selector
    address.mScope = scope
    address.mElement = kAudioObjectPropertyElementMaster
    cdef OSStatus status = AudioObjectGetPropertyData(obj, &address, 0, NULL, &propertySize, prop)
    if status:
        from struct import pack
        raise RuntimeError, "Unable to get property %s (error %d)" % (pack("!I", selector), status)

cdef AudioObjectGetPropertySize(AudioObjectID obj, AudioObjectPropertySelector selector, AudioObjectPropertyScope scope):
    cdef AudioObjectPropertyAddress address
    address.mSelector = selector
    address.mScope = scope
    address.mElement = kAudioObjectPropertyElementMaster
    cdef UInt32 propertySize
    cdef OSStatus status = AudioObjectGetPropertyDataSize(obj, &address, 0, NULL, &propertySize)
    if status:
        from struct import pack
        raise RuntimeError, "Unable to get property size %s" % pack("!I", selector)
    return propertySize

cdef AudioObjectSetProperty(AudioObjectID obj, AudioObjectPropertySelector selector, AudioObjectPropertyScope scope, UInt32 propertySize, void *prop):
    cdef AudioObjectPropertyAddress address
    address.mSelector = selector
    address.mScope = scope
    address.mElement = 0
    cdef OSStatus status = AudioObjectSetPropertyData(obj, &address, 0, NULL, propertySize, prop)
    if status:
        from struct import pack
        raise RuntimeError, "Unable to set property %s" % pack("!I", selector)

cdef unicodeFromCFString(CFStringRef string):
    cdef CFDataRef data = CFStringCreateExternalRepresentation(NULL, string, kCFStringEncodingUTF8, 0)
    out = PyUnicode_FromStringAndSize(<const char *>CFDataGetBytePtr(data), CFDataGetLength(data))
    CFRelease(data)
    return out

def getDevices():
    cdef int size = AudioObjectGetPropertySize(kAudioObjectSystemObject, kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal)
    cdef UInt32 deviceCount = size / sizeof(AudioDeviceID)
    cdef AudioDeviceID *devices = <AudioDeviceID *>malloc(size)
    cdef CFStringRef deviceUID = NULL, deviceName = NULL,  deviceManufacturer = NULL
    AudioObjectGetProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, size, devices)
    result = []
    for i from 0 <= i < deviceCount:
        uid, name, manufacturer = None, None, None
        AudioObjectGetProperty(devices[i], kAudioDevicePropertyDeviceUID, kAudioObjectPropertyScopeGlobal, sizeof(CFStringRef), &deviceUID)
        uid = unicodeFromCFString(deviceUID)
        CFRelease(deviceUID)
        AudioObjectGetProperty(devices[i], kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, sizeof(CFStringRef), &deviceName)
        name = unicodeFromCFString(deviceName)
        CFRelease(deviceName)
        AudioObjectGetProperty(devices[i], kAudioObjectPropertyManufacturer, kAudioObjectPropertyScopeGlobal, sizeof(CFStringRef), &deviceManufacturer)
        manufacturer = unicodeFromCFString(deviceManufacturer)
        CFRelease(deviceManufacturer)
        result.append({'deviceID': devices[i],
                       'deviceUID': uid,
                       'deviceName': name,
                       'deviceManufacturer': manufacturer})
    free(devices)
    return result

cdef UInt32 outBufSize = 2048

def getOutBufSize():
    return outBufSize

def startPlayback(cb, sampleRate, device, startTime):
    cdef AudioDeviceID outputDeviceID = 0
    if device is None:
        # Get the default sound output device
        AudioObjectGetProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, sizeof(outputDeviceID), &outputDeviceID)
        if not outputDeviceID:
            raise RuntimeError, "Default audio device was unknown."
    else:
        devices = getDevices()
        if 0 <= device < len(devices):
            outputDeviceID = devices[device]['deviceID']
        else:
            raise RuntimeError, "No such audio device."

    AudioObjectSetProperty(outputDeviceID, kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyScopeOutput, sizeof(outBufSize), &outBufSize)

    cdef AudioStreamBasicDescription sbd
    sbd.mSampleRate = sampleRate
    sbd.mFormatID = kAudioFormatLinearPCM
    sbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
    sbd.mBytesPerPacket = 8
    sbd.mFramesPerPacket = 1
    sbd.mBytesPerFrame = 8
    sbd.mChannelsPerFrame = 2
    sbd.mBitsPerChannel = 32
    sbd.mReserved = 0
    AudioObjectSetProperty(outputDeviceID, kAudioDevicePropertyStreamFormat, kAudioDevicePropertyScopeOutput, sizeof(sbd), &sbd)

    cdef AudioDeviceIOProcID ioProcID
    if AudioDeviceCreateIOProcID(outputDeviceID, <AudioDeviceIOProc>&playbackCallback, <void *>cb, &ioProcID):
        raise RuntimeError, "Failed to add the IO Proc."
    if AudioDeviceStart(outputDeviceID, ioProcID):
        raise RuntimeError, "Couldn't start the device."

    Py_INCREF(cb)
    cb.playbackDeviceID = outputDeviceID
    cb.playbackFs = sampleRate
    cb.playbackIOProcID = <long>ioProcID
    cb.playbackASBD = sbd
    cb.playbackStarted = False
    cb.playbackStartHostTime = <UInt64>startTime
    cb.nanosecondsPerAbsoluteTick = nanosecondsPerAbsoluteTick()


def stopPlayback(cb):
    cdef AudioDeviceIOProcID ioProcID = <AudioDeviceIOProcID><long>cb.playbackIOProcID
    AudioDeviceStop(cb.playbackDeviceID, ioProcID)
    AudioDeviceDestroyIOProcID(cb.playbackDeviceID, ioProcID)
    del cb.playbackDeviceID
    del cb.playbackFs
    del cb.playbackIOProcID
    del cb.playbackASBD
    del cb.playbackStarted
    del cb.playbackStartHostTime
    Py_DECREF(cb)

cdef UInt32 inBufSize = 2048

def getInBufSize():
    return inBufSize

def startRecording(cb, sampleRate, device, startTime):
    cdef AudioDeviceID inputDeviceID = 0
    if device is None:
        # Get the default sound input device
        AudioObjectGetProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal, sizeof(inputDeviceID), &inputDeviceID)
        if not inputDeviceID:
            raise RuntimeError, "Default audio device was unknown."
    else:
        devices = getDevices()
        if 0 <= device < len(devices):
            inputDeviceID = devices[device]['deviceID']
        else:
            raise RuntimeError, "No such audio device."

    AudioObjectSetProperty(inputDeviceID, kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyScopeInput, sizeof(inBufSize), &inBufSize)

    cdef AudioStreamBasicDescription sbd
    sbd.mSampleRate = sampleRate
    sbd.mFormatID = kAudioFormatLinearPCM
    sbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
    sbd.mBytesPerPacket = 8
    sbd.mFramesPerPacket = 1
    sbd.mBytesPerFrame = 8
    sbd.mChannelsPerFrame = 2
    sbd.mBitsPerChannel = 32
    sbd.mReserved = 0
    AudioObjectSetProperty(inputDeviceID, kAudioDevicePropertyStreamFormat, kAudioDevicePropertyScopeInput, sizeof(sbd), &sbd)

    cdef AudioDeviceIOProcID ioProcID
    if AudioDeviceCreateIOProcID(inputDeviceID, <AudioDeviceIOProc>&recordingCallback, <void *>cb, &ioProcID):
        raise RuntimeError, "Failed to add the IO Proc."
    if AudioDeviceStart(inputDeviceID, ioProcID):
        raise RuntimeError, "Couldn't start the device."

    Py_INCREF(cb)
    cb.recordingDeviceID = inputDeviceID
    cb.recordingFs = sampleRate
    cb.recordingIOProcID = <long>ioProcID
    cb.recordingASBD = sbd
    cb.recordingStarted = False
    cb.recordingStartHostTime = <UInt64>startTime
    cb.nanosecondsPerAbsoluteTick = nanosecondsPerAbsoluteTick()


def stopRecording(cb):
    cdef AudioDeviceIOProcID ioProcID = <AudioDeviceIOProcID><long>cb.recordingIOProcID
    AudioDeviceStop(cb.recordingDeviceID, ioProcID)
    AudioDeviceDestroyIOProcID(cb.recordingDeviceID, ioProcID)
    del cb.recordingDeviceID
    del cb.recordingFs
    del cb.recordingIOProcID
    del cb.recordingASBD
    del cb.recordingStarted
    del cb.recordingStartHostTime
    Py_DECREF(cb)

def hostTimeNow():
    return mach_absolute_time()

def nanosecondsPerAbsoluteTick():
    cdef mach_timebase_info_data_t info
    mach_timebase_info(&info)
    return <double>info.numer / info.denom

cnp.import_array()
