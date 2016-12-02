# -*- coding: utf-8 -*-
"""
Class for reading data from an Axona dataset
Depends on: scipy
            h5py >= 2.5.0
            numpy
            quantities
Supported: Read
Authors: Mikkel E. Lepperød @CINPLA, Milad H. Mobarhan @CINPLA, Svenn-Arne Dragly @CINPLA
"""

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import sys
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, AnalogSignal,
                      ChannelIndex, Block)
import neo.io.tools
import numpy as np
import quantities as pq
import os

python_version = sys.version_info.major
if python_version == 2:
    from future.builtins import str


def parse_params(text):
    params = {}

    for line in text.split("\n"):
        line = line.strip()
        
        if len(line) == 0:
            continue
            
        line_splitted = line.split(" ", 1)
        
        name = line_splitted[0]
        params[name] = None

        if len(line_splitted) > 1:
            params[name] = line_splitted[1]
            
    return params


def parse_header_and_leave_cursor(file_handle):
    header = ""
    while True:
        search_string = "data_start"
        byte = file_handle.read(1)
        header += str(byte, 'latin-1')

        if not byte:
            raise IOError("Hit end of file '" + eeg_filename + "'' before '" + search_string + "' found.")

        if header[-len(search_string):] == search_string:
            break
            
    params = parse_params(header)

    return params
    

def assert_end_of_data(file_handle):
    remaining_data = str(file_handle.read(), 'latin1')
    assert(remaining_data.strip() == "data_end")


class AxonaIO(BaseIO):
    """
    Class for "reading" experimental data from an Axona dataset.
    """
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal, ChannelIndex, SpikeTrain]

    readable_objects = [Block, SpikeTrain]

    writeable_objects = []

    has_header = False
    is_streameable = False

    name = 'Axona'
    description = 'This IO reads experimental data from an Axona dataset'
    extensions = ['set']
    mode = 'file'

    def __init__(self, filename):
        """
        Arguments:
            filename : the filename
            dataset: points to a specific dataset in the .kwik and .raw.kwd file,
                     however this can be an issue to change in e.g. OpenElectrophy or Spykeviewer
        """
        BaseIO.__init__(self)
        self._absolute_filename = filename
        self._path, relative_filename = os.path.split(filename)
        self._base_filename, extension = os.path.splitext(relative_filename)

        if extension != ".set":
            raise ValueError("file extension must be '.set'")
            
        with open(self._absolute_filename, "r") as f:
            text = f.read()
            
        self._params = parse_params(text)
        print(self._params)

        # TODO read the set file and store necessary values as attributes on this object

    def read_block(self,
                   lazy=False,
                   cascade=True,
                   channel_index=None):
        """
        Arguments:
            Channel_index: can be int, iterable or None to select one, many or all channel(s)

        """

        blk = Block()
        if cascade:
            seg = Segment(file_origin=self._filename)
            blk.segments += [seg]

            if channel_index:
                if type(channel_index) is int: channel_index = [ channel_index ]
                if type(channel_index) is list: channel_index = np.array( channel_index )
            else:
                channel_index = np.arange(0,self._attrs['shape'][1])

            chx = ChannelIndex(name='all channels',
                               index=channel_index)
            blk.channel_indexes.append(chx)

            ana = self.read_analogsignal(channel_index=channel_index,
                                         lazy=lazy,
                                         cascade=cascade)

            # TODO Call all other read functions

            ana.channel_index = chx
            seg.duration = (self._attrs['shape'][0]
                          / self._attrs['kwik']['sample_rate']) * pq.s

            # neo.tools.populate_RecordingChannel(blk)
        blk.create_many_to_one_relationship()
        return blk

    def read_epoch():
        # TODO read epoch data
        pass

    def read_spiketrain(self, tetrode_index=0):
        # TODO add parameter to allow user to read raw data or not?
        assert(SpikeTrain in self.readable_objects)
        raw_filename = os.path.join(self._path, self._base_filename + "." + str(tetrode_index + 1))
        with open(raw_filename, "rb") as f:
            params = parse_header_and_leave_cursor(f)

            bytes_per_timestamp = int(params.get("bytes_per_timestamp", 4))
            bytes_per_sample = int(params.get("bytes_per_sample", 1))
            num_spikes = int(params.get("num_spikes", 0))
            num_chans = int(params.get("num_chans", 1))
            samples_per_spike = int(params.get("samples_per_spike", 50))

            bytes_per_spike_without_timestamp = samples_per_spike * bytes_per_sample
            bytes_per_spike = bytes_per_spike_without_timestamp + bytes_per_timestamp
            
            timestamp_dtype = ">u" + str(bytes_per_timestamp)
            waveform_dtype = "<i" + str(bytes_per_sample)            
            
            dtype = np.dtype([("times", (timestamp_dtype, 1), 1), ("waveforms", (waveform_dtype, 1), samples_per_spike)])
            
            data = np.fromfile(f, dtype=dtype, count=num_spikes * num_chans)
            assert_end_of_data(f)
            
        times = data["times"]
        waveforms = data["waveforms"]
        # TODO ensure waveforms is properly reshaped
        waveforms = waveforms.reshape(num_spikes, num_chans, samples_per_spike)
        
        # TODO get proper units
        # TODO get proper t_stop
        spike_train = SpikeTrain(times, units="ms", t_stop=times[-1],
                                 waveforms=waveforms)
                                     
        return spike_train
        
    def read_tracking(self):
        # TODO fix for multiple .pos files
        pos_filename = os.path.join(self._path, self._base_filename+".pos")
        if not os.path.exists(pos_filename):
            raise IOError("'.pos' file not found:" + pos_filename)

        with open(pos_filename, "rb") as f:
            params = parse_header_and_leave_cursor(f)
            print(params)

            sample_rate_split = params["sample_rate"].split(" ")
            assert(sample_rate_split[1] == "hz")
            sample_rate = float(sample_rate_split[0]) * pq.Hz  # sample_rate 50.0 hz

            eeg_samples_per_position = float(params["EEG_samples_per_position"])  # TODO remove?
            pos_samples_count = int(params["num_pos_samples"])
            bytes_per_timestamp = int(params["bytes_per_timestamp"])
            bytes_per_coord = int(params["bytes_per_coord"])
            tracked_spots_count = 2  # TODO read this from .set file (tracked_spots_count)

            timestamp_dtype = ">i" + str(bytes_per_timestamp)
            coord_dtype = "<i" + str(bytes_per_coord)

            bytes_per_pixel_count = 4
            pixel_count_dtype = ">i"+str(bytes_per_pixel_count)

            bytes_per_pos = (bytes_per_timestamp + 2*tracked_spots_count*bytes_per_coord + 8)  # pos_format is as follows for this file t,x1,y1,x2,y2,numpix1,numpix2.

            print(sample_rate, eeg_samples_per_position, pos_samples_count)

            # read data:
            # TODO: we need two dtype versions, one for one diode and another for two
            dtype = np.dtype([("t", (timestamp_dtype, 1)),
                              ("r1", (coord_dtype, 1), 2),
                              ("r2", (coord_dtype, 1), 2),
                              ("pixel_count", (pixel_count_dtype, 1), 2)])

            data = np.fromfile(f, dtype=dtype, count=pos_samples_count)
            assert_end_of_data(f)

            print(data)

    def read_analogsignal(self,
                          channel_index=None,
                          lazy=False,
                          cascade=True):
        """
        Read raw traces
        Arguments:
            channel_index: must be integer array
        """

        # TODO check that .egf file exists

        eeg_filename = os.path.join(self._path, self._base_filename+".eeg")
        if not os.path.exists(eeg_filename):
            raise IOError("'.eeg' file not found:" + eeg_filename)

        with open(eeg_filename, "rb") as f:
            params = parse_header_and_leave_cursor(f)

            sample_count = int(params["num_EEG_samples"])  # num_EEG_samples 120250
            sample_rate_split = params["sample_rate"].split(" ")
            assert(sample_rate_split[1] == "hz")
            sample_rate = float(sample_rate_split[0]) * pq.Hz  # sample_rate 250.0 hz

            if lazy:
                analog_signal = AnalogSignal([],
                                             units="uV",  # TODO get correct unit
                                             sampling_rate=sample_rate)
                # we add the attribute lazy_shape with the size if loaded
                # anasig.lazy_shape = self._attrs['shape'][0] # TODO do we need this
                # TODO Implement lazy loading
            else:
                data = np.fromfile(f, dtype='int8', count=sample_count)
                assert_end_of_data(f)
                # data = self._kwd['recordings'][str(self._dataset)]['data'].value[:, channel_index]
                # data = data * bit_volts[channel_index]
                analog_signal = AnalogSignal(data,
                                             units="uV",  # TODO get correct unit
                                             sampling_rate=sample_rate)
                # TODO read start time
            # for attributes out of neo you can annotate
            # anasig.annotate(info='raw traces')
            return analog_signal


if __name__ == "__main__":
    import sys
    io = AxonaIO(sys.argv[1])
    io.read_analogsignal()
    io.read_spiketrain()
    # io.read_spiketrainlist()
    io.read_tracking()
