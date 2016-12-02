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

# I need to subclass BaseIO
from neo.io.baseio import BaseIO

# to import from core
from neo.core import (Segment, SpikeTrain, Unit, Epoch, AnalogSignal,
                      ChannelIndex, Block)
import neo.io.tools
import numpy as np
import quantities as pq
import os

from future.builtins import str


class AxonaIO(BaseIO):
    """
    Class for "reading" experimental data from an Axona dataset.
    """
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal, ChannelIndex]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects = [Block]

    # This class is not able to write objects
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

    def read_spiketrains():
        # TODO read spiketrains from raw data and cut files
        # TODO add parameter to allow user to read raw data or not
        pass

    def read_tracking(self):

        # TODO fix for multiple .pos files
        pos_filename = os.path.join(self._path, self._base_filename+".pos")
        if not os.path.exists(pos_filename):
            raise IOError("'.pos' file not found:" + pos_filename)

        with open(pos_filename, "rb") as f:
            header = ""
            while True:
                search_string = "data_start"
                byte = f.read(1)
                header += str(byte)

                if not byte:
                    raise IOError("Hit end of file '" + pos_filename + "'' before '" + search_string + "' found.")

                if header[-len(search_string):] == search_string:
                    print("HEADER:")
                    print(header)
                    break


            params = {}

            for line in header.split("\r\n"):
                line_splitted = line.split(" ", 1)

                name = line_splitted[0]
                params[name] = None

                if len(line_splitted) > 1:
                    params[name] = line_splitted[1]

            sample_rate_split = params["sample_rate"].split(" ")
            assert(sample_rate_split[1] == "hz")
            sample_rate = float(sample_rate_split[0]) * pq.Hz # sample_rate 50.0 hz

            eeg_samples_per_position = float(params["EEG_samples_per_position"]) #TODO remove?
            pos_samples_count = int(params["num_pos_samples"])
            bytes_per_timestamp = int(params["bytes_per_timestamp"])
            bytes_per_coord = int(params["bytes_per_coord"])
            tracked_spots_count = 2 #TODO read this from .set file (tracked_spots_count)

            bytes_per_pos = (bytes_per_timestamp + 2*tracked_spots_count*bytes_per_coord + 8)# pos_format is as follows for this file t,x1,y1,x2,y2,numpix1,numpix2.

            print(sample_rate, eeg_samples_per_position, pos_samples_count)

            #read data:
            data = np.fromfile(f, dtype='int8', count=pos_samples_count*bytes_per_pos)
            remaining_data = str(f.read(), 'latin-1')
            assert(remaining_data == "\r\ndata_end\r\n")

            big_endian_vec = 256**np.arange(bytes_per_timestamp)[::-1].reshape(
                                                -1, bytes_per_timestamp)
            big_endian_mat = np.array([[256, 256, 256, 256], [1,1,1,1]])

            t_values = np.zeros(pos_samples_count)
            x_values  = np.zeros((pos_samples_count, tracked_spots_count))
            y_values  = np.zeros((pos_samples_count, tracked_spots_count))

            for i in range(pos_samples_count):
                pos_offset = i*bytes_per_pos
                t_bytes = data[pos_offset:pos_offset + bytes_per_timestamp]
                pos_offset += bytes_per_timestamp
                c_bytes=data[pos_offset:pos_offset+2*tracked_spots_count*bytes_per_coord].reshape(-1, bytes_per_coord).T
                coords = (c_bytes * big_endian_mat).sum(axis=0)

                #fill in values
                t_values[i] = (t_bytes * big_endian_vec).sum()
                x_values[i, ] = coords[::2]
                y_values[i, ] = coords[1::2]



###########################################################################################


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
            header = ""
            while True:
                search_string = "data_start"
                byte = f.read(1)
                header += str(byte, 'latin-1')

                if not byte:
                    raise IOError("Hit end of file '" + eeg_filename + "'' before '" + search_string + "' found.")

                if header[-len(search_string):] == search_string:
                    break

            params = {}

            for line in header.split("\r\n"):
                line_splitted = line.split(" ", 1)

                name = line_splitted[0]
                params[name] = None

                if len(line_splitted) > 1:
                    params[name] = line_splitted[1]

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
                remaining_data = str(f.read(), 'latin-1')
                assert(remaining_data == "\r\ndata_end\r\n")
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
    # io.read_analogsignal()
    io.read_tracking()
