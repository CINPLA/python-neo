# -*- coding: utf-8 -*-
"""
Class for reading data from a .kwik dataset
Depends on: scipy
            h5py >= 2.5.0
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

class AxonaIO(BaseIO):
    """
    Class for "reading" experimental data from an Axona dataset.
    """
    is_readable = True # This class can only read data
    is_writable = False # write is not supported

    supported_objects    = [ Block, Segment, AnalogSignal,
                             ChannelIndex]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects  = [ Block ]

    # This class is not able to write objects
    writeable_objects   = [ ]

    has_header         = False
    is_streameable     = False

    name               = 'Axona'
    description        = 'This IO reads experimental data from an Axona dataset'
    extensions         = [ 'set' ]
    mode = 'file'

    def __init__(self, filename) :
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

        print("Extension:", extension)

        assert(extension == ".set") # TODO more friendly error message?

        # TODO read the set file and store necessary values as attributes on this object

    def read_block(self,
                     lazy=False,
                     cascade=True,
                     channel_index=None
                    ):
        """
        Arguments:
            Channel_index: can be int, iterable or None to select one, many or all channel(s)

        """

        blk = Block()
        if cascade:
            seg = Segment( file_origin=self._filename )
            blk.segments += [ seg ]

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

    def read_tracking():
        # TODO read tracking
        # TODO nag about this missing function
        pass

    def read_analogsignal(self,
                      channel_index=None,
                      lazy=False,
                      cascade=True,
                      ):
        """
        Read raw traces
        Arguments:
            channel_index: must be integer array
        """

        # TODO check that .eeg or .egf file exists

        # if self._attrs['app_data']:
        #     bit_volts = self._attrs['app_data']['channel_bit_volts']
        #     sig_unit = 'uV'
        # else:
        #     bit_volts = np.ones((self._attrs['shape'][1])) # TODO: find conversion in phy generated files
        #     sig_unit =  'bit'
        if lazy:
            # anasig = AnalogSignal([],
            #                       units=sig_unit,
            #                       sampling_rate=self._attrs['kwik']['sample_rate']*pq.Hz,
            #                       t_start=self._attrs['kwik']['start_time']*pq.s,
            #                       )
            # # we add the attribute lazy_shape with the size if loaded
            # anasig.lazy_shape = self._attrs['shape'][0]
            # TODO Implement lazy loading
            pass
        else:
            # data = self._kwd['recordings'][str(self._dataset)]['data'].value[:, channel_index]
            # data = data * bit_volts[channel_index]
            # anasig = AnalogSignal(data,
            #                            units=sig_unit,
            #                            sampling_rate=self._attrs['kwik']['sample_rate']*pq.Hz,
            #                            t_start=self._attrs['kwik']['start_time']*pq.s,
            #                            )
            # data = []  # delete from memory
            # TODO Read the data from the file
            pass
        # for attributes out of neo you can annotate
        # anasig.annotate(info='raw traces')
        # return anasig


if __name__ == "__main__":
    print("Test")
    io = AxonaIO("/tmp/2012/2012-08/2012-08-31-104220-1199/raw/DVH_2012083105.set")
    io.read_analogsignal()
