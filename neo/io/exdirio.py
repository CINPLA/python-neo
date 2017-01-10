# -*- coding: utf-8 -*-
"""
This is the implementation of the NEO IO for the exdir format.
Depends on: scipy
            h5py >= 2.5.0
            numpy
            quantities
Supported: Read
Authors: Milad H. Mobarhan @CINPLA,
         Svenn-Arne Dragly @CINPLA,
         Mikkel E. Lepperød @CINPLA
"""

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import sys
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, AnalogSignal,
                      ChannelIndex, Block, IrregularlySampledSignal)
import neo.io.tools
import numpy as np
import quantities as pq
import os
import glob
import exdir
import yaml
import copy

python_version = sys.version_info.major
if python_version == 2:
    from future.builtins import str


def yaml_write(fname, data):
    tmp = copy.deepcopy(data)
    try:
        with open(fname, 'r') as outfile:
            existing_data = yaml.load(outfile)
        tmp.update(existing_data)
    except FileNotFoundError:
        pass
    except:
        raise
    with open(fname, 'w') as outfile:
        outfile.write(yaml.dump(tmp, default_flow_style=False))


def set_attr(dataset, **kwargs):
    yaml_write(dataset.attributes_filename, kwargs)


def set_unit_attr(dataset, data):
    attr = {'unit': str(data.dimensionality)}
    yaml_write(dataset.attributes_filename, attr)


class ExdirIO(BaseIO):
    """
    Class for reading/writting of exdir fromat
    """

    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal, ChannelIndex, SpikeTrain]
    readable_objects = [Block, SpikeTrain]
    writeable_objects = []

    has_header = False
    is_streameable = False

    name = 'exdir'
    description = 'This IO reads experimental data from an eds folder'

    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # this info is for GUI stuff also
    mode = 'dir'

    def __init__(self, folder_path, mode='a'):
        """
        Arguments:
            folder_path : the folder path
        """
        BaseIO.__init__(self)
        self._absolute_folder_path = folder_path
        self._path, relative_folder_path = os.path.split(folder_path)
        self._base_folder, extension = os.path.splitext(relative_folder_path)

        if extension != ".exdir":
            raise ValueError("folder extension must be '.exdir'")

        self._exdir_folder = exdir.File(folder=folder_path, mode=mode)

        # TODO check if group exists
        self._processing = self._exdir_folder.require_group("processing")
        self._epochs = self._exdir_folder.require_group("epochs")
        if mode != 'w':
            self.path_to_segment = {}
            for epoch in self._epochs: # have to get all the segments out first
                if('Segment' in epoch):
                    seg = self.read_segment(path=epoch)
                    seg_path = self._epochs[epoch].relative_path
                    self.path_to_segment[seg_path] = seg
            self.path_to_channel_index = {}
            for process in self._processing:
                if('channel_group' in process):
                    group_id = int(process[-1])
                    index = None
                    for rec in ['EventWaveform', 'LFP']:
                        if rec in self._processing[process]:
                            rec_group = self._processing[process][rec]
                            for key in rec_group:
                                try:
                                    index = rec_group[key]['electrode_idx']
                                    break
                                except:
                                    raise
                                    pass
                    assert index is not None
                    chx = ChannelIndex(index=index,
                                       name='group_id #{}'.format(group_id),
                                       **{'group_id': group_id})
                    self.path_to_channel_index[process] = chx

    def _sptrs_to_times(self, sptrs):
        out = np.array([t for sptr in sptrs
                        for t in sptr.times.rescale('s').magnitude])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out)) * pq.s

    def _sptrs_to_wfseriesf(self, sptrs):
        wfs = np.vstack([sptr.waveforms for sptr in sptrs])
        assert wfs.shape[1:] == sptrs[0].waveforms.shape[1:]
        # neo: num_spikes, num_chans, samples_per_spike = wfs.shape
        return wfs

    def _sptrs_to_spike_clusters(self, sptrs):
        out = np.array([i for j, sptr in enumerate(sptrs)
                        for i in [j]*len(sptr)])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out))

    def _save_event_waveform(self, spike_times, waveforms, channel_indexes,
                            sampling_rate, channel_group, t_start,
                            t_stop):
        event_wf_group = channel_group.create_group('EventWaveform')
        # timeserie
        wf_group = event_wf_group.create_group('waveform_timeseries')
        start_time = wf_group.create_dataset('start_time', t_start)
        set_unit_attr(start_time, t_start)
        stop_time = wf_group.create_dataset('stop_time', t_stop)
        set_unit_attr(stop_time, t_stop)
        wf_group.create_dataset('electrode_idx', channel_indexes)
        # timestamps
        ts_data = wf_group.create_dataset("timestamps", spike_times)
        set_unit_attr(ts_data, spike_times)
        #  waveforms
        wf = wf_group.create_dataset("waveforms", waveforms)
        set_attr(wf, unit=str(waveforms.dimensionality),
                 sample_rate={'value': float(sampling_rate),
                              'unit': str(sampling_rate.dimensionality)})

    def _save_clusters(self, spike_clusters, channel_group, t_start,
                       t_stop):
        cl_group = channel_group.create_group('Clustering')
        start_time = cl_group.create_dataset('start_time', t_start)
        set_unit_attr(start_time, t_start)
        stop_time = cl_group.create_dataset('stop_time', t_stop)
        set_unit_attr(stop_time, t_stop)
        cl_data = cl_group.create_dataset('cluster_nums', spike_clusters)

    def _save_unit_times(self, sptrs, channel_group, t_start, t_stop):
        unit_times_group = channel_group.create_group('UnitTimes')
        start_time = unit_times_group.create_dataset('start_time', t_start)
        set_unit_attr(start_time, t_start)
        stop_time = unit_times_group.create_dataset('stop_time', t_stop)
        set_unit_attr(stop_time, t_stop)
        for sptr_id, sptr in enumerate(sptrs):
            times_group = unit_times_group.create_group('{}'.format(sptr_id))
            ts_data = times_group.create_dataset('times', sptr.times.magnitude)
            set_unit_attr(ts_data, sptr.times)

    def save(self, blk):
        for seg_idx, seg in enumerate(blk.segments):
            t_start = seg.t_start
            t_stop = seg.t_stop
            chxs = set([st.channel_index for st in seg.spiketrains]) # TODO must check if this makes sense
            # TODO sort indexes in case group_id is not provided
            for group_id, chx in enumerate(chxs):
                grp = chx.annotations['group_id'] or group_id
                grp_name = 'channel_group_{}'.format(grp)
                ch_group = self._processing.create_group(grp_name)
                sptrs = [st for st in seg.spiketrains
                         if st.channel_index == chx]
                sampling_rate = sptrs[0].sampling_rate.rescale('Hz')
                # self.sorted_idxs = np.argsort(times)
                spike_times = self._sptrs_to_times(sptrs)
                ns, = spike_times.shape
                num_chans = len(chx.index)
                waveforms = self._sptrs_to_wfseriesf(sptrs)
                assert waveforms.shape[::2] == (ns, num_chans)
                self._save_event_waveform(spike_times, waveforms, chx.index,
                                         sampling_rate, ch_group, t_start,
                                         t_stop)

                spike_clusters = self._sptrs_to_spike_clusters(sptrs)
                assert spike_clusters.shape == (ns,)
                self._save_clusters(spike_clusters, ch_group, t_start, t_stop)

                self._save_unit_times(sptrs, ch_group, t_start, t_stop)

    def read_block(self,
                   lazy=False,
                   cascade=True):
        # TODO read block
        blk = Block(file_origin=self._absolute_folder_path)
        if cascade:
            for epoch in self._epochs:
                if('Segment' not in epoch):
                    self.read_epoch(epoch)
            for process in self._processing:
                if(process == "Position"):
                    self.read_tracking(epo_path=epoch)
                for key in self._processing[process]:
                    if(key == "LFP"):
                        self.read_analogsignals(group_path=process)
                        seg.analogsignals.extend(ana)
                    if(key == "EventWaveform" or key == "UnitTimes"):
                        self.read_spiketrains(group_path=process,
                                              spike_path=key)
            for _, seg in self.path_to_segment.items():
                blk.segments.append(seg)

            # TODO add duration
            # TODO May need to "populate_RecordingChannel"

        return blk

    def read_segment(self, path):
        seg_group = self._epochs[path]
        t_start = pq.Quantity(seg_group["start_time"].data,
                              seg_group["start_time"].attrs["unit"])
        duration = pq.Quantity(seg_group["stop_time"].data,
                               seg_group["stop_time"].attrs["unit"])
        seg = Segment()
        seg.t_start = t_start
        seg.duration = duration
        print(seg)
        return seg

    def read_analogsignals(self, group_path):
        if(len(path) == 0):
            lfp_group = self._processing["LFP"]
        else:
            lfp_group = self._processing[path]["LFP"]

        analogsignals = []

        for key in lfp_group:
            timeserie = lfp_group[key]
            signal = timeserie["data"]
            analogsignal = AnalogSignal(
                signal.data,
                units=signal.attrs["unit"],
                sampling_rate=pq.Quantity(
                    timeserie.attrs["sample_rate"]["value"],
                    timeserie.attrs["sample_rate"]["unit"]
                )
            )

            analogsignals.append(analogsignal)

            # TODO: what about channel index
            # TODO: read attrs?

        return analogsignals

    def read_spiketrains(self, group_path, spike_path):
        if spike_path == "EventWaveform":
            return self.read_event_waveform(group_path)
        elif spike_path == 'UnitTimes':
            return self.read_unit_times(group_path)
        else:
            raise ValueError('spike path {}'.format(spike_path) +
                             'is not recognized, should be either ' +
                             '"EventWaveform" or "UnitTimes"')

    def read_event_waveform(self, group_path):
        # TODO implement read spike train
        event_waveform_group = self._processing[group_path]["EventWaveform"]
        clustering_group = self._processing[group_path]["Clustering"]

        chx = self.path_to_channel_index[group_path]

        spike_trains = []
        for key in event_waveform_group:
            timeserie = event_waveform_group[key]
            timestamps = timeserie["timestamps"]
            seg_path = timestamps.meta['segment']
            seg = self.path_to_segment[seg_path]
            waveforms = timeserie["waveforms"]
            clusters = clustering_group["cluster_nums"]
            for cluster in np.unique(clusters):
                indices, = np.where(clusters == cluster)
                spike_train = SpikeTrain(
                    times=pq.Quantity(timestamps.data[indices],
                                      timestamps.attrs["unit"]),
                    t_stop=seg.t_stop,
                    t_start=seg.t_start,
                    waveforms=pq.Quantity(
                        waveforms.data[indices, :, :],
                        waveforms.attrs["unit"]
                        ),
                    sampling_rate=pq.Quantity(
                        waveforms.attrs['sample_rate']['value'],
                        waveforms.attrs['sample_rate']['unit']
                        ),
                    )
                spike_train.channel_index = chx
                unit = Unit()
                unit.spiketrains.append(spike_train)
                chx.units.append(unit)
                spike_trains.append(spike_train)
                seg.spiketrains.append(spike_train)
            # TODO: read attrs?

        return spike_trains

    def read_unit_times(self, group_path):
        print('Warning: "read_unit_times" not implemented')
        pass

    def read_epoch(self):
        print('Warning: "read_epoch" not implemented')
        # TODO read epoch data
        pass

    def read_tracking(self, path):
        """
        Read tracking data_end
        """
        if(len(path) == 0):
            pos_group = self._processing["Position"]
        else:
            pos_group = self._processing[path]["Position"]
        irr_signals = []
        for key in pos_group:
            spot_group = pos_group[key]
            times = spot_group["timestamps"]
            coords = spot_group["data"]
            irr_signal = IrregularlySampledSignal(name=pos_group[key].name,
                                                  signal=coords.data,
                                                  times=times.data,
                                                  units=coords.attrs["unit"],
                                                  time_units=times.attrs["unit"])
            irr_signals.append(irr_signal)
        return irr_signals


if __name__ == "__main__":
    import sys
    testfile = "/tmp/test.exdir"
    io = ExdirIO(testfile)

    block = io.read_block()

    from neo.io.hdf5io import NeoHdf5IO

    testfile = "/tmp/test_exdir_to_neo.h5"
    try:
        os.remove(testfile)
    except:
        pass
    hdf5io = NeoHdf5IO(testfile)
    hdf5io.write(block)
