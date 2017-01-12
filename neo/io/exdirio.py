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
import os.path as op
import glob
import exdir
import yaml
import copy

python_version = sys.version_info.major
if python_version == 2:
    from future.builtins import str


def get_quantity_attr(dataset, key):
    return pq.Quantity(dataset.attrs[key]['value'], dataset.attrs[key]['unit'])


def get_quantity(data, dataset):
    return pq.Quantity(data, dataset.attrs["unit"])


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
    extensions = ['exdir']
    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # this info is for GUI stuff also
    mode = 'dir'

    def __init__(self, filename, mode='a'):
        """
        Arguments:
            folder_path : the folder path
        """
        BaseIO.__init__(self)
        self._absolute_folder_path = filename
        self._path, relative_folder_path = os.path.split(filename)
        self._base_folder, extension = os.path.splitext(relative_folder_path)

        if extension != ".exdir":
            raise ValueError("folder extension must be '.exdir'")

        self._exdir_folder = exdir.File(folder=filename, mode=mode)

        # TODO check if group exists
        self._processing = self._exdir_folder.require_group("processing")

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
        if 'cluster_id' in sptrs[0].annotations:  # assumes its true for all
            out = np.array([i for sptr in sptrs for i in
                           [sptr.annotations['cluster_id']]*len(sptr)])
        else:
            out = np.array([i for j, sptr in enumerate(sptrs)
                            for i in [j]*len(sptr)])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out))

    def _save_event_waveform(self, spike_times, waveforms, channel_indexes,
                             sampling_rate, channel_group, t_start, t_stop):
        event_wf_group = channel_group.create_group('EventWaveform')
        wf_group = event_wf_group.create_group('waveform_timeseries')
        wf_group.attrs['start_time'] = t_start
        wf_group.attrs['stop_time'] = t_stop
        wf_group.attrs['electrode_idx'] = channel_indexes
        ts_data = wf_group.create_dataset("timestamps", spike_times)
        wf = wf_group.create_dataset("waveforms", waveforms)
        wf.attrs['sample_rate'] = sampling_rate

    def _save_clusters(self, spike_clusters, channel_group, t_start,
                       t_stop, cluster_groups):
        cl_group = channel_group.create_group('Clustering')
        cl_group.attrs['start_time'] = t_start
        cl_group.attrs['stop_time'] = t_stop
        if cluster_groups is not None:
            for key, val in cluster_groups.items():
                cl_group.attrs[key] = val
        cluster_nums = np.unique(spike_clusters)
        cl_data = cl_group.create_dataset('cluster_nums', cluster_nums)
        cl_data = cl_group.create_dataset('nums', spike_clusters)

    def _save_unit_times(self, sptrs, channel_group, t_start, t_stop):
        unit_times_group = channel_group.create_group('UnitTimes')
        unit_times_group.attrs['start_time'] = t_start
        unit_times_group.attrs['stop_time'] = t_stop
        for idx, sptr in enumerate(sptrs):
            if 'cluster_id' in sptr.annotations:
                sptr_id = sptr.annotations['cluster_id']
            else:
                sptr_id = idx
            times_group = unit_times_group.create_group('{}'.format(sptr_id))
            for key, val in sptr.annotations.items():
                times_group.attrs[key] = val
            ts_data = times_group.create_dataset('times', sptr.times)

    def save(self, blk):
        # TODO save block annotations
        for seg_idx, seg in enumerate(blk.segments):
            t_start = seg.t_start
            t_stop = seg.t_stop
            seg_name = seg.name or 'Segment_{}'.format(seg_idx)
            seg_group = self._processing.create_group(seg_name)
            for key, val in seg.annotations.items():
                seg_group.attrs[key] = val
            seg_group.attrs['duration'] = t_stop - t_start
            chxs = set([st.channel_index for st in seg.spiketrains]) # TODO must check if this makes sense
            # TODO sort indexes in case group_id is not provided
            for group_id, chx in enumerate(chxs):
                grp = chx.annotations['group_id'] or group_id
                grp_name = 'channel_group_{}'.format(grp)
                ch_group = seg_group.create_group(grp_name)
                ch_group.attrs['electrode_idx'] = chx.index
                for key, val in chx.annotations.items():
                    ch_group.attrs[key] = val
                sptrs = [st for st in seg.spiketrains
                         if st.channel_index == chx]
                sampling_rate = sptrs[0].sampling_rate.rescale('Hz')
                # self.sorted_idxs = np.argsort(times)
                spike_times = self._sptrs_to_times(sptrs)
                ns, = spike_times.shape
                num_chans = len(chx.index)
                waveforms = self._sptrs_to_wfseriesf(sptrs)
                assert waveforms.shape[:2] == (ns, num_chans)
                self._save_event_waveform(spike_times, waveforms, chx.index,
                                          sampling_rate, ch_group, t_start,
                                          t_stop)

                spike_clusters = self._sptrs_to_spike_clusters(sptrs)
                assert spike_clusters.shape == (ns,)
                cluster_groups = None
                if 'group' in chx.annotations:
                    cluster_groups = chx.annotations['group']
                self._save_clusters(spike_clusters, ch_group, t_start, t_stop,
                                    cluster_groups)

                self._save_unit_times(sptrs, ch_group, t_start, t_stop)

                # TODO save analogsignals
                # TODO save epochs

    def read_block(self,
                   lazy=False,
                   cascade=True):
        '''
        if you have several segments, cluster nums are assumed to be unique,
        thus if two clusters have same num they are assumed to be of same
        neo.Unit and their spiketrains are appended.
        '''
        # TODO read block
        blk = Block(file_origin=self._absolute_folder_path)
        if cascade:
            for prc_name, prc_group in self._processing.items():
                if('Segment' in prc_name):
                    duration = get_quantity_attr(prc_group, 'duration')
                    seg = Segment(name=prc_name)
                    seg.duration = duration # TODO BUG in neo, cannot be set as argument
                    blk.segments.append(seg)
                for prc_sub_name, prc_sub_group in prc_group.items():
                    if(prc_sub_name == "Position"):
                        self.read_tracking(group=prc_sub_group)
                    if('channel_group' in prc_sub_name):
                        chx = self._get_channel_index(group=prc_sub_group)
                        blk.channel_indexes.append(chx)
                        anas = self.read_analogsignals(group=prc_sub_group)
                        if anas is not None:
                            seg.analogsignals.extend(anas)
                            chx.analogsignals.extend(anas)
                            for ana in anas:
                                ana.channel_index = chx

                        sptrs = self.read_event_waveform(group=prc_sub_group)
                        if sptrs is None:
                            sptrs = self.read_unit_times(group=prc_sub_group)
                        if sptrs is not None:
                            seg.spiketrains.extend(sptrs)
                            for sptr in sptrs:
                                sptr.channel_index = chx
                                cluster_id = sptr.annotations['cluster_id']
                                units = {unit.annotations['cluster_id']: unit
                                         for unit in chx.units}
                                if cluster_id in units:
                                    unit = units[cluster_id]
                                else:
                                    unit = Unit(name='Unit #{}'.format(cluster_id),
                                                **{'cluster_id': cluster_id})
                                    chx.units.append(unit)
                                unit.spiketrains.append(sptr)

            # TODO add duration
            # TODO May need to "populate_RecordingChannel"

        return blk

    def _get_channel_index(self, group):
        name = group.name
        assert 'channel_group_' in name # TODO assert that there actually is a number after _
        group_id = int(name[-1])
        index = group.attrs['electrode_idx']
        chx = ChannelIndex(index=index,
                           name='group_id #{}'.format(group_id),
                           **{'group_id': group_id})
        return chx

    def read_analogsignals(self, group):
        if group.name.split('/')[-1] == 'LFP':
            pass
        elif 'LFP' in group.keys():
            group = group['LFP']
        else:
            return None
        for key, value in group.items():
            signal = value["data"]
            analogsignal = AnalogSignal(
                signal.data,
                units=signal.attrs["unit"],
                sampling_rate=pq.Quantity(
                    value.attrs["sample_rate"]["value"],
                    value.attrs["sample_rate"]["unit"]
                )
            )

            analogsignals.append(analogsignal)

            # TODO: what about channel index
            # TODO: read attrs?

        return analogsignals

    def read_spiketrains(self, group):
        if "EventWaveform" == op.split(group.name)[-1]:
            return self.read_event_waveform(group)
        elif 'UnitTimes' == op.split(group.name)[-1]:
            return self.read_unit_times(group)
        else:
            raise ValueError('group name {} '.format(group.name) +
                             'is not recognized, the deepest folder should' +
                             ' be either "EventWaveform" or "UnitTimes"')

    def read_event_waveform(self, group):
        if group.name.split('/')[-1] == 'EventWaveform':
            pass
        elif 'EventWaveform' in group.keys():
            group = group['EventWaveform']
        else:
            return None
        spike_trains = []
        clustering_name = '/'.join(group.name.split('/')[:-1]) + '/Clustering'
        clustering = self._exdir_folder[clustering_name]
        for key, value in group.items():
            clusters = clustering["nums"].data
            for cluster in np.unique(clusters):
                indices, = np.where(clusters == cluster)
                metadata = {'cluster_id': cluster}
                metadata.update(value["timestamps"].attrs)
                metadata.update(value["waveforms"].attrs)
                metadata.update(group.attrs)
                # TODO if groups in clustering get them
                spike_train = SpikeTrain(
                    times=get_quantity(value["timestamps"].data[indices],
                                       value["timestamps"]),
                    t_stop=get_quantity_attr(value, 'stop_time'),
                    t_start=get_quantity_attr(value, 'start_time'),
                    waveforms=get_quantity(value["waveforms"].data[indices, :, :],
                                           value["waveforms"]),
                    sampling_rate=get_quantity_attr(value["waveforms"],
                                                    'sample_rate'),
                    **metadata
                    )
                spike_trains.append(spike_train)
            # TODO: read attrs?
        return spike_trains

    def read_unit_times(self, group):
        if group.name.split('/')[-1] == 'UnitTimes':
            pass
        elif 'UnitTimes' in group.keys():
            group = group['UnitTimes']
        else:
            return None
        spike_trains = []
        for value in group.values():
            spike_train = SpikeTrain(
                times=get_quantity(value.data[indices], timestamps),
                t_stop=get_quantity_attr(value, 'stop_time'),
                t_start=get_quantity_attr(value, 'start_time'),
                **{'cluster_id': cluster}
                )
            spike_trains.append(spike_train)
            # TODO: read attrs?
        return spike_trains

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
