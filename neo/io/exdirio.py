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


def get_quantity_attr(exdir_object, key):
    return pq.Quantity(exdir_object.attrs[key]['value'],
                       exdir_object.attrs[key]['unit'])


def get_quantity(data, exdir_object):
    return pq.Quantity(data, exdir_object.attrs["unit"])


class ExdirIO(BaseIO):
    """
    Class for reading/writting of exdir fromat
    """

    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, AnalogSignal, ChannelIndex, SpikeTrain]
    readable_objects = [Block, SpikeTrain]
    writeable_objects = []

    has_header = False
    is_streameable = False

    name = 'exdir'
    description = 'This IO reads experimental data from an exdir folder'
    extensions = ['exdir']
    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # this info is for GUI stuff also
    mode = 'dir'

    def __init__(self, dirname, mode='a'):
        """
        Arguments:
            folder_path : the folder path
        """
        BaseIO.__init__(self)
        self._absolute_folder_path = dirname
        self._path, relative_folder_path = os.path.split(dirname)
        self._base_folder, extension = os.path.splitext(relative_folder_path)

        if extension != ".exdir":
            raise ValueError("folder extension must be '.exdir'")

        self._exdir_folder = exdir.File(folder=dirname, mode=mode)

        # TODO check if group exists
        self._processing = self._exdir_folder.require_group("processing")
        self._epochs = self._exdir_folder.require_group("epochs")

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
                             sampling_rate, channel_group, t_start, t_stop,
                             channel_ids):
        event_wf_group = channel_group.require_group('EventWaveform')
        wf_group = event_wf_group.require_group('waveform_timeseries')
        wf_group.attrs['start_time'] = t_start
        wf_group.attrs['stop_time'] = t_stop
        wf_group.attrs['electrode_idx'] = channel_indexes
        wf_group.attrs['electrode_identities'] = channel_ids
        ts_data = wf_group.require_dataset("timestamps", spike_times)
        wf = wf_group.require_dataset("waveforms", waveforms)
        wf.attrs['sample_rate'] = sampling_rate

    def _save_clusters(self, spike_clusters, channel_group, t_start,
                       t_stop, cluster_groups):
        cl_group = channel_group.require_group('Clustering')
        cl_group.attrs['start_time'] = t_start
        cl_group.attrs['stop_time'] = t_stop
        cluster_nums = np.unique(spike_clusters)
        cl_group.attrs['cluster_groups'] = cluster_groups
        cl_data = cl_group.require_dataset('cluster_nums', cluster_nums)
        cl_data = cl_group.require_dataset('nums', spike_clusters)

    def _save_unit_times(self, sptrs, channel_group, t_start, t_stop):
        unit_times_group = channel_group.require_group('UnitTimes')
        unit_times_group.attrs['start_time'] = t_start
        unit_times_group.attrs['stop_time'] = t_stop
        for idx, sptr in enumerate(sptrs):
            if 'cluster_id' in sptr.annotations:
                sptr_id = sptr.annotations['cluster_id']
            else:
                sptr_id = idx
            times_group = unit_times_group.require_group('{}'.format(sptr_id))
            for key, val in sptr.annotations.items():
                times_group.attrs[key] = val
            ts_data = times_group.require_dataset('times', sptr.times)

    def _save_LFP(self, anas, channel_group, channel_indexes):
        lfp_group = channel_group.require_group('LFP')
        lfp_subgroup = lfp_group.require_group('LFP timeseries')
        lfp_subgroup.attrs['electrode_idx'] = channel_indexes
        for idx, ana in enumerate(anas):
            lfp_data = lfp_subgroup.require_dataset('data', ana)
            lfp_data.attrs['annotations'] = ana.annotations

    def _save_epochs(self, epochs, t_start, t_stop, group):
        for epo_num, epo in enumerate(epochs):
            epo_group = group.require_group('Epoch_{}'.format(epo_num))
            epo_group.attrs['start_time'] = t_start
            epo_group.attrs['stop_time'] = t_stop
            data_group = epo_group.require_group('data')
            data_group.require_dataset('timestamps', epo.times)
            data_group.require_dataset('durations', epo.durations)
            data_group.require_dataset('data', epo.labels)
            data_group.attrs['annotations'] = epo.annotations

    def save(self, blk):
        # TODO save block annotations
        # TODO add clustering version, type, algorithm etc.
        if not all(['group_id' in chx.annotations
                    for chx in blk.channel_indexes]):
            print('Warning "group_id" is not in channel_index.annotations ' +
                  'indexing group_id as appended to Block.channel_indexes')
            channel_indexes = {i: chx for i, chx in
                               enumerate(blk.channel_indexes)}
        else:
            channel_indexes = {int(chx.annotations['group_id']): chx
                               for chx in blk.channel_indexes}
        for seg in blk.segments:

            seg_name = 'segment_{}'.format(seg.index)
            self._save_epochs([epo for epo in seg.epochs], seg.t_start, seg.t_stop,
                              self._epochs)

            for group_id, chx in channel_indexes.items():
                grp = chx.annotations['group_id'] or group_id
                grp_name = 'channel_group_{}'.format(grp)
                ch_group = self._processing.require_group(grp_name+'_'+seg_name)
                ch_group.attrs['electrode_idx'] = chx.index
                ch_group.attrs['electrode_identities'] = chx.channel_ids
                ch_group.attrs['electrode_group_id'] = group_id
                ch_group.attrs['segment_id'] = seg.index
                for key, val in seg.annotations.items():
                    seg_group.attrs[key] = val
                for key, val in chx.annotations.items():
                    ch_group.attrs[key] = val
                sptrs = [st for st in seg.spiketrains
                         if st.channel_index == chx]
                sampling_rate = sptrs[0].sampling_rate.rescale('Hz')
                spike_times = self._sptrs_to_times(sptrs)
                ns, = spike_times.shape
                num_chans = len(chx.index)
                waveforms = self._sptrs_to_wfseriesf(sptrs)
                assert waveforms.shape[:2] == (ns, num_chans)
                self._save_event_waveform(spike_times, waveforms, chx.index,
                                          sampling_rate, ch_group, seg.t_start,
                                          seg.t_stop, chx.channel_ids)

                spike_clusters = self._sptrs_to_spike_clusters(sptrs)
                assert spike_clusters.shape == (ns,)
                if 'group' in chx.annotations:
                    cluster_groups = chx.annotations['group']
                else:
                    cluster_groups = {int(cl): 'Unsorted'
                                      for cl in np.unique(spike_clusters)}
                self._save_clusters(spike_clusters, ch_group, seg.t_start, seg.t_stop,
                                    cluster_groups)

                self._save_unit_times(sptrs, ch_group, seg.t_start, seg.t_stop)

                anas = [ana for ana in chx.analogsignals]
                if len(anas) > 0:
                    self._save_LFP(anas, ch_group, chx.index)

    def _read_segments_channel_indexes(self):
        self._segments = {}
        self._channel_indexes = {}
        for prc_name, prc_group in self._processing.items(): # TODO maybe look deeper as well?
            if 'segment_id' in prc_group.attrs:
                idx = prc_group.attrs['segment_id']
                if not idx in self._segments:
                    seg = Segment(name='Segment {}'.format(idx),
                                  index=idx)
                    t_start = get_quantity_attr(prc_group, 'start_time')
                    t_stop = get_quantity_attr(prc_group, 'stop_time')
                    seg.duration = t_stop - t_start
                    self._segments[idx] = seg
            if 'electrode_group_id' in prc_group.attrs:
                idx = prc_group.attrs['electrode_group_id']
                if not idx in self._channel_indexes:
                    chx = ChannelIndex(name='Channel group {}'.format(idx),
                                       index=prc_group.attrs['electrode_idx'],
                                       channel_ids=prc_group.attrs['electrode_identities'],
                                       **{'group_id': prc_group.attrs['electrode_group_id']})
                    self._channel_indexes[idx] = chx
        return self._segments, self._channel_indexes


    def read_block(self,
                   lazy=False,
                   cascade=True):
        '''

        '''
        # TODO read_block with annotations
        blk = Block(file_origin=self._absolute_folder_path)
        if cascade:
            # for prc_name, prc_group in self._epochs.items():
                # self.read_epochs(group=prc_sub_group)
            if not hasattr(self, '_segments'):
                self._read_segments_channel_indexes()
            blk.segments.extend(list(self._segments.values()))
            blk.channel_indexes.extend(list(self._channel_indexes.values()))

            for prc_name, prc_group in self._processing.items():
                self.read_tracking(group=prc_group)
                self.read_analogsignals(group=prc_group)

                spike_trains = self.read_event_waveforms(group=prc_group)
                if spike_trains is None:
                    spike_trains = self.read_unit_times(group=prc_group)


            for chx_id, chx in self._channel_indexes.items():
                clusters = [(sptr.annotations['cluster_id'], sptr)
                            for seg in self._segments.values()
                            for sptr in seg.spiketrains
                            if sptr.channel_index==chx]
                cluster_ids = np.unique([int(cid) for cid, _ in clusters])
                for cluster_id in cluster_ids:
                    unit = Unit(name='Cluster #{}'.format(cluster_id),
                                **{'cluster_id': cluster_id})
                    sptrs = [sptr for cid, sptr in clusters if cid==cluster_id]
                    if len(sptrs) > 1:
                        assert all(sptr1.channel_index == sptr2.channel_index
                                   for sptr1 in sptrs for sptr2 in sptrs)
                    chx = sptrs[0].channel_index
                    unit.spiketrains.extend(sptrs)
                    unit.channel_index = chx
                    chx.units.append(unit)


            # TODO May need to "populate_RecordingChannel"

        return blk

    def read_analogsignals(self, group):
        group = self._find_my_group(group, 'LFP')
        if group is None:
            return None
        if not hasattr(self, '_segments'):
            self._read_segments_channel_indexes()
        analogsignals = []
        for lfp_grp in group.values():
            signal = lfp_grp["data"]
            ana = AnalogSignal(
                signal.data,
                units=signal.attrs["unit"],
                sampling_rate=get_quantity_attr(lfp_grp, 'sample_rate')
                )
            self._segments[lfp_grp.attrs['segment_id']].analogsignals.append(ana)
            chx = self._channel_indexes[lfp_grp.attrs['electrode_group_id']]
            chx.analogsignals.append(ana)
            ana.channel_index = chx
            analogsignals.append(ana)

            # TODO: read attrs?

        return analogsignals

    def read_spiketrains(self, group):
        if "EventWaveform" == op.split(group.name)[-1]:
            return self.read_event_waveforms(group)
        elif 'UnitTimes' == op.split(group.name)[-1]:
            return self.read_unit_times(group)
        else:
            raise ValueError('group name {} '.format(group.name) +
                             'is not recognized, the deepest folder should' +
                             ' be either "EventWaveform" or "UnitTimes"')

    def read_event_waveforms(self, group):
        group = self._find_my_group(group, 'EventWaveform')
        if group is None:
            return None
        if not hasattr(self, '_segments'):
            self._read_segments_channel_indexes()
        spike_trains = []
        container_name = '/'.join(group.name.split('/')[:-1])
        container_group = self._exdir_folder[container_name]
        if 'Clustering' in container_group:
            clustering = container_group['Clustering']
            spike_clusters = np.array(clustering["nums"].data, dtype=int)
            cluster_groups = clustering.attrs['cluster_groups']
        else:
            spike_clusters = np.zeros(group.attrs['num_samples'], dtype=int)
            cluster_groups = {0:'unsorted'}
        for cluster in np.unique(spike_clusters):
            indices, = np.where(spike_clusters == cluster)
            metadata = {'cluster_id': cluster,
                        'cluster_group': cluster_groups[cluster]} # TODO add clustering version, type, algorithm etc.
            sptr = SpikeTrain(
                times=get_quantity(group["timestamps"].data[indices],
                                   group["timestamps"]),
                t_stop=get_quantity_attr(group, 'stop_time'),
                t_start=get_quantity_attr(group, 'start_time'),
                waveforms=get_quantity(group["data"].data[indices, :, :],
                                       group["data"]),
                sampling_rate=get_quantity_attr(group["data"], 'sample_rate'),
                **metadata
                )
            spike_trains.append(sptr)
            self._segments[group.attrs['segment_id']].spiketrains.append(sptr)
            chx = self._channel_indexes[group.attrs['electrode_group_id']]
            sptr.channel_index = chx
        return spike_trains

    def read_unit_times(self, group):
        group = self._find_my_group(group, 'UnitTimes')
        if group is None:
            return None
        if not hasattr(self, '_segments'):
            self._read_segments_channel_indexes()
        spike_trains = []
        for un_ti_grp in group.values():
            sptr = SpikeTrain(
                times=get_quantity(un_ti_grp.data[indices], timestamps),
                t_stop=get_quantity_attr(un_ti_grp, 'stop_time'),
                t_start=get_quantity_attr(un_ti_grp, 'start_time'),
                **{'cluster_id': un_ti_grp.attrs['cluster_id'],
                   'cluster_group': un_ti_grp.attrs['cluster_group'], # TODO add clustering version, type, algorithm etc.
                   }
                )
            spike_trains.append(sptr)
            self._segments[un_ti_grp.attrs['segment_id']].spiketrains.append(sptr)
            chx = self._channel_indexes[un_ti_grp.attrs['electrode_group_id']]
            chx.spiketrains.append(sptr)
            sptr.channel_index = chx
        return spike_trains

    # def read_epochs(self, group):
    #     epos_group = group.require_group('epochs')
    #     epos = []
    #     for epo_num, epo in enumerate(epochs):
    #         epo_group.require_group('Epoch_{}'.format(epo_num))
    #         epo_group.attrs['start_time'] = t_start
    #         epo_group.attrs['stop_time'] = t_stop
    #         data_group = epo_group.require_group('data')
    #         epo
    #         data_group['timestamps'], epo.times)
    #         data_group['durations'], epo.durations)
    #         data_group['data'], epo.labels)
    #         data_group['annotations'] = epo.annotations

    def read_tracking(self, group):
        """
        Read tracking data_end
        """
        group = self._find_my_group(group, 'Position')
        if group is None:
            return None
        if not hasattr(self, '_segments'):
            self._read_segments_channel_indexes()
        irr_signals = []
        for spot_group in group.values():
            times = spot_group["timestamps"]
            coords = spot_group["data"]
            irrsig = IrregularlySampledSignal(name=spot_group.name.split('/')[-1],
                                              signal=coords.data,
                                              times=times.data,
                                              units=coords.attrs["unit"],
                                              time_units=times.attrs["unit"],
                                              file_origin=spot_group.folder)
            irr_signals.append(irrsig)
            self._segments[spot_group.attrs['segment_id']].irregularlysampledsignals.append(irrsig)

        return irr_signals

    def _find_my_group(self, group, name):
        if group.name.split('/')[-1] == name:
            pass
        elif name in group.keys():
            group = group[name]
        else:
            return None
        return group

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
