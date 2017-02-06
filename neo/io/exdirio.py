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
import shutil

python_version = sys.version_info.major
if python_version == 2:
    from future.builtins import str


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
    description = 'This IO reads experimental data from an exdir directory'
    extensions = ['exdir']
    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # thinfo is for GUI stuff also
    mode = 'dir'

    def __init__(self, dirname, mode='a'):
        """
        Arguments:
            directory_path : the directory path
        """
        BaseIO.__init__(self)
        self._absolute_directory_path = dirname
        self._path, relative_directory_path = os.path.split(dirname)
        self._base_directory, extension = os.path.splitext(relative_directory_path)

        if extension != ".exdir":
            raise ValueError("directory extension must be '.exdir'")

        self._exdir_directory = exdir.File(directory=dirname, mode=mode)

        # TODO check if group exists
        self._processing = self._exdir_directory.require_group("processing")
        self._epochs = self._exdir_directory.require_group("epochs")

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
                           [sptr.annotations['cluster_id']] * len(sptr)])
        else:
            out = np.array([i for j, sptr in enumerate(sptrs)
                            for i in [j] * len(sptr)])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out))

    def write_event_waveform(self, spike_times, waveforms, sampling_rate,
                             exdir_group, **annotations):
        group = exdir_group.require_group('EventWaveform')
        wf_group = group.require_group('waveform_timeseries')
        attr = {'num_samples': len(spike_times),
                'sample_length': waveforms.shape[1]}
        attr.update(annotations)
        wf_group.attrs = attr
        ts_data = wf_group.require_dataset("timestamps", spike_times)
        wf = wf_group.require_dataset("data", waveforms)
        wf.attrs['sample_rate'] = sampling_rate

    def write_clusters(self, spike_clusters, spike_times, exdir_group,
                      **annotations):
        cl_group = exdir_group.require_group('Clustering')
        if annotations:
            cl_group.attrs = annotations
        cluster_nums = np.unique(spike_clusters)
        cl_group.require_dataset('cluster_nums', cluster_nums)
        cl_group.require_dataset('nums', spike_clusters)
        cl_group.require_dataset('timestamps', spike_times)

    def write_unit_times(self, sptrs, exdir_group, **annotations):
        if 'UnitTimes' in exdir_group:
            shutil.rmtree(exdir_group['UnitTimes'].directory)
        unit_times_group = exdir_group.require_group('UnitTimes')
        if annotations:
            unit_times_group.attrs = annotations
        for idx, sptr in enumerate(sptrs):
            if 'cluster_id' in sptr.annotations:
                sptr_id = sptr.annotations['cluster_id']
            else:
                sptr_id = idx
            times_group = unit_times_group.require_group('{}'.format(sptr_id))
            times_group.attrs = sptr.annotations
            ts_data = times_group.require_dataset('times', sptr.times)

    def write_LFP(self, anas, exdir_group, **annotations):
        lfp_group = exdir_group.require_group('LFP')
        if annotations:
            lfp_group.attrs = annotations
        for idx, ana in enumerate(anas):
            lfp_data = lfp_group.require_dataset('data', ana) # TODO sampling_rate etc
            lfp_data.attrs = ana.annotations

    def write_epochs(self, epochs, group, **annotations):
        for epo_num, epo in enumerate(epochs):
            if epo.name is None:
                epo_group = group.require_group('Epoch_{}'.format(epo_num))
            else:
                epo_group = group.require_group(epo.name)
            if annotations:
                epo_group.attrs = annotations
            data_group = epo_group.require_group('data')
            data_group.require_dataset('timestamps', epo.times)
            data_group.require_dataset('durations', epo.durations)
            data_group.require_dataset('data', epo.labels)
            data_group.attrs = epo.annotations

    def write_block(self, blk):
        # TODO save block annotations
        # TODO add clustering version, type, algorithm etc.
        # TODO save stuff even if no channel_indexes?
        # TODO save segment as epoch? with segment annotations
        if any(ana for seg in blk.segments for ana in seg.analogsignals):
            print('Warning: saving analogsignals is not supported by this' +
                  'funtion, use write_LFP in stead')

        if not all(['group_id' in chx.annotations
                    for chx in blk.channel_indexes]):
            print('Warning "group_id" is not in channel_index.annotations ' +
                  'indexing group_id as appended to Block.channel_indexes')
            channel_indexes = {i: chx for i, chx in
                               enumerate(blk.channel_indexes)}
        else:
            channel_indexes = {int(chx.annotations['group_id']): chx
                               for chx in blk.channel_indexes}
        if len(blk.segments) > 1:
            raise NotImplementedError('sorry, exdir supports only one segment')
        seg = blk.segments[0]
        self._exdir_directory.attrs['session_duration'] = seg.duration
        if len(seg.epochs) > 0:
            self.write_epochs([epo for epo in seg.epochs], self._epochs,
                              t_start=seg.t_start, t_stop=seg.t_stop)
        elphys = self._processing.require_group('electrophysiology') # TODO find right place without name
        for group_id, chx in channel_indexes.items():
            grp_name = 'channel_group_{}'.format(group_id)
            exdir_group = elphys.require_group(grp_name)
            annotations = {'electrode_idx': chx.index,
                           'electrode_group_id': group_id,
                           'start_time': seg.t_start,
                           'stop_time': seg.t_stop,
                           'electrode_identities': chx.channel_ids}
            exdir_group.attrs = annotations
            sptrs = [st for st in seg.spiketrains
                     if st.channel_index == chx]
            sampling_rate = sptrs[0].sampling_rate.rescale('Hz')
            spike_times = self._sptrs_to_times(sptrs)
            ns, = spike_times.shape
            num_chans = len(chx.index)
            waveforms = self._sptrs_to_wfseriesf(sptrs)
            assert waveforms.shape[:2] == (ns, num_chans)
            self.write_event_waveform(spike_times, waveforms, sampling_rate,
                                      exdir_group, **annotations)
            self.write_unit_times(sptrs, exdir_group, **annotations)

            spike_clusters = self._sptrs_to_spike_clusters(sptrs)
            assert spike_clusters.shape == (ns,)
            if 'group' in chx.annotations:
                cluster_groups = chx.annotations['group']
            else:
                cluster_groups = {int(cl): 'Unsorted'
                                  for cl in np.unique(spike_clusters)}
            annotations.update({'cluster_groups': cluster_groups})
            self.write_clusters(spike_clusters, spike_times, exdir_group,
                                **annotations)

    def _get_channel_indexes(self, processing):
        channel_indexes = dict()
        assert 'electrophysiology' in processing
        for sub_processing in processing['electrophysiology'].values():
            if 'electrode_group_id' in sub_processing.attrs:
                idx = sub_processing.attrs['electrode_group_id']
                if idx not in channel_indexes:
                    chx = ChannelIndex(name='Channel group {}'.format(idx),
                                       index=sub_processing.attrs['electrode_idx'],
                                       channel_ids=sub_processing.attrs['electrode_identities'],
                                       **{'group_id': sub_processing.attrs['electrode_group_id']})
                    channel_indexes[idx] = chx
        return channel_indexes

    def read_block(self,
                   lazy=False,
                   cascade=True,
                   cluster_group='all',
                   read_waveforms=True):
        '''

        '''
        blk = Block(file_origin=self._absolute_directory_path,
                    **self._exdir_directory.attrs)
        seg = self.read_segment(cascade=cascade, lazy=lazy,
                                read_waveforms=read_waveforms)
        if not hasattr(self, '_channel_indexes'):
            self._channel_indexes = self._get_channel_indexes(self._processing)
        blk.channel_indexes.extend(list(self._channel_indexes.values()))
        blk.segments.append(seg)
        return blk

    def read_segment(self, cascade=True, lazy=False, read_waveforms=True):
        seg = Segment(name='Segment #0', index=0)
        seg.duration = self._exdir_directory.attrs['session_duration']
        if cascade:
            for group in self._epochs.values():
                epo = self.read_epoch(group.name, cascade, lazy)
                seg.epochs.append(epo)

            for prc_group in self._processing.values():
                for sub_group in prc_group.values():
                    if 'LFP' in sub_group:
                        for lfp_group in sub_group['LFP'].values():
                            ana = self.read_analogsignal(lfp_group.name,
                                                         cascade=cascade,
                                                         lazy=lazy)
                            seg.analogsignals.append(ana)
                            # chx = self._channel_indexes[lfp_group.attrs['electrode_group_id']]
                            # chx.analogsignals.append(ana)
                            # ana.channel_index = chx
                    if 'UnitTimes' in sub_group:
                        for unit_group in sub_group['UnitTimes'].values():
                            sptr = self.read_spiketrain(
                                unit_group.name,
                                cascade=cascade,
                                lazy=lazy,
                                read_waveforms=read_waveforms
                            )
                            seg.spiketrains.append(sptr)
                    elif 'EventWaveform' in sub_group:
                        sptr = self.read_spiketrain(
                            sub_group['UnitTimes'].name,
                            cascade=cascade,
                            lazy=lazy,
                            read_waveforms=read_waveforms
                        )
                        seg.spiketrains.append(sptr)
        return seg
        
    def read_channelindex():
        for chx_id, chx in self._channel_indexes.items():
            sptrs = [sptr for sptr in seg.spiketrains
                     if sptr.channel_index == chx]
            for sptr in sptrs:
                cluster_id = sptr.annotations['cluster_id']
                unit = Unit(name='Cluster #{}'.format(cluster_id),
                            **{'cluster_id': cluster_id})
                unit.spiketrains.append(sptr)
                unit.channel_index = chx
                chx.units.append(unit)

    def read_epoch(self, path, cascade=True, lazy=False):
        group = self._exdir_directory[path]
        times = pq.Quantity(group['timestamps'].data,
                            group['timestamps'].attrs['unit'])
        durations = pq.Quantity(group['durations'].data,
                                group['durations'].attrs['unit'])
        if 'data' in group:
            if 'unit' in group['data'].attrs:
                labels = group['data'].data
            else:
                labels = pq.Quantity(group['data'].data,
                                     group['data'].attrs['unit'])
        else:
            labels = None
        annotations = group.attrs._open_or_create() #HACK TODO make a function in
        epo = Epoch(times=times, durations=durations, labels=labels,
                    name=group.object_name, **annotations)

        return epo

    def read_analogsignal(self, path, cascade=True, lazy=False):
        group = self._exdir_directory[path]
        signal = group["data"]
        ana = AnalogSignal(signal.data,
                           units=signal.attrs["unit"],
                           sampling_rate=group.attrs['sample_rate'],
                           **{'channel_index': group.attrs['electrode_idx'],
                              'channel_identity': group.attrs['electrode_identity']})
        return ana

    def read_spiketrains(self, group, cluster_group='all'):
        group_name = group.object_name
        if "EventWaveform" == group_name:
            return self.read_event_waveforms(group, cluster_group=cluster_group)
        elif 'UnitTimes' == group_name:
            return self.read_unit_times(group, cluster_group=cluster_group)
        else:
            raise ValueError('group name {} '.format(group_name) +
                             'is not recognized, the deepest directory should' +
                             ' be either "EventWaveform" or "UnitTimes"')

    def read_event_waveforms(self, group, cluster_group='all'):
        spike_trains = []
        container_name = '/'.join(group.name.split('/')[:-2])
        container_group = self._exdir_directory[container_name]
        if 'Clustering' in container_group:
            clustering = container_group['Clustering']
            spike_clusters = np.array(clustering["nums"].data, dtype=int)
            cluster_groups = clustering.attrs['cluster_groups']
        else:
            spike_clusters = np.zeros(group.attrs['num_samples'], dtype=int)
            cluster_groups = {0: 'unsorted'}
        for cluster in np.unique(spike_clusters):
            if cluster_groups[cluster] is None:
                print('Warning: found cluster group None ', cluster)
            if cluster_group != 'all':
                if cluster_groups[cluster] is not None:
                    if cluster_groups[cluster].lower() != cluster_group.lower():
                        continue
            indices, = np.where(spike_clusters == cluster)
            metadata = {'cluster_id': cluster,
                        'cluster_group': cluster_groups[cluster]} # TODO add clustering version, type, algorithm etc.
            if 'unit' in group["data"].attrs:
                waveforms = pq.Quantity(group["data"].data[indices, :, :],
                                        group["data"].attrs['unit'])
            else:
                waveforms = group["data"].data[indices, :, :]
            sptr = SpikeTrain(
                times=pq.Quantity(group["timestamps"].data[indices],
                                   group["timestamps"].attrs['unit']),
                t_stop=group.attrs['stop_time'],
                t_start=group.attrs['start_time'],
                waveforms=waveforms,
                sampling_rate=group["data"].attrs['sample_rate'],
                **metadata
                )
            spike_trains.append(sptr)
            chx = self._channel_indexes[group.attrs['electrode_group_id']]
            sptr.channel_index = chx
        return spike_trains

    def read_unit_times(self, group, cluster_group='all'):
        spike_trains = []
        for un_ti_grp in group.values():
            sptr = SpikeTrain(
                times=pq.Quantity(un_ti_grp.data[indices], un_ti_grp.attrs['unit']),
                t_stop=un_ti_grp.attrs['stop_time'],
                t_start=un_ti_grp.attrs['start_time'],
                **un_ti_grp.attrs
            )
            spike_trains.append(sptr)
            chx.spiketrains.append(sptr)
            sptr.channel_index = chx
        return spike_trains

    def read_spiketrain(self, path, cascade=True, lazy=False, cluster_num=None,
                        read_waveforms=True):
        group = self._exdir_directory[path]
        metadata = {}
        if group.parent.object_name == 'UnitTimes':
            times = pq.Quantity(group['times'].data,
                                group['times'].attrs['unit'])
            wf_group = group.parent.parent['EventWaveform']
            cluster_group = group.parent.parent['Clustering']
            # assert cluster_num == int(group.object_name)
            cluster_ids = cluster_group['nums'].data
            indices, = np.where(cluster_ids == cluster_num)
            metadata.update(group.attrs)
            t_stop = group.parent.attrs['stop_time']
            t_start = group.parent.attrs['start_time']
        elif group.object_name == 'EventWaveform':
            sub_group = group.values()[0]
            # TODO assert all timestamps to be equal if several waveform_timeseries exists
            wf_group = group
            tmp_path = '/'.join(group.name.split('/')[:-2])
            tmp_group = self._exdir_directory[tmp_path]
            if 'Clustering' in tmp_group:
                cluster_group = tmp_group['Clustering']
                cluster_ids = cluster_group['num'].data
                indices, = np.where(cluster_ids == cluster_num)
                times = pq.Quantity(sub_group["timestamps"].data[indices],
                                    sub_group["timestamps"].attrs['unit'])
            else:
                times = pq.Quantity(sub_group["timestamps"].data,
                                    sub_group["timestamps"].attrs['unit'])
                indices = range(len(times))
            t_stop = sub_group.attrs['stop_time']
            t_start = sub_group.attrs['start_time']
            metadata.update(sub_group.attrs)
        else:
            raise ValueError('Expected a sub group of UnitTimes or an ' +
                             'EventWaveform group')
        if read_waveforms:
            waveforms = []
            for wf in wf_group.values():
                data = pq.Quantity(wf["data"].data[indices, :, :],
                                   wf["data"].attrs['unit'])
                waveforms.append(data)
                metadata.update(wf.attrs)
            waveforms = np.vstack(waveforms)
            # TODO assert shape of waveforms relative to channel_ids etc
            sampling_rate = wf["data"].attrs['sample_rate']
        else:
            waveforms = None
            sampling_rate = None

        sptr = SpikeTrain(times=times,
                          t_stop=t_stop,
                          t_start=t_start,
                          waveforms=waveforms,
                          sampling_rate=sampling_rate,
                          **metadata)
        return sptr
