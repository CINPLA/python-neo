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
import numpy as np
import quantities as pq
import os
import os.path as op
import glob
import yaml
import copy
import shutil

python_version = sys.version_info.major
if python_version == 2:
    from future.builtins import str

try:
    import exdir
    HAVE_EXDIR = True
except ImportError:
    HAVE_EXDIR = False

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
            sptr_annot = sptr.annotations
            sptr_annot.update({'name': sptr.name,
                               'description': sptr.description})
            times_group.attrs = sptr_annot
            ts_data = times_group.require_dataset('times', sptr.times)

    def write_LFP(self, anas, exdir_group, **annotations):
        group = exdir_group.require_group('LFP')
        for idx, ana in enumerate(anas):
            lfp_group = group.require_group('lfp_timeseries_{}'.format(idx))
            attrs = ana.annotations
            attrs.update(annotations)
            attrs.update({'name': ana.name, 'description': ana.description})
            lfp_group.attrs = attrs
            lfp_data = lfp_group.require_dataset('data', ana) # TODO sampling_rate etc
            lfp_data.attrs['sample_rate'] = ana.sampling_rate

    def write_epochs(self, epochs, group, **annotations):
        for epo_num, epo in enumerate(epochs):
            if epo.name is None:
                epo_group = group.require_group('Epoch_{}'.format(epo_num))
            else:
                epo_group = group.require_group(epo.name)
            if annotations:
                epo_group.attrs = annotations
            epo_group.require_dataset('timestamps', epo.times)
            epo_group.require_dataset('durations', epo.durations)
            epo_group.require_dataset('data', epo.labels)
            epo_group.attrs = epo.annotations

    def write_block(self, blk):
        # TODO save stuff even if no channel_indexes?
        if any(isinstance(ana, AnalogSignal) for seg in blk.segments
               for ana in seg.analogsignals):
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
        annotations = blk.annotations
        annotations.update({'session_duration': seg.t_stop - seg.t_start})
        self._exdir_directory.attrs = annotations
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

    def read_block(self,
                   lazy=False,
                   cascade=True,
                   cluster_group='all',
                   read_waveforms=True,
                   elphys_directory_name='electrophysiology'):
        '''

        '''
        blk = Block(file_origin=self._absolute_directory_path,
                    **self._exdir_directory.attrs._open_or_create())
        seg = Segment(name='Segment #0', index=0)
        blk.segments.append(seg)
        if cascade:
            for group in self._epochs.values():
                epo = self.read_epoch(group.name, cascade, lazy)
                seg.epochs.append(epo)
            for channel_group in self._processing[elphys_directory_name].values():
                group_id = channel_group.attrs['electrode_group_id']
                chx = ChannelIndex(
                    name='Channel group {}'.format(group_id),
                    index=channel_group.attrs['electrode_idx'],
                    channel_ids=channel_group.attrs['electrode_identities'],
                    **{'group_id': group_id}
                )
                if 'LFP' in channel_group:
                    for lfp_group in channel_group['LFP'].values():
                        ana = self.read_analogsignal(lfp_group.name,
                                                     cascade=cascade,
                                                     lazy=lazy)
                        chx.analogsignals.append(ana)
                        ana.channel_index = chx
                        seg.analogsignals.append(ana)
                sptrs = []
                if 'UnitTimes' in channel_group:
                    for unit_group in channel_group['UnitTimes'].values():
                        sptr = self.read_spiketrain(
                            unit_group.name,
                            cascade=cascade,
                            lazy=lazy,
                            read_waveforms=read_waveforms
                        )
                        sptrs.append(sptr)
                elif 'EventWaveform' in channel_group:
                    sptr = self.read_spiketrain(
                        channel_group['EventWaveform'].name,
                        cascade=cascade,
                        lazy=lazy,
                        read_waveforms=read_waveforms
                    )
                    sptrs.append(sptr)
                for sptr in sptrs:
                    unit = Unit(name=sptr.name,
                                **sptr.annotations)
                    unit.spiketrains.append(sptr)
                    unit.channel_index = chx
                    sptr.channel_index = chx
                    chx.units.append(unit)
                    seg.spiketrains.append(sptr)
        return blk

    def read_segment(self, cascade=True, lazy=False, read_waveforms=True):
        seg = Segment(name='Segment #0', index=0)
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

                    sptr = None
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
                    if sptr is not None:
                        seg.spiketrains.append(sptr)
        return seg

    def read_channelindex(self, path, cascade=True, Lazy=False):
        channel_group = self._exdir_directory[path]
        group_id = channel_group.attrs['electrode_group_id']
        chx = ChannelIndex(
            name='Channel group {}'.format(group_id),
            index=channel_group.attrs['electrode_idx'],
            channel_ids=channel_group.attrs['electrode_identities'],
            **{'group_id': group_id}
        )
        if 'LFP' in channel_group:
            for lfp_group in channel_group['LFP'].values():
                ana = self.read_analogsignal(lfp_group.name,
                                             cascade=cascade,
                                             lazy=lazy)
                chx.analogsignals.append(ana)
                ana.channel_index = chx
        sptr = None
        if 'UnitTimes' in channel_group:
            for unit_group in channel_group['UnitTimes'].values():
                sptr = self.read_spiketrain(
                    unit_group.name,
                    cascade=cascade,
                    lazy=lazy,
                    read_waveforms=read_waveforms
                )
        elif 'EventWaveform' in channel_group:
            sptr = self.read_spiketrain(
                channel_group['UnitTimes'].name,
                cascade=cascade,
                lazy=lazy,
                read_waveforms=read_waveforms
            )
        if sptr is not None:
            cluster_id = sptr.annotations['cluster_id']
            unit = Unit(name='Cluster #{}'.format(cluster_id),
                        **{'cluster_id': cluster_id})
            unit.spiketrains.append(sptr)
            unit.channel_index = chx
            sptr.channel_index = chx
            chx.units.append(unit)
        return chx

    def read_epoch(self, path, cascade=True, lazy=False):
        group = self._exdir_directory[path]
        times = pq.Quantity(group['timestamps'].data,
                            group['timestamps'].attrs['unit'])
        durations = pq.Quantity(group['durations'].data,
                                group['durations'].attrs['unit'])
        if 'data' in group:
            if 'unit' not in group['data'].attrs:
                labels = group['data'].data
            else:
                labels = pq.Quantity(group['data'].data,
                                     group['data'].attrs['unit'])
        else:
            labels = None
        annotations = group.attrs._open_or_create()
        epo = Epoch(times=times, durations=durations, labels=labels,
                    name=group.object_name, **annotations)

        return epo

    def read_analogsignal(self, path, cascade=True, lazy=False):
        group = self._exdir_directory[path]
        signal = group["data"]
        ana = AnalogSignal(signal.data,
                           units=signal.attrs["unit"],
                           sampling_rate=signal.attrs['sample_rate'],
                           **group.attrs._open_or_create())
        return ana

    def read_spiketrain(self, path, cascade=True, lazy=False, cluster_num=None,
                        read_waveforms=True):
        group = self._exdir_directory[path]
        metadata = {}
        if group.parent.object_name == 'UnitTimes':
            times = pq.Quantity(group['times'].data,
                                group['times'].attrs['unit'])
            t_stop = group.parent.attrs['stop_time']
            t_start = group.parent.attrs['start_time']
            if read_waveforms:
                wf_group = group.parent.parent['EventWaveform']
                cluster_group = group.parent.parent['Clustering']
                cluster_num = cluster_num or int(group.object_name)
                cluster_ids = cluster_group['nums'].data
                indices, = np.where(cluster_ids == cluster_num)
                metadata.update(group.attrs._open_or_create())
        elif group.object_name == 'EventWaveform':
            sub_group = group.values()[0]
            # TODO assert all timestamps to be equal if several waveform_timeseries exists
            wf_group = group
            if 'Clustering' in group.parent:
                cluster_group = group.parent['Clustering']
                cluster_ids = cluster_group['num'].data
                if len(np.unique(cluster_ids)) > 1:
                    assert cluster_num is not None, 'You must set cluster_num'
                else:
                    cluster_num = cluster_num or int(np.unique(cluster_ids))
                indices, = np.where(cluster_ids == cluster_num)
                times = pq.Quantity(sub_group["timestamps"].data[indices],
                                    sub_group["timestamps"].attrs['unit'])
            else:
                times = pq.Quantity(sub_group["timestamps"].data,
                                    sub_group["timestamps"].attrs['unit'])
                indices = range(len(times))
            t_stop = sub_group.attrs['stop_time']
            t_start = sub_group.attrs['start_time']
            metadata.update(sub_group.attrs._open_or_create())
        else:
            raise ValueError('Expected a sub group of UnitTimes or an ' +
                             'EventWaveform group')
        if read_waveforms:
            waveforms = []
            for wf in wf_group.values():
                data = pq.Quantity(wf["data"].data[indices, :, :],
                                   wf["data"].attrs['unit'])
                waveforms.append(data)
                metadata.update(wf.attrs._open_or_create())
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
